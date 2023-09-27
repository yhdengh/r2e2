#include <atomic>
#include <chrono>
#include <deque>
#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include <pbrt/accelerators/cloudbvh.h>
#include <pbrt/core/geometry.h>
#include <pbrt/main.h>
#include <pbrt/raystate.h>

#include <concurrentqueue/blockingconcurrentqueue.h>
#include <concurrentqueue/concurrentqueue.h>

#include <nlohmann/json.hpp>

#include "util/eventloop.hh"
#include "util/exception.hh"
#include "util/timerfd.hh"

#include "commlib/commlib.hh"

using namespace std;
using namespace std::chrono;
using namespace locusta;
using json = nlohmann::json;

class LocustaWorker
{
private:
  struct Peer
  {
    StreamHandle stream;
    deque<pbrt::RayStatePtr> outgoing_rays {};

    string write_buffer {};
    string read_buffer {};

    Peer( StreamHandle&& s )
      : stream( std::move( s ) )
    {
    }
  };

  // handling peers
  EventLoop event_loop_ {};

  // Communication Library
  unique_ptr<CommunicationLibrary> comm_;
  map<int, Peer> peers_ {}; // treelet_id -> peer

  // scene data
  const uint32_t total_workers_;
  const uint32_t my_treelet_id_;
  const bool is_accumulator_ { my_treelet_id_ == total_workers_ - 1 };
  pbrt::SceneBase scene_;
  filesystem::path output_path_;
  shared_ptr<pbrt::CloudBVH> treelet_;

  vector<thread> threads_ {};

  moodycamel::BlockingConcurrentQueue<pbrt::RayStatePtr> input_rays_ {};
  moodycamel::BlockingConcurrentQueue<pbrt::RayStatePtr> output_rays_ {};
  moodycamel::BlockingConcurrentQueue<pbrt::RayStatePtr> output_samples_ {};

  atomic<size_t> input_rays_size_ { 0 };
  atomic<size_t> output_rays_size_ { 0 };
  atomic<size_t> output_samples_size_ { 0 };

  atomic<size_t> total_rays_sent_ { 0 };
  atomic<size_t> total_rays_processed_ { 0 };
  atomic<size_t> total_rays_received_ { 0 };

  atomic<bool> terminated { false };

  TimerFD status_timer_ { 1s, 1s };

  void generate_rays();

  void render_thread();
  void accumulation_thread();
  void status();

public:
  LocustaWorker( const string& scene_path,
                 const uint32_t treelet_id,
                 const int spp,
                 unique_ptr<CommunicationLibrary> comm,
                 const uint32_t total_workers )
    : comm_( std::move( comm ) )
    , total_workers_( total_workers )
    , my_treelet_id_( treelet_id )
    , scene_( scene_path, spp )
    , output_path_( scene_path )
    , treelet_( treelet_id < total_workers_ - 1 ? pbrt::LoadTreelet( scene_path, treelet_id ) : nullptr )
  {
    if ( total_workers_ != scene_.TreeletCount() + 1 ) {
      throw runtime_error( "treelet count mismatch" );
    }

    cerr << "Loaded treelet " << treelet_id << "." << endl;

    for ( uint32_t i = 0; i < total_workers_; i++ ) {
      if ( i != my_treelet_id_ ) {
        peers_.emplace( i, comm_->stream_open( i, 0 ) );
      }
    }

    cerr << "Connections established to all peers." << endl;

    if ( my_treelet_id_ == 0 ) {
      generate_rays();
      cout << "All rays generated." << endl;
    }

    // now setting up the main event loop
    event_loop_.set_fd_failure_callback( [] { cerr << "FD FAILURE CALLBACK" << endl; } );

    event_loop_.add_rule(
      "Status", Direction::In, status_timer_, bind( &LocustaWorker::status, this ), [] { return true; } );

    event_loop_.add_rule(
      "Output rays",
      [&] {
        pbrt::RayStatePtr ray;
        while ( output_rays_.try_dequeue( ray ) ) {
          output_rays_size_--;
          auto peer_it = peers_.find( ray->CurrentTreelet() );
          peer_it->second.outgoing_rays.emplace_back( std::move( ray ) );
        }
      },
      [this] { return output_rays_size_ > 0; } );

    event_loop_.add_rule(
      "Output samples",
      [&] {
        pbrt::RayStatePtr ray;
        while ( output_samples_.try_dequeue( ray ) ) {
          output_samples_size_--;

          auto peer_it = peers_.find( total_workers_ - 1 ); // last worker is the accumulator
          peer_it->second.outgoing_rays.emplace_back( std::move( ray ) );
        }
      },
      [this] { return output_samples_size_ > 0; } );

    for ( auto& [peer_id, peer] : peers_ ) {
      auto peer_it = peers_.find( peer_id );

      peer.stream->first.set_blocking( false );
      peer.stream->second.set_blocking( false );

      if ( peer.stream->first.fd_num() == peer.stream->second.fd_num() ) {
        event_loop_.add_rule(
          "SOCKET Peer " + to_string( peer_id ),
          peer.stream->first,
          [peer_it] { // IN callback
            string buffer( 4096, '\0' );
            auto data_len = peer_it->second.stream->first.read( { buffer } );
            peer_it->second.read_buffer.append( buffer, 0, data_len );
          },
          [] { return true; },
          [peer_it] { // OUT callback
            auto data_len = peer_it->second.stream->first.write( string_view { peer_it->second.write_buffer } );
            peer_it->second.write_buffer.erase( 0, data_len );
          },
          [peer_it] { return not peer_it->second.write_buffer.empty(); } );

      } else {
        // socket read and write events
        event_loop_.add_rule(
          "SOCKET Peer " + to_string( peer_id ),
          Direction::In,
          peer.stream->first,
          [peer_it] { // IN callback
            string buffer( 4096, '\0' );
            auto data_len = peer_it->second.stream->first.read( { buffer } );
            peer_it->second.read_buffer.append( buffer, 0, data_len );
          },
          [] { return true; } );

        event_loop_.add_rule(
          "SOCKET Peer " + to_string( peer_id ),
          Direction::Out,
          peer.stream->second,
          [peer_it] { // OUT callback
            auto data_len = peer_it->second.stream->second.write( string_view { peer_it->second.write_buffer } );
            peer_it->second.write_buffer.erase( 0, data_len );
          },
          [peer_it] { return not peer_it->second.write_buffer.empty(); } );
      }

      // incoming rays event
      event_loop_.add_rule(
        "RAY-IN Peer " + to_string( peer_id ),
        [this, peer_it] {
          auto& buffer = peer_it->second.read_buffer;
          const uint32_t len = *reinterpret_cast<const uint32_t*>( buffer.data() );
          pbrt::RayStatePtr ray = pbrt::RayState::Create();
          ray->Deserialize( buffer.data() + 4, len );
          buffer.erase( 0, len + 4 );

          this->input_rays_.enqueue( std::move( ray ) );
          input_rays_size_++;

          total_rays_received_++;
        },
        [peer_it] {
          /* is there a full ray to read? */
          auto& buffer = peer_it->second.read_buffer;
          return not buffer.empty() and buffer.length() >= 4
                 and *reinterpret_cast<const uint32_t*>( buffer.data() ) <= buffer.length() - 4;
        } );

      // outgoing rays event
      event_loop_.add_rule(
        "RAY-OUT Peer " + to_string( peer_id ),
        [this, peer_it] {
          auto& buffer = peer_it->second.write_buffer;
          auto& rays = peer_it->second.outgoing_rays;

          while ( not rays.empty() and buffer.length() < 4096 * 1024 ) {
            auto ray = std::move( peer_it->second.outgoing_rays.front() );
            peer_it->second.outgoing_rays.pop_front();
            string serialized_buffer( ray->MaxCompressedSize(), '\0' );
            const uint32_t serialized_len = ray->Serialize( serialized_buffer.data() );
            buffer.append( serialized_buffer, 0, serialized_len );
            total_rays_sent_++;
          }
        },
        [peer_it] {
          return not peer_it->second.outgoing_rays.empty() and peer_it->second.write_buffer.length() < 4096 * 1024;
        } );
    }

    // let's start the threads
    if ( is_accumulator_ ) {
      threads_.emplace_back( &LocustaWorker::accumulation_thread, this );
    } else {
      threads_.emplace_back( &LocustaWorker::render_thread, this );
    }
  }

  void run()
  {
    while ( not terminated and event_loop_.wait_next_event( -1 ) != EventLoop::Result::Exit ) {
    }
  }

  ~LocustaWorker()
  {
    terminated = true;
    input_rays_.enqueue( nullptr );

    for ( auto& thread : threads_ ) {
      thread.join();
    }
  }
};

void LocustaWorker::generate_rays()
{
  for ( int sample = 0; sample < scene_.SamplesPerPixel(); sample++ ) {
    for ( pbrt::Point2i pixel : scene_.SampleBounds() ) {
      input_rays_.enqueue( scene_.GenerateCameraRay( pixel, sample ) );
      input_rays_size_++;
    }
  }
}

void LocustaWorker::render_thread()
{
  pbrt::RayStatePtr ray;
  pbrt::MemoryArena mem_arena;

  while ( not terminated ) {
    input_rays_.wait_dequeue( ray );
    if ( ray == nullptr ) {
      return;
    }

    input_rays_size_--;

    pbrt::ProcessRayOutput output;
    scene_.ProcessRay( std::move( ray ), *treelet_, mem_arena, output );

    total_rays_processed_++;

    for ( auto& r : output.rays ) {
      if ( r ) {
        if ( r->CurrentTreelet() == my_treelet_id_ ) {
          input_rays_.enqueue( std::move( r ) );
          input_rays_size_++;
        } else {
          output_rays_.enqueue( std::move( r ) );
          output_rays_size_++;
        }
      }
    }

    if ( output.sample ) {
      output_samples_.enqueue( std::move( output.sample ) );
      output_samples_size_++;
    }
  }

  cerr << "Render thread exiting." << endl;
}

void LocustaWorker::accumulation_thread()
{
  pbrt::RayStatePtr sample;
  vector<pbrt::Sample> current_samples;

  auto last_write = steady_clock::now();

  do {
    while ( input_rays_.try_dequeue( sample ) ) {
      if ( sample == nullptr ) {
        return;
      }

      input_rays_size_--;
      current_samples.emplace_back( *sample );
    }

    if ( current_samples.empty() ) {
      this_thread::sleep_for( 1s );
      continue;
    }

    scene_.AccumulateImage( current_samples );
    total_rays_processed_ += current_samples.size();

    const auto now = steady_clock::now();
    if ( now - last_write > 1s ) {
      scene_.WriteImage( output_path_ / "output.png" );
      last_write = now;
    }

    current_samples.clear();
  } while ( not terminated );

  cerr << "Accumulation thread exiting." << endl;
}

void LocustaWorker::status()
{
  const static auto start_time = steady_clock::now();

  status_timer_.read_event();

  const auto now = steady_clock::now();
  cerr << "(" << duration_cast<seconds>( now - start_time ).count() << "s)\t in_queue=" << input_rays_size_
       << ", out_queue=" << output_rays_size_ << ", sent=" << total_rays_sent_ << ", recv=" << total_rays_received_
       << ", proc=" << total_rays_processed_ << endl;
}

void usage( const char* argv0 )
{
  cerr << argv0 << " TREELET VMCA-SOCKET SCENE-DATA SPP TOTAL-WORKERS" << endl;
}

int main( int, char const* argv[] )
{
  try {
    auto args = json::parse( argv[1] );
    auto cluster_id = args["cluster-id"].get<uint32_t>();
    auto node_id = args["node-id"].get<uint32_t>();
    auto cluster_size = args["cluster-size"].get<size_t>();
    auto scene_path = args["scene_path"].get<string>();
    auto spp = args["spp"].get<int>();

    FLAGS_log_prefix = false;
    google::InitGoogleLogging( argv[0] );

    pbrt::PbrtOptions.compressRays = false;

    auto comm = std::make_unique<CommunicationLibrary>( cluster_id, node_id, cluster_size );
    LocustaWorker worker { scene_path, node_id, spp, std::move( comm ), static_cast<uint32_t>( cluster_size ) };
    worker.run();

  } catch ( exception& ex ) {
    print_exception( argv[0], ex );
    return EXIT_FAILURE;
  }

  printf( "{ \"msg\": \"Done\" }" );
  return EXIT_SUCCESS;
}
