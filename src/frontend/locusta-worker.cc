#include <algorithm>
#include <atomic>
#include <chrono>
#include <deque>
#include <execution>
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

#include "util/exception.hh"

#include <commlib/commlib.hh>

using namespace std;
using namespace std::chrono;

class LocustaWorker
{
private:
  struct Peer
  {
    locusta::StreamHandle stream;
    deque<pbrt::RayStatePtr> outgoing_rays {};

    string write_buffer {};
    string read_buffer {};

    Peer( locusta::StreamHandle&& s )
      : stream( move( s ) )
    {
    }
  };

  std::unique_ptr<locusta::CommunicationLibrary> commlib_;

  // handling peers
  locusta::EventLoop event_loop_ {};
  map<int, Peer> peers_ {}; // treelet_id -> peer

  // scene data
  const uint32_t total_workers_;
  const uint32_t my_treelet_id_;
  const bool is_accumulator_ { my_treelet_id_ == total_workers_ - 1 };
  pbrt::SceneBase scene_;
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

  locusta::TimerFD status_timer_ { 1s, 1s };

  void generate_rays();

  void render_thread();
  void accumulation_thread();
  void status();

public:
  LocustaWorker( const string& scene_path,
                 const uint32_t treelet_id,
                 const int spp,
                 const int total_workers,
                 std::unique_ptr<locusta::CommunicationLibrary>&& commlib )
    : commlib_( move( commlib ) )
    , total_workers_( total_workers )
    , my_treelet_id_( treelet_id )
    , scene_( scene_path, spp )
    , treelet_( ( my_treelet_id_ < total_workers_ - 1 ) ? pbrt::LoadTreelet( scene_path, treelet_id ) : nullptr )
  {
    if ( total_workers_ != scene_.TreeletCount() + 1 ) {
      throw runtime_error( "treelet count mismatch" );
    }

    cerr << "Loaded treelet " << my_treelet_id_ << "." << endl;

    // each worker connects to the workers after it
    for ( int i = my_treelet_id_ + 1; i < static_cast<int>( total_workers_ ); i++ ) {
      locusta::StreamHandle stream_handle = commlib_->stream_open( i, 0 );
      commlib_->write( stream_handle, reinterpret_cast<const char*>( &treelet_id ), sizeof( treelet_id ) );
      cerr << "Stream opened to node " << i << "." << endl;
      peers_.emplace( i, move( stream_handle ) );
    }

    cerr << "Connections established to all peers." << endl;

    if ( my_treelet_id_ == 0 ) {
      generate_rays();
      cout << "All rays generated." << endl;
    }

    // now setting up the main event loop
    event_loop_.set_fd_failure_callback( [] { cerr << "FD FAILURE CALLBACK" << endl; } );

    /* STATUS */
    event_loop_.add_rule(
      "Status", locusta::Direction::In, status_timer_, bind( &LocustaWorker::status, this ), [] { return true; } );

    /* PROCESSED RAYS -> OUTGOING QUEUE */
    event_loop_.add_rule(
      "Output rays",
      [&] {
        pbrt::RayStatePtr ray;
        while ( output_rays_.try_dequeue( ray ) ) {
          output_rays_size_--;
          auto peer_it = peers_.find( ray->CurrentTreelet() );
          peer_it->second.outgoing_rays.emplace_back( move( ray ) );
        }
      },
      [this] { return output_rays_size_ > 0; } );

    /* PROCESSED SAMPLES -> OUTGOING QUEUE */
    event_loop_.add_rule(
      "Output samples",
      [&] {
        pbrt::RayStatePtr ray;
        while ( output_samples_.try_dequeue( ray ) ) {
          output_samples_size_--;

          auto peer_it = peers_.find( total_workers_ - 1 ); // last worker is the accumulator
          peer_it->second.outgoing_rays.emplace_back( move( ray ) );
        }
      },
      [this] { return output_samples_size_ > 0; } );

    for ( auto& [peer_id, peer] : peers_ ) {
      auto peer_it = peers_.find( peer_id );

      // socket read and write events
      event_loop_.add_rule(
        "SOCKET READ Peer " + to_string( peer_id ),
        locusta::Direction::In,
        peer.stream->first,
        [this, peer_it] { // IN callback
          string buffer( 4096, '\0' );
          auto data_len = commlib_->read( peer_it->second.stream, buffer.data(), buffer.length() );
          peer_it->second.read_buffer.append( buffer, 0, data_len );
        },
        [] { return true; } );

      event_loop_.add_rule(
        "SOCKET WRITE Peer " + to_string( peer_id ),
        locusta::Direction::Out,
        peer.stream->second,
        [this, peer_it] { // OUT callback
          auto data_len = commlib_->write(
            peer_it->second.stream, peer_it->second.write_buffer.data(), peer_it->second.write_buffer.length() );
          peer_it->second.write_buffer.erase( 0, data_len );
        },
        [peer_it] { return not peer_it->second.write_buffer.empty(); } );

      // incoming rays event
      event_loop_.add_rule(
        "RAY-IN Peer " + to_string( peer_id ),
        [this, peer_it] {
          auto& buffer = peer_it->second.read_buffer;
          const uint32_t len = *reinterpret_cast<const uint32_t*>( buffer.data() );
          pbrt::RayStatePtr ray = pbrt::RayState::Create();
          ray->Deserialize( buffer.data() + 4, len );
          buffer.erase( 0, len + 4 );

          this->input_rays_.enqueue( move( ray ) );
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
            auto ray = move( peer_it->second.outgoing_rays.front() );
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
    while ( not terminated and event_loop_.wait_next_event( -1 ) != locusta::EventLoop::Result::Exit ) {
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
    scene_.ProcessRay( move( ray ), *treelet_, mem_arena, output );

    total_rays_processed_++;

    for ( auto& r : output.rays ) {
      if ( r ) {
        if ( r->CurrentTreelet() == my_treelet_id_ ) {
          input_rays_.enqueue( move( r ) );
          input_rays_size_++;
        } else {
          output_rays_.enqueue( move( r ) );
          output_rays_size_++;
        }
      }
    }

    if ( output.sample ) {
      output_samples_.enqueue( move( output.sample ) );
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
      scene_.WriteImage( "output.png" );
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
  cerr << argv0 << " <LOCUSTA-NODE-ID> <LOCUSTA-FD> SCENE-DATA SPP" << endl;
}

int main( int argc, char const* argv[] )
{
  try {
    if ( argc <= 0 ) {
      abort();
    }

    if ( argc != 5 ) {
      usage( argv[0] );
      return EXIT_FAILURE;
    }

    FLAGS_log_prefix = false;
    google::InitGoogleLogging( argv[0] );

    const uint32_t node_id = static_cast<uint32_t>( stoi( argv[1] ) );
    const int fd = stoi( argv[2] );

    auto commlib = make_unique<locusta::CommunicationLibrary>( node_id, fd );

    pbrt::PbrtOptions.compressRays = false;

    const string scene_path { argv[3] };
    const uint32_t treelet_id { commlib->get_node_id() };
    const int spp { stoi( argv[4] ) };

    LOG( INFO ) << "Serving treelet " << treelet_id << " of scene " << scene_path << " with " << spp << " spp." << endl;

    const size_t total_workers = 6;

    LocustaWorker worker { scene_path, treelet_id, spp, total_workers, move( commlib ) };
    worker.run();

  } catch ( exception& ex ) {
    print_exception( argv[0], ex );
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
