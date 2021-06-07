#pragma once

#include <pbrt/main.h>
#include <pbrt/raystate.h>

#include <iostream>
#include <memory>
#include <string>

#include "common/lambda.hh"
#include "messages/message.hh"
#include "net/address.hh"
#include "net/client.hh"
#include "net/session.hh"
#include "net/socket.hh"
#include "net/transfer_s3.hh"
#include "util/eventloop.hh"
#include "util/exception.hh"
#include "util/temp_dir.hh"

namespace r2t2 {

class TileHelper
{
private:
  const uint32_t accumulators_;
  const pbrt::Bounds2<uint32_t> bounds_;
  const pbrt::Vector2<uint32_t> extent_;
  const uint32_t spp_;

  uint32_t tile_size_ {};
  uint32_t active_accumulators_ {};

  pbrt::Vector2<uint32_t> n_tiles_ {};

public:
  TileHelper();

  TileHelper( const uint32_t accumulators,
              const pbrt::Bounds2i& sample_bounds,
              const uint32_t spp );

  uint32_t tile_id( const pbrt::Sample& sample ) const;

  uint32_t tile_size() const { return tile_size_; }
  uint32_t active_accumulators() const { return active_accumulators_; }

  pbrt::Bounds2<uint32_t> bounds( const uint32_t tile_id ) const;
};

class Accumulator
{
private:
  EventLoop loop_ {};
  TempDirectory working_directory_ { "/tmp/r2t2-accumulator" };
  TCPSocket listener_socket_ {};

  std::string job_id_ {};
  std::unique_ptr<S3TransferAgent> job_transfer_agent_ { nullptr };

  std::pair<uint32_t, uint32_t> dimensions_ {};
  uint32_t tile_count_ {};
  uint32_t tile_id_ {};

  pbrt::scene::Base scene_ {};

  struct Worker
  {
    Worker( TCPSocket&& socket )
      : client( std::move( socket ) )
    {}

    meow::Client<TCPSession> client;
  };

  std::list<Worker> workers_ {};

  meow::Client<TCPSession>::RuleCategories rule_categories {
    loop_.add_category( "Socket" ),
    loop_.add_category( "Message read" ),
    loop_.add_category( "Message write" ),
    loop_.add_category( "Process message" )
  };

  void process_message( std::list<Worker>::iterator worker_it,
                        meow::Message&& msg );

public:
  Accumulator( const uint16_t listen_port );
  void run();
};

/* responsible for getting the samples to the accumulator servers */
class SampleAgent
{
public:
  SampleAgent( const std::vector<Address>& accumulators );
  ~SampleAgent();

  void add_sample( pbrt::Sample&& sample );
};

}
