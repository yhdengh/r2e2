#pragma once

#include <vector>

#include "net/address.hh"
#include "net/memcached.hh"
#include "transfer.hh"
#include "util/eventloop.hh"

namespace memcached {

class TransferAgent : public ::TransferAgent
{
private:
  std::vector<Address> _servers {};
  EventFD _action_event {};

  void do_action( Action&& action ) override;
  void worker_thread( const size_t thread_id ) override;

public:
  TransferAgent( const std::vector<Address>& servers );
  ~TransferAgent();

  void flush_all();
};

} // namespace memcached
