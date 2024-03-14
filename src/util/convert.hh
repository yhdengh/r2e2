#pragma once

#include <string_view>
#include <cstdint>

uint64_t to_uint64( const std::string_view str, const int base = 10 );
