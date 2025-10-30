#pragma once

#include "nct/core.h"

namespace nct::data {

auto base64_encode(Slice<const u8> in) -> String;
auto base64_decode(Str in) -> Vec<u8>;

}  // namespace nct::data
