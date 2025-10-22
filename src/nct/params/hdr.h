#pragma once

#include "nct/data.h"
#include "nct/math.h"

namespace nct::params {

using namespace sfc;

static constexpr auto MAX_FFS = 8U;
using s64 = char[64];

struct MduHdr {
  u32 version = 0;
  u32 type = 0;
  u32 length = 0;
  u32 head_offset = 0;
  u32 data_offset = 0;
  u32 data_length = 0;
  u32 timestamp = 0;
  u32 checksum = 0;
  u32 data_type = 0;
  u32 reserved = 0;

 public:
  void visit(this auto&& self, auto&& f) {
    f("version", self.version);
    f("type", self.type);
    f("length", self.length);
    f("head_offset", self.head_offset);
    f("data_offset", self.data_offset);
    f("data_length", self.data_length);
    f("timestamp", self.timestamp);
    f("checksum", self.checksum);
    f("data_type", self.data_type);
    f("reserved", self.reserved);
  }

  // trait: fmt::Display
  void fmt(auto& f) const {
    auto imp = f.debug_struct();
    this->visit([&](const auto& name, const auto& val) { imp.field(name, val); });
  }
};
}  // namespace nct::params
