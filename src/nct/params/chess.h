#pragma once

#include "nct/params/hdr.h"

namespace nct::params {

struct ChessData {
  static constexpr auto TAG = u16{0x10CC};

  struct {
    i32 version;
    i32 nSereiesInitStructSize;
    i32 nNumDicoms;
    i32 nDicomSizes[200];
    i32 nEcgOrigSize;
    i32 nEcgEditedSize;
    i32 Spare[11];
  };
};

}  // namespace nct::params
