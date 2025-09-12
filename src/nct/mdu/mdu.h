#pragma once

#include "nct/mdu/dicom.h"

namespace nct::mdu {

class MduTbl {
  Vec<DcmElmt> _elmts;

 public:
  MduTbl() noexcept;
  ~MduTbl() noexcept;

  MduTbl(MduTbl&&) noexcept = default;
  MduTbl& operator=(MduTbl&&) noexcept = default;

  MduTbl(const MduTbl&) = delete;
  MduTbl& operator=(const MduTbl&) = delete;

  static auto load(Str path) -> MduTbl;

  static auto from_buf(Slice<const u8> buf) -> MduTbl;

  auto as_slice() const -> Slice<const DcmElmt>;

  auto get(DcmTag tag) const -> Option<const DcmElmt&>;
};

}  // namespace nct::mdu
