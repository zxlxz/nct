#pragma once

#include "nct/mdu/dicom.h"

namespace nct::mdu {

class MduList {
  Vec<DcmElmt> _elmts;

 public:
  MduList() noexcept;
  ~MduList() noexcept;

  MduList(MduList&&) noexcept = default;
  MduList& operator=(MduList&&) noexcept = default;

  MduList(const MduList&) = delete;
  MduList& operator=(const MduList&) = delete;

  static auto load(Str path) -> MduList;

  static auto from_buf(Slice<const u8> buf) -> MduList;

  auto as_slice() const -> Slice<const DcmElmt>;

  auto get(DcmTag tag) const -> Option<const DcmElmt&>;
};

}  // namespace nct::mdu
