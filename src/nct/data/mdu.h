#pragma once

#include "nct/data/dcm.h"

namespace nct::data {

class MduTbl {
  Vec<DcmElmt> _elmts{};

 public:
  MduTbl() noexcept = default;
  ~MduTbl() noexcept = default;

  MduTbl(MduTbl&&) noexcept = default;
  MduTbl& operator=(MduTbl&&) noexcept = default;

 public:
  auto load(Slice<const u8> buf) -> bool;

  auto as_slice() const -> Slice<const DcmElmt>;

  auto get(DcmTag tag) const -> Option<const DcmElmt&>;
};

}  // namespace nct::data
