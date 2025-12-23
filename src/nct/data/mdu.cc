#include "nct/data/mdu.h"
#include <sfc/fs.h>

namespace nct::data {

MduTbl::MduTbl() noexcept {}

MduTbl::~MduTbl() noexcept {}

auto MduTbl::load(Slice<const u8> buf) -> bool {
  for (; !buf.is_empty();) {
    auto tmp = DcmElmt{};

    const auto len = tmp.decode(buf);
    if (len == 0) {
      return false;
    }
    _elmts.push(mem::move(tmp));
    buf = buf[{len, $}];
  }

  return true;
}

auto MduTbl::as_slice() const -> Slice<const DcmElmt> {
  return _elmts.as_slice();
}

auto MduTbl::get(DcmTag tag) const -> Option<const DcmElmt&> {
  for (const auto& elmt : *_elmts) {
    if (elmt.tag == tag) {
      return elmt;
    }
  }
  return {};
}

}  // namespace nct::data
