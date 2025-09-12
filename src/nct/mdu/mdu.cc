#include "nct/mdu/mdu.h"
#include <sfc/fs.h>

namespace nct::mdu {

MduTbl::MduTbl() noexcept {}

MduTbl::~MduTbl() noexcept {}

auto MduTbl::load(Str path) -> MduTbl {
  auto file_res = fs::File::open(path);
  if (file_res.is_err()) {
    return {};
  }

  auto file = mem::move(file_res).unwrap();
  auto buf = Vec<u8>{};
  if (file.read_to_end(buf).is_err()) {
    return {};
  }

  return MduTbl::from_buf(buf.as_slice());
}

auto MduTbl::from_buf(Slice<const u8> buf) -> MduTbl {
  auto res = MduTbl{};
  for (auto pos = 0U;;) {
    auto tmp = DcmElmt{};
    auto len = tmp.decode(buf[{pos, _}]);
    if (len == 0) {
      break;
    }
    res._elmts.push(mem::move(tmp));
    pos += len;
  }
  return res;
}

auto MduTbl::as_slice() const -> Slice<const DcmElmt> {
  return _elmts.as_slice();
}

auto MduTbl::get(DcmTag tag) const -> Option<const DcmElmt&> {
  for (const auto& elmt : _elmts) {
    if (elmt.tag() == tag) {
      return elmt;
    }
  }
  return {};
}

}  // namespace nct::mdu
