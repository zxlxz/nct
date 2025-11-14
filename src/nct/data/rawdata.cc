#include "nct/data/rawdata.h"
#include "sfc/fs.h"

namespace nct::data {

auto RawData::load_file(Str path) -> RawData {
  auto data = fs::read(path).unwrap();

  const auto head = reinterpret_cast<const DMSHead*>(data.as_ptr());
  auto res = RawData{};
  return res;
}

}  // namespace nct::data
