#include "nct/data/rawdata.h"
#include "sfc/fs.h"

namespace nct::data {

auto RawData::load_file(Str path) -> RawData {
  auto data = fs::read(path).unwrap();

  const auto head = reinterpret_cast<const DMSHead*>(data.as_ptr());
  const auto ndet = head->DetSamplesPerSlice;
  const auto nslice = head->NumOfSlices;
  const auto fan_size = sizeof(DMSFoot) + ndet * sizeof(u16);
  const auto view_size = sizeof(DMSHead) + nslice * fan_size;
  const auto view_cnt = data.len() / view_size;

  auto res = RawData{};
  res._ndet = ndet;
  res._nslice = nslice;
  res._view_size = view_size;
  res._view_cnt = data.len() / view_size;
  res._data = std::move(data);

  return res;
}

}  // namespace nct::data
