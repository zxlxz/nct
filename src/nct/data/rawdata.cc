#include "nct/data/rawdata.h"
#include "sfc/fs.h"

namespace nct::data {

auto RawData::load_file(Str path) -> RawData {
  auto data = fs::read(path).unwrap();

  const auto head = reinterpret_cast<const DMSHead*>(data.as_ptr());
  const auto ndet = head->DetSamplesPerSlice;
  const auto nfan = head->NumOfSlices;
  const auto fan_size = sizeof(DMSFoot) + ndet * sizeof(u16);
  const auto view_size = sizeof(DMSHead) + nfan * fan_size;
  const auto view_cnt = data.len() / view_size;

  auto res = RawData{};
  res._ndet = ndet;
  res._nfan = nfan;
  res._nview = data.len() / view_size;
  res._fan_size = fan_size;
  res._view_size = view_size;
  res._data = std::move(data);

  return res;
}

auto RawData::get(usize view_idx, usize slice_idx) const -> Slice<const u16> {
  if (view_idx > _nview || slice_idx > _nfan) {
    return {};
  }
  const auto view_ptr = _data.as_ptr() + view_idx * _view_size;
  const auto data_ptr = reinterpret_cast<const u16*>(view_ptr + slice_idx * _fan_size);
  return {data_ptr, _ndet};
}

}  // namespace nct::data
