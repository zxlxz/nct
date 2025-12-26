#include "nct/data/rawdata.h"
#include "sfc/fs.h"

namespace nct::data {

auto RawData::load_file(Str path) -> RawData {
  auto data = fs::read(path).unwrap();

  const auto head = reinterpret_cast<const DMSHead*>(data.as_ptr());
  const auto ndet = head->DetSamplesPerSlice;
  const auto nslice = head->NumOfSlices;
  const auto slice_size = sizeof(DMSFoot) + ndet * sizeof(u16);
  const auto view_size = sizeof(DMSHead) + nslice * slice_size;
  const auto view_cnt = data.len() / view_size;

  auto res = RawData{
      .det_cnt = ndet,
      .slice_cnt = nslice,
      .view_cnt = view_cnt,
      .slice_size = slice_size,
      .view_size = view_size,
      .raw_data = mem::move(data),
  };
  return res;
}

auto RawData::get(usize view_idx, usize slice_idx) const -> Slice<const u16> {
  if (view_idx > view_cnt || slice_idx > slice_cnt) {
    return {};
  }
  const auto view_ptr = raw_data.as_ptr() + view_idx * view_size;
  const auto data_ptr = reinterpret_cast<const u16*>(view_ptr + slice_idx * slice_size);
  return {data_ptr, det_cnt};
}

}  // namespace nct::data
