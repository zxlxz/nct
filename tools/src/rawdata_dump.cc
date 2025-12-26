#include "nct/data/rawdata.h"
#include "sfc/app.h"
#include "sfc/fs.h"

using namespace nct;
using namespace nct::data;

void rawdata_dump(Str raw_path, Str out_path) {
  const auto raw_data = RawData::load_file(raw_path);

  const auto nview = raw_data.view_cnt;
  const auto slice_cnt = raw_data.slice_cnt;

  auto out_file = fs::File::create(out_path).unwrap();
  for (auto view_idx = 0UL; view_idx < nview; ++view_idx) {
    for (auto slice_idx = 0UL; slice_idx < slice_cnt; ++slice_idx) {
      const auto slice = raw_data.get(view_idx, slice_idx);
      out_file.write(slice.as_bytes()).unwrap();
    }
  }
}

int main(int argc, const char* argv[]) {
  auto cmd = app::Clap{"rawdata_dump"};
  cmd.flag("h:help", "Print help");
  cmd.arg("i:input", "raw data path", "INPUT");
  cmd.arg("o:output", "output file path", "OUTPUT?");

  if (!cmd.parse_cmdline(argc, argv)) {
    cmd.print_help();
    return -1;
  }

  if (cmd.get("help")) {
    cmd.print_help();
    return 0;
  }

  const auto raw_path = cmd.get("input").unwrap();
  const auto out_path = cmd.get("output").unwrap();
  rawdata_dump(raw_path, out_path);

  return 0;
}
