#include "nct/data/rawdata.h"
#include "sfc/app.h"

using namespace nct;
using namespace nct::data;

int main(int argc, const char* argv[]) {
  auto cmd = app::Clap{"rawdata_dump"};
  cmd.add_opt("h:help", "Print help");
  cmd.add_arg("i:input", "raw file path", "INPUT");

  if (cmd.parse_cmdline(argc, argv) < 0) {
    cmd.print_help();
    return -1;
  }

  if (cmd.get("help")) {
    cmd.print_help();
    return 0;
  }

  const auto input_path = cmd.get("input").unwrap();
  const auto rawdata = RawData::load_file(input_path);
  (void)rawdata;

  return 0;
}
