#include "sfc/io.h"
#include "sfc/fs.h"
#include "sfc/app.h"
#include "nct/mdu/mdu.h"

using namespace sfc;
using namespace nct;

void dump_mdu(fs::Path path) {
  io::println("load '{}'", path);
  const auto mdu_list = mdu::MduList::load(path.as_str());
  for (const auto& mdu_elmt : mdu_list.as_slice()) {
    io::println("{}", mdu_elmt);
  }
}

int main(int argc, const char* argv[]) {
  auto cmd = app::Cmd{"mdu_unpack"};
  cmd.add_arg({'i', "input", "Input file path"});
  cmd.add_arg({'o', "output", "Output file path"});
  cmd.add_arg({'h', "help", "Print help"});
  cmd.parse_cmdline(argc, argv);

  if (argc == 1 || cmd.get("h")) {
    cmd.print_help();
    return 0;
  }

  const auto mdu_path = cmd.get("input").unwrap_or({});
  if (mdu_path) {
    dump_mdu(mdu_path);
    return 0;
  }

  return 0;
}
