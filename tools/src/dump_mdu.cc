#include "nct/params/hcor.h"
#include "nct/params/scan.h"
#include "nct/params/det.h"
#include "nct/params/air.h"
#include "nct/params/scatter.h"

#include "sfc/fs.h"
#include "sfc/app.h"

using namespace nct;
using namespace nct::data;
using namespace nct::params;

void dump_mdu(const MduTbl& mdu) {
  for (const auto& elmt : mdu.as_slice()) {
    io::println("{}", elmt);
  }
}

auto load_mdu(fs::Path path) -> MduTbl {
  io::println("load '{}'", path);

  auto buf = Vec<u8>{};
  {
    auto file = fs::File::open(path.as_str()).unwrap();
    file.read_to_end(buf);
  }

  auto mdu = MduTbl{};
  if (!mdu.load(buf.as_slice())) {
    io::println("load mdu failed");
  }
  return mdu;
}

int main(int argc, const char* argv[]) {
  auto cmd = app::Clap{"dump_mdu"};
  cmd.add_opt("h:help", "Print help");
  cmd.add_arg("i:input", "Input file path", "INPUT");

  cmd.parse_cmdline(argc, argv);
  if (cmd.get("help")) {
    cmd.print_help();
    return 0;
  }

  const auto mdu_path = cmd.get("input");
  if (!mdu_path) {
    return -1;
  }

  const auto mdu_data = load_mdu(*mdu_path);
  dump_mdu(mdu_data);

  return 0;
}
