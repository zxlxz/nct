#include "sfc/fs.h"
#include "sfc/app.h"

#include "nct/params/hcor.h"
#include "nct/params/scan.h"
#include "nct/params/det.h"
#include "nct/params/air.h"
#include "nct/params/scatter.h"

using namespace nct;
using namespace nct::data;
using namespace nct::params;

template <class T>
void dump_tbl(const MduTbl& mdu) {
  auto elmt = mdu.get({0x01F7, T::TAG});
  if (!elmt) {
    io::println("load {}: failed", str::type_name<T>());
    return;
  }

  auto buf = elmt->val.as_bytes();

  auto tbl = T{};

  if constexpr (requires { tbl.load_data(buf); }) {
    auto hdr = MduHdr{};
    auto{buf}.read(mem::as_bytes_mut(hdr));
    tbl.load_head(buf[{hdr.head_offset, hdr.data_offset}]);
    tbl.load_data(buf[{hdr.data_offset, hdr.data_offset + hdr.data_length}]);
  } else {
    buf.read(mem::as_bytes_mut(tbl));
  }

  io::println("{} = {#?}", str::type_name<T>(), tbl);
}

void dump_mdu(const MduTbl& mdu) {
  for (const auto& elmt : mdu.as_slice()) {
    io::println("{}", elmt);
  }

  dump_tbl<HCorTbl>(mdu);
  dump_tbl<DetPosTbl>(mdu);
  dump_tbl<AirCorTbl>(mdu);
  dump_tbl<ScatterTbl>(mdu);
  dump_tbl<ScanParam>(mdu);
}

auto load_mdu(fs::Path path) {
  io::println("load '{}'", path);

  auto buf = Vec<u8>{};
  {
    auto file = fs::File::open(path.as_str()).unwrap();
    file.read_to_end(buf);
  }

  auto mdu = MduTbl{};
  mdu.load(buf.as_slice());
  return mdu;
}

int main(int argc, const char* argv[]) {
  auto cmd = app::Clap{"mdu_unpack"};
  cmd.add_opt("h:help", "Print help");
  cmd.add_arg("i:input", "Input file path", "INPUT");
  if (cmd.parse_cmdline(argc, argv) != 0) {
    cmd.print_help();
    return -1;
  }

  if (cmd.get("help")) {
    cmd.print_help();
    return 0;
  }

  const auto mdu_path = cmd.get("input").unwrap();
  const auto mdu = load_mdu(mdu_path);
  dump_mdu(mdu);
  return 0;
}
