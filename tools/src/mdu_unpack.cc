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

template <class T>
auto unpack_tbl(const DcmElmt& elmt) -> T {
  auto tbl = T{};

  auto buf = elmt.val.as<Vec<u8>>().as_slice();
  if constexpr (requires { tbl.load_data(buf); }) {
    auto hdr = MduHdr{};
    (void)buf.read(mem::as_bytes_mut(hdr));
    tbl.load_head(buf[{hdr.head_offset, hdr.data_offset}]);
    tbl.load_data(buf[{hdr.data_offset, hdr.data_offset + hdr.data_length}]);
  } else {
    (void)buf.read(mem::as_bytes_mut(tbl));
  }

  return tbl;
}

template <class T>
void save_tbl(const MduTbl& mdu, fs::Path out_dir) {
  auto elmt = mdu.get({0x01F7, T::TAG});
  if (!elmt) {
    io::println("get<{}>: failed", str::type_name<T>());
    return;
  }

  const auto tbl = unpack_tbl<T>(*elmt);
  if constexpr (__is_same(T, DetPosTbl)) {
    (void)fs::write(*out_dir.join("detpos_x.bin"), tbl.xpos.as_bytes());
    (void)fs::write(*out_dir.join("detpos_y.bin"), tbl.ypos.as_bytes());
    (void)fs::write(*out_dir.join("detpos_z.bin"), tbl.zpos.as_bytes());
    return;
  }
}

template <class T>
void dump_tbl(const MduTbl& mdu) {
  auto elmt = mdu.get({0x01F7, T::TAG});
  if (!elmt) {
    io::println("load {}: failed", str::type_name<T>());
    return;
  }

  const auto tbl = unpack_tbl<T>(*elmt);

  // dump
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
  dump_tbl<ScannerTechnicalData>(mdu);
}

void save_tbl(const MduTbl& mdu, fs::Path out_dir) {
  (void)fs::create_dir(out_dir);
  save_tbl<DetPosTbl>(mdu, out_dir);
}

auto load_mdu(fs::Path path) -> MduTbl {
  io::println("load '{}'", path);

  auto buf = Vec<u8>{};
  {
    auto file = fs::File::open(path.as_str()).unwrap();
    (void)file.read_to_end(buf);
  }

  auto mdu = MduTbl{};
  mdu.load(buf.as_slice());
  return mdu;
}

int main(int argc, const char* argv[]) {
  auto cmd = app::Clap{"dump_mdu"};
  cmd.flag("h:help", "Print help");
  cmd.arg("i:input", "Input file path", "INPUT");
  cmd.arg("o:output", "Output directory path", "OUTPUT");

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

  if (auto out_path = cmd.get("output")) {
    save_tbl(mdu_data, *out_path);
  }
  return 0;
}
