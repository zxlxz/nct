#include "sfc/io.h"
#include "sfc/fs.h"
#include "sfc/app.h"

#include "nct/mdu/mdu.h"

#include "nct/params/hcor.h"
#include "nct/params/scan.h"
#include "nct/params/det_pos.h"


using namespace sfc;
using namespace nct;
using namespace nct::mdu;
using namespace nct::params;

template <class T>
void dump_elmt(Option<const DcmElmt&> elmt) {
  if (!elmt) {
    return;
  }

  auto val = T::from_raw(elmt->val().as_bytes());
  io::println("{}: {#}", str::type_name<T>(), val);
}

void dump_mdu(const MduTbl& mdu) {
  for (const auto& elmt : mdu.as_slice()) {
    io::println("{}", elmt);
  }

  dump_elmt<HCorTbl>(mdu.get({0x01F7, 0x00D9}));

  // Scan Parameters
  if (auto elmt = mdu.get({0x01F7, 0x1011})) {}

  // Scanner Technical Data
  if (auto elmt = mdu.get({0x01F7, 0x1014})) {}

  // Data Description
  if (auto elmt = mdu.get({0x01F7, 0x1015})) {}

  // Scan Description
  if (auto elmt = mdu.get({0x01F7, 0x1016})) {}

  // Acq Parameters
  if (auto elmt = mdu.get({0x01F7, 0x1018})) {}

  // Recon Parameters
  if (auto elmt = mdu.get({0x01F7, 0x1019})) {}

  // Prep Parameters
  if (auto elmt = mdu.get({0x01F7, 0x101B})) {}

  // Image Balance Parameters
  if (auto elmt = mdu.get({0x01F7, 0x101C})) {}

  // Spiral Parameters
  if (auto elmt = mdu.get({0x01F7, 0x101F})) {}

  // Spiral Parameters
  if (auto elmt = mdu.get({0x01F7, 0x1027})) {}

  // FilterBP Parameters
  if (auto elmt = mdu.get({0x01F7, 0x1029})) {}

  // Surview Recon Parameters
  if (auto elmt = mdu.get({0x01F7, 0x102B})) {}

  // Air Calibration Table[Base]
  if (auto elmt = mdu.get({0x01F7, 0x1050})) {}

  // Wedge Vector
  if (auto elmt = mdu.get({0x01F7, 0x1058})) {}

  // Bad Detector Table
  if (auto elmt = mdu.get({0x01F7, 0x1059})) {}

  // Spectrum Correction Table
  if (auto elmt = mdu.get({0x01F7, 0x105F})) {}

  // Scatter Kernel Table
  if (auto elmt = mdu.get({0x01F7, 0x1065})) {}

  // Wedge Scatter Table
  if (auto elmt = mdu.get({0x01F7, 0x1066})) {}

  // Scatter Correction Parameters
  if (auto elmt = mdu.get({0x01F7, 0x1067})) {}

  // HCOR coeffs
  if (auto elmt = mdu.get({0x01F7, 0x1075})) {}

  // Channel Cor Table
  if (auto elmt = mdu.get({0x01F7, 0x107A})) {}

  // Detector Position Table
  if (auto elmt = mdu.get({0x01F7, 0x108D})) {}

  // Chess Data Structures
  if (auto elmt = mdu.get({0x01F7, 0x10CC})) {}
}

auto load_mdu(fs::Path path) {
  io::println("load '{}'", path);
  auto mdu = MduTbl::load(path.as_str());
  return mdu;
}

int main(int argc, const char* argv[]) {
  auto cmd = app::Cmd{"mdu_unpack"};
  cmd.add_arg({'i', "input", "Input file path"});
  cmd.add_arg({'o', "output", "Output file path"});
  cmd.add_opt({'h', "help", "Print help"});
  cmd.parse_cmdline(argc, argv);

  if (argc <= 1 || cmd.get("help")) {
    cmd.print_help();
    return 0;
  }

  const auto mdu_path = cmd.get("input").unwrap_or({});
  if (mdu_path) {
    const auto mdu = load_mdu(mdu_path);
    dump_mdu(mdu);
    return 0;
  }

  return 0;
}
