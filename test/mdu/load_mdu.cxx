#include "sfc/test.h"
#include "sfc/io.h"
#include "sfc/fs.h"
#include "nct/mdu/mdu.h"

namespace nct::mdu::test {

using namespace sfc;

SFC_TEST(load_mdu) {
  const auto dir = fs::Path{"D:/nct/data"};
  const auto path = Str{"1.3.46.670589.61.128.7.2025031917204055405395646270001.mdu"};

  const auto mdu_list = MduList::load(dir.join(path).as_path().as_str());
  io::println("load '{}'", path);
  for (const auto& mdu_elmt : mdu_list.as_slice()) {
    io::println("{}", mdu_elmt);
  }
}

}  // namespace nct::mdu::test
