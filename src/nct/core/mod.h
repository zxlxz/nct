#pragma once

#include <math.h>

#ifdef __CUDACC__
#include <sfc/core/mem.h>
#include <sfc/core/ptr.h>
#else
#include <sfc/core.h>
#include <sfc/alloc.h>
#include <sfc/io/mod.h>
#endif

namespace nct {
using namespace sfc;
}
