#pragma once

#include <math.h>
#include <new>

#ifdef __CUDACC__
#include <sfc/core/mem.h>
#else
#include <sfc/core.h>
#include <sfc/alloc.h>
#endif

#ifndef __CUDACC__
#define __hd__
#else
#define __hd__ __host__ __device__
#endif

namespace nct {
using namespace sfc;
}
