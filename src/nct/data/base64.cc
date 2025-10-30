#include "nct/data/base64.h"

namespace nct::data {

static void base64_encode_blk(usize n, const u8 in[], u8 (&out)[4]) {
  static constexpr auto ENCODE_TBL =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      "abcdefghijklmnopqrstuvwxyz"
      "0123456789+/";

  u32 val = 0;
  for (u32 i = 0; i < n; ++i) {
    val = (val << 8) | in[i];
  }

  out[0] = ENCODE_TBL[(val >> 18) & 0x3F];
  out[1] = ENCODE_TBL[(val >> 12) & 0x3F];
  out[2] = (n > 1) ? ENCODE_TBL[(val >> 6) & 0x3F] : '=';
  out[3] = (n > 2) ? ENCODE_TBL[val & 0x3F] : '=';
}

static auto base64_decode_blk(usize out_len, const u8 in[4], u8 out[3]) -> bool {
  static const u8 DECODE_MAP[123] = {
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  62, 0,  0,  0,  63, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 0,  0,  0,  0,  0,
      0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
      19, 20, 21, 22, 23, 24, 25, 0,  0,  0,  0,  0,  0,  26, 27, 28, 29, 30, 31, 32, 33,
      34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
  };

  const auto v0 = in[0] < sizeof(DECODE_MAP) ? DECODE_MAP[in[0]] : 0U;
  const auto v1 = in[1] < sizeof(DECODE_MAP) ? DECODE_MAP[in[1]] : 0U;
  const auto v2 = in[2] < sizeof(DECODE_MAP) ? DECODE_MAP[in[2]] : 0U;
  const auto v3 = in[3] < sizeof(DECODE_MAP) ? DECODE_MAP[in[3]] : 0U;
  const auto val = (v0 << 18) | (v1 << 12) | (v2 << 6) | v3;

  if (out_len > 0) {
    out[0] = (val >> 16) & 0xFF;
  }
  if (out_len > 1) {
    out[1] = (val >> 8) & 0xFF;
  }
  if (out_len > 2) {
    out[2] = val & 0xFF;
  }

  return true;
}

auto base64_encode(Slice<const u8> in) -> String {
  const auto out_len = ((in._len + 2) / 3) * 4;

  auto out = String{};
  out.reserve(out_len);

  auto& out_buf = out.as_mut_vec();
  for (auto i = 0u, j = 0U; i < in._len; i += 3) {
    const auto in_len = in._len - i >= 3 ? 3 : in._len - i;
    const auto in_blk = in._ptr + i;

    u8 out_blk[4] = {};
    base64_encode_blk(in_len, in_blk, out_blk);

    out_buf.extend_from_slice({out_blk, 4});
  }

  return out;
}

auto base64_decode(Str in) -> Vec<u8> {
  if (in.len() % 4 != 0) {
    return {};
  }

  const auto in_buf = in.as_bytes();

  const auto out_len = (in._len / 4) * 3;
  auto out = Vec<u8>{};
  out.reserve(out_len);

  for (auto i = 0u; i < in._len; i += 4) {
    const auto in_blk = in_buf.as_ptr() + i;
    const auto out_len = (in_blk[2] == '=') ? 1U : (in_blk[3] == '=') ? 2U : 3U;

    u8 out_blk[3] = {};
    base64_decode_blk(out_len, in_blk, out_blk);

    out.extend_from_slice({out_blk, out_len});
  }

  return out;
}

}  // namespace nct::data
