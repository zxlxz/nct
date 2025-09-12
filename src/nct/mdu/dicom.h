#pragma once

#include <sfc/alloc.h>

namespace nct::mdu {

using namespace sfc;

using DcmVL = u32;

struct DcmTag {
  u16 group = 0;
  u16 element = 0;

 public:
  static auto from_u32(u32 value) {
    const auto l = static_cast<u16>(value & 0xFFFF);
    const auto h = static_cast<u16>((value >> 16) & 0xFFFF);
    return DcmTag{h, l};
  }

  auto as_u32() -> u32 {
    const auto h = static_cast<u32>(group) << 16;
    const auto l = static_cast<u32>(element);
    return h | l;
  }

  auto operator==(const DcmTag& other) const noexcept -> bool {
    return group == other.group && element == other.element;
  }

  void fmt(auto& f) const {
    f.write_fmt("{04X}:{04X}", group, element);
  }
};

struct DcmVR {
  alignas(sizeof(u16)) char _val[2];

 public:
  DcmVR() = default;

  DcmVR(const char (&s)[3]) noexcept : _val{s[0], s[1]} {}

  auto operator==(const DcmVR& other) const noexcept -> bool {
    return _val[0] == other._val[0] && _val[1] == other._val[1];
  }

  auto is_i16() const -> bool;
  auto is_u16() const -> bool;
  auto is_i32() const -> bool;
  auto is_u32() const -> bool;
  auto is_f32() const -> bool;
  auto is_f64() const -> bool;
  auto is_num() const -> bool;
  auto is_str() const -> bool;
  auto is_bin() const -> bool;

  void fmt(auto& f) const {
    f.write_str({_val, 2});
  }
};

struct DcmVal {
  Variant<i16, u16, i32, u32, f32, f64, String, Vec<u8>> _inn;

 public:
  template <class T>
  DcmVal(T val) noexcept : _inn{static_cast<T&&>(val)} {}

  ~DcmVal() noexcept {}

  DcmVal(DcmVal&&) noexcept = default;
  DcmVal& operator=(DcmVal&&) noexcept = default;

  template <class T>
  auto is() const -> bool {
    return _inn.template is<T>();
  }

  template <class T>
  auto as() const -> const T& {
    return _inn.template as<T>();
  }

  void map(auto&& f) const {
    _inn.map(f);
  }

  auto as_bytes() const -> Slice<const u8> {
    return _inn.template as<Vec<u8>>().as_slice();
  }

  void fmt(auto& f) const {
    if (_inn.is<Vec<u8>>()) {
      auto buf = this->as_bytes();
      f.write_fmt("bin(len={})", buf.len());
    } else {
      _inn.fmt(f);
    }
  }
};

class DcmElmt {
  DcmTag _tag = {};  // 4B
  DcmVR _vr = {};    // 2B
  DcmVL _vl = 0;     // 2B|4B
  DcmVal _val = u32(0);

 public:
  auto tag() const -> DcmTag {
    return _tag;
  }

  auto vr() const -> DcmVR {
    return _vr;
  }

  auto vl() const -> DcmVL {
    return _vl;
  }

  auto val() const -> const DcmVal& {
    return _val;
  }

  void fmt(auto& f) const {
    f.write_fmt("{}:{} {}", _tag, _vr, _val);
  }

  auto decode(Slice<const u8> buf) -> usize;
  void encode(Vec<u8>& buf);

  auto decode_head(Slice<const u8> buf) -> usize;
  auto decode_data(Slice<const u8> buf) -> usize;
};

}  // namespace nct::mdu
