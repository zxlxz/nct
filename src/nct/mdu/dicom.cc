#include "nct/mdu/dicom.h"
#include <sfc/io.h>

namespace nct::mdu {

struct DcmDecoder {
  Slice<const u8> _inn;

 public:
  template <class T>
  auto read() -> Option<T> {
    static_assert(__is_trivially_copyable(T));

    if (_inn._len < sizeof(T)) {
      return {};
    }
    const auto [a, b] = _inn.split_at(sizeof(T));
    _inn = b;
    return ptr::read_unaligned<T>(a._ptr);
  }

  auto read_buf(usize len) -> Slice<const u8> {
    if (_inn._len < len) {
      return {};
    }
    const auto [a, b] = _inn.split_at(len);
    _inn = b;
    return a;
  }

  auto read_str(usize len) -> Str {
    if (_inn._len < len) {
      return {};
    }
    const auto [a, b] = _inn.split_at(len);
    _inn = b;
    return Str::from_u8(a);
  }
};

struct DcmEncoder {
  Vec<u8>& _inn;

 public:
  template <class T>
  void write(const T& val) {
    static_assert(__is_trivially_copyable(T));

    _inn.reserve(sizeof(T));
    ptr::write_unaligned(_inn.as_mut_ptr() + _inn.len(), val);
    _inn.set_len(_inn.len() + sizeof(T));
  }

  void write_buf(Slice<const u8> buf) {
    _inn.extend_from_slice(buf);
  }

  void write_str(Str buf) {
    _inn.extend_from_slice(buf.as_bytes());
  }
};

auto DcmVR::is_i16() const -> bool {
  return *this == "SS";
}

auto DcmVR::is_u16() const -> bool {
  return *this == "US";
}

auto DcmVR::is_i32() const -> bool {
  return *this == "SL";
}

auto DcmVR::is_u32() const -> bool {
  return *this == "UL";
}

auto DcmVR::is_f32() const -> bool {
  return *this == "FL";
}

auto DcmVR::is_f64() const -> bool {
  return *this == "FD";
}

auto DcmVR::is_num() const -> bool {
  static const DcmVR pats[] = {"SS", "US", "SL", "UL", "FL", "FD"};
  for (const auto& pat : pats) {
    if (*this == pat) {
      return true;
    }
  }
  return false;
}

auto DcmVR::is_str() const -> bool {
  static const DcmVR pats[] = {
      "AT", "IS", "DA", "DS", "DT", "AE", "AS", "CS", "DA", "TM",
      "UI", "UR", "LO", "LT", "PN", "SH", "ST", "UT", "UC",
  };
  for (const auto& pat : pats) {
    if (*this == pat) {
      return true;
    }
  }
  return false;
}

auto DcmVR::is_bin() const -> bool {
  static const DcmVR pats[] = {"OB", "OD", "OF", "OL", "OV", "OW", "UN"};
  for (const auto& pat : pats) {
    if (*this == pat) {
      return true;
    }
  }
  return false;
}

auto DcmElmt::decode_head(Slice<const u8> buf) -> usize {
  if (buf._len < 8) {
    return 0;
  }

  auto reader = DcmDecoder{buf};
  _tag = reader.read<DcmTag>().unwrap();
  _vr = reader.read<DcmVR>().unwrap();
  _vl = reader.read<u16>().unwrap();

  if (_vr.is_num() || _vr.is_str()) {
    return 8;
  }

  if (_vr.is_bin()) {
    if (buf._len < 12) {
      return 0;
    }
    _vl = reader.read<u32>().unwrap();
    return 12;
  }

  return 0;
};

auto DcmElmt::decode_data(Slice<const u8> buf) -> usize {
  if (_vl > buf.len()) {
    return 0;
  }

  auto decoder = DcmDecoder{buf};
  if (_vr.is_i16()) {
    _val = decoder.read<i16>().unwrap_or({});
  } else if (_vr.is_u16()) {
    _val = decoder.read<u16>().unwrap_or({});
  } else if (_vr.is_i32()) {
    _val = decoder.read<i32>().unwrap_or({});
  } else if (_vr.is_u32()) {
    _val = decoder.read<u32>().unwrap_or({});
  } else if (_vr.is_f32()) {
    _val = decoder.read<f32>().unwrap_or({});
  } else if (_vr.is_f64()) {
    _val = decoder.read<f64>().unwrap_or({});
  } else if (_vr.is_str()) {
    _val = String::from(decoder.read_str(_vl));
  } else {
    _val = Vec<u8>::from(decoder.read_buf(_vl));
  }

  return _vl;
}

auto DcmElmt::decode(Slice<const u8> buf) -> usize {
  const auto head_size = this->decode_head(buf);
  if (head_size == 0) {
    return 0;
  }
  const auto data_size = this->decode_data(buf[{head_size, _}]);
  if (data_size == 0) {
    return 0;
  }
  return head_size + data_size;
}

void DcmElmt::encode(Vec<u8>& buf) {
  auto writer = DcmEncoder{buf};
  writer.write(_tag);
  writer.write(_vr);

  if (!_val.is<Vec<u8>>()) {
    writer.write(static_cast<u16>(_vl));
  } else {
    writer.write(u16{0});
    writer.write(_vl);
  }

  _val.map([&](const auto& t) {});
}

}  // namespace nct::mdu
