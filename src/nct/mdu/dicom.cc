#include "nct/mdu/dicom.h"

namespace nct::mdu {

struct DcmElmtReader {
  Slice<const u8> _buf;

 public:
  template <class T>
  auto read() -> Option<T> {
    static_assert(__is_trivially_copyable(T));

    if (_buf._len < sizeof(T)) {
      return {};
    }

    const auto p = _buf._ptr;
    _buf._ptr += sizeof(T);
    _buf._len -= sizeof(T);

    auto res = ptr::read_unaligned<T>(p);
    return res;
  }

  auto read_str(u32 len) -> String {
    if (_buf._len < len) {
      return {};
    }

    auto res = String{};
    res.write_str({_buf._ptr, len});
    _buf._ptr += len;
    _buf._len -= len;
    return res;
  }

  auto read_vec(u32 len) -> Vec<u8> {
    if (_buf._len < len) {
      return {};
    }

    auto res = Vec<u8>{};
    res.extend_from_slice({_buf._ptr, len});
    _buf._ptr += len;
    _buf._len -= len;
    return res;
  }
};

struct DcmElmtWriter {
  Vec<u8>& _buf;

 public:
  template <class T>
  void write(const T& val) {
    static_assert(__is_trivially_copyable(T));
    _buf.reserve(sizeof(T));
    _buf.extend_from_slice({reinterpret_cast<const u8*>(&val), sizeof(T)});
  }

  void write(const String& str) {
    _buf.extend_from_slice(str.as_str().as_bytes());
  }

  void write(const Vec<u8>& bin) {
    _buf.extend_from_slice(bin.as_slice());
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
  auto reader = DcmElmtReader{buf};
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

  auto reader = DcmElmtReader{buf};
  if (_vr.is_i16()) {
    _val = reader.read<i16>().unwrap_or({});
  } else if (_vr.is_u16()) {
    _val = reader.read<u16>().unwrap_or({});
  } else if (_vr.is_i32()) {
    _val = reader.read<i32>().unwrap_or({});
  } else if (_vr.is_u32()) {
    _val = reader.read<u32>().unwrap_or({});
  } else if (_vr.is_f32()) {
    _val = reader.read<f32>().unwrap_or({});
  } else if (_vr.is_f64()) {
    _val = reader.read<f64>().unwrap_or({});
  } else if (_vr.is_str()) {
    _val = reader.read_str(_vl);
  } else {
    _val = reader.read_vec(_vl);
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
  auto writer = DcmElmtWriter{buf};
  writer.write(_tag);
  writer.write(_vr);

  if (!_val.is<Vec<u8>>()) {
    writer.write(static_cast<u16>(_vl));
  } else {
    writer.write(u16{0});
    writer.write(_vl);
  }

  _val.map([&](const auto& t) { writer.write(t); });
}

}  // namespace nct::mdu
