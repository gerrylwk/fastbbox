#pragma once
/**
 * fastbbox_fp16.h
 *
 * Shared float16 utilities and input-view helpers for fastbbox nanobind modules.
 *
 * All IoU computations run in float32.  Input arrays may be either float32
 * (zero-copy fast path) or float16 (converted to a temporary float32 buffer).
 */

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;

// ---------------------------------------------------------------------------
// Typed array aliases – shape-constrained, dtype-unconstrained.
// Nanobind will accept any dtype matching the shape/layout; we validate dtype
// at runtime inside F32View.
// ---------------------------------------------------------------------------
using Array2D4 = nb::ndarray<nb::numpy, nb::shape<-1, 4>, nb::c_contig, nb::device::cpu>;
using Array2D5 = nb::ndarray<nb::numpy, nb::shape<-1, 5>, nb::c_contig, nb::device::cpu>;

// ---------------------------------------------------------------------------
// Portable IEEE 754 float16 → float32 bit-manipulation conversion.
// Handles: ±0, subnormals, normals, ±Inf, NaN.
// ---------------------------------------------------------------------------
inline float f16_to_f32(uint16_t h) noexcept {
    const uint32_t sign     = (uint32_t)(h & 0x8000u) << 16;
    uint32_t       exp      = (h >> 10u) & 0x1Fu;
    uint32_t       mantissa = (uint32_t)(h & 0x03FFu);
    uint32_t       bits;

    if (exp == 0u) {
        if (mantissa == 0u) {
            bits = sign;                                    // ±zero
        } else {
            // Subnormal f16 → normalised f32.
            // Start exponent at 113 (= 127 − 14); shift mantissa left until
            // the implicit leading 1 appears at bit 10, decrementing exp each step.
            exp = 113u;
            while (!(mantissa & 0x0400u)) { mantissa <<= 1; --exp; }
            mantissa &= 0x03FFu;                           // strip the implicit 1
            bits = sign | (exp << 23u) | (mantissa << 13u);
        }
    } else if (exp == 31u) {
        bits = sign | 0x7F800000u | (mantissa << 13u);     // ±Inf / NaN
    } else {
        // Normal: rebias exponent from 15 to 127  (127 − 15 = 112)
        bits = sign | ((exp + 112u) << 23u) | (mantissa << 13u);
    }

    float result;
    std::memcpy(&result, &bits, sizeof(float));
    return result;
}

// Bulk-convert n float16 elements (raw uint16_t bytes) to float32.
inline void cvt_f16_to_f32(const void* src, float* dst, size_t n) noexcept {
    const uint16_t* s = static_cast<const uint16_t*>(src);
    for (size_t i = 0; i < n; ++i)
        dst[i] = f16_to_f32(s[i]);
}

// ---------------------------------------------------------------------------
// F32View<ArrayT>
//
// RAII helper that presents a `const float*` over an ndarray regardless of
// whether the underlying data is float32 or float16.
//
// float32 input → zero-copy: ptr aliases the original buffer.
// float16 input → converts to an owned float32 buffer; ptr points to it.
// Any other dtype → throws std::invalid_argument.
// ---------------------------------------------------------------------------
template <typename ArrayT>
struct F32View {
    std::vector<float> buf;   // non-empty only for float16 inputs
    const float*       ptr;

    explicit F32View(const ArrayT& arr) {
        const auto   dt  = arr.dtype();
        const size_t sz  = arr.shape(0) * arr.shape(1);

        if (dt.code == 2 && dt.bits == 32) {
            // float32 — zero-copy
            ptr = static_cast<const float*>(arr.data());
        } else if (dt.code == 2 && dt.bits == 16) {
            // float16 — convert to temporary float32 buffer
            buf.resize(sz);
            cvt_f16_to_f32(arr.data(), buf.data(), sz);
            ptr = buf.data();
        } else {
            throw std::invalid_argument(
                "fastbbox: expected float32 or float16 input array "
                "(got dtype code=" + std::to_string(dt.code) +
                ", bits=" + std::to_string(dt.bits) + ")");
        }
    }
};
