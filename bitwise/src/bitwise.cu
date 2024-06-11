
#include <cstdio>
#include <cub/cub.cuh>
#include <stdint.h>

template <typename T>
__device__ void bitwise_and(const T *d_in1, const T *d_in2, T *d_out,
                            const uint32_t N) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    d_out[idx] = d_in1[idx] & d_in2[idx];
  }
}

#define BITWISE_AND_OP(TYPENAME, RUST_NAME)                                    \
  extern "C" __global__ void bitwise_and_##RUST_NAME(                          \
      const TYPENAME *d_in1, const TYPENAME *d_in2, TYPENAME *d_out,           \
      uint32_t N) {                                                            \
    bitwise_and(d_in1, d_in2, d_out, N);                                       \
  }

BITWISE_AND_OP(uint8_t, u8)
BITWISE_AND_OP(uint32_t, u32)
BITWISE_AND_OP(int64_t, i64)

template <typename T>
__device__ void bitwise_or(const T *d_in1, const T *d_in2, T *d_out,
                           const uint32_t N) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    d_out[idx] = d_in1[idx] | d_in2[idx];
  }
}

#define BITWISE_OR_OP(TYPENAME, RUST_NAME)                                     \
  extern "C" __global__ void bitwise_or_##RUST_NAME(                           \
      const TYPENAME *d_in1, const TYPENAME *d_in2, TYPENAME *d_out,           \
      uint32_t N) {                                                            \
    bitwise_or(d_in1, d_in2, d_out, N);                                        \
  }

BITWISE_OR_OP(uint8_t, u8)
BITWISE_OR_OP(uint32_t, u32)
BITWISE_OR_OP(int64_t, i64)

template <typename T>
__device__ void bitwise_xor(const T *d_in1, const T *d_in2, T *d_out,
                            const uint32_t N) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    d_out[idx] = d_in1[idx] ^ d_in2[idx];
  }
}

#define BITWISE_XOR_OP(TYPENAME, RUST_NAME)                                    \
  extern "C" __global__ void bitwise_xor_##RUST_NAME(                          \
      const TYPENAME *d_in1, const TYPENAME *d_in2, TYPENAME *d_out,           \
      uint32_t N) {                                                            \
    bitwise_xor(d_in1, d_in2, d_out, N);                                       \
  }

BITWISE_XOR_OP(uint8_t, u8)
BITWISE_XOR_OP(uint32_t, u32)
BITWISE_XOR_OP(int64_t, i64)