// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#ifdef __MUSACC__
#include <musa.h>
#include <musa_fp16.h>

typedef _Float16 v8f16_t __attribute__((vector_size(16)));
#endif

#include "paddle/phi/kernels/funcs/aligned_vector.h"

namespace phi {
namespace fusion {

template <typename T, typename MPType, int VecSize = 2>
__device__ void VectorizedGetSinCos(phi::Array<const T*, 2> sin_cos_data,
                                    const int64_t* position_ids_data,
                                    bool flag_sin_cos,
                                    int64_t index,
                                    int64_t seq_len,
                                    int64_t num_heads,
                                    int64_t head_dim,
                                    MPType* out_sin,
                                    MPType* out_cos,
                                    MPType div_c) {
  MPType* sin_value = out_sin;
  MPType* cos_value = out_cos;

  if (flag_sin_cos) {
#pragma unroll
    for (int64_t nx = 0; nx < VecSize; ++nx) {
      int64_t index_wc = (index + nx) % (seq_len * num_heads * head_dim);
      int64_t pos_seq_ori = index_wc / (num_heads * head_dim);
      int64_t pos_seq;
      if (position_ids_data) {
        int64_t pos_bs = (index + nx) / (seq_len * num_heads * head_dim);
        int64_t index_ids = pos_bs * seq_len + pos_seq_ori;
        pos_seq = position_ids_data[index_ids];
      } else {
        pos_seq = pos_seq_ori;
      }
      int64_t pos_head = index_wc % head_dim;
      int64_t index_sc = pos_seq * head_dim + pos_head;
      const T* sin_input = sin_cos_data[0] + index_sc;
      const T* cos_input = sin_cos_data[1] + index_sc;

      sin_value[nx] = static_cast<MPType>(sin_input[0]);
      cos_value[nx] = static_cast<MPType>(cos_input[0]);
    }
  } else {
#pragma unroll
    for (int nx = 0; nx < VecSize; ++nx) {
      // get sin_index and cos_index
      int64_t index_wc = (index + nx) % (seq_len * num_heads * head_dim);
      int64_t pos_seq = index_wc / (num_heads * head_dim);
      MPType idx = static_cast<MPType>((index_wc % head_dim) / 2 * 2.0);
      MPType indicses =
          static_cast<MPType>(1) /
          pow(static_cast<MPType>(10000), idx * static_cast<MPType>(div_c));
      MPType value = pos_seq * indicses;
      sin_value[nx] = sin(value);
      cos_value[nx] = cos(value);
    }
  }
}

template <typename T, typename MPType, int VecSize = 2>
__global__ void VectorizedFusedRopeWithRotateEveryTwoKernel(
    phi::Array<const T*, 3> ins_data,
    phi::Array<const T*, 2> sin_cos_data,
    const int64_t* position_ids_data,
    bool flag_sin_cos,
    int sign,
    int64_t batch_size,
    int64_t seq_len,
    int64_t num_heads,
    int64_t head_dim,
    phi::Array<T*, 3> outs_data,
    int num_inputs,
    MPType div_c) {
  int64_t index =
      (static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
       threadIdx.x) *
      VecSize;
  int64_t stride = static_cast<int64_t>(gridDim.x) *
                   static_cast<int64_t>(blockDim.x) * VecSize;
  int64_t size = batch_size * seq_len * num_heads * head_dim;
  MPType sin_value[VecSize];
  MPType cos_value[VecSize];
  MPType result[VecSize];
  T store[VecSize];
  using VecType = phi::AlignedVector<T, VecSize>;
  constexpr int kVectorsPerThread = VecSize / 2;

  for (; index < size; index += stride) {
    VectorizedGetSinCos(sin_cos_data,
                        position_ids_data,
                        flag_sin_cos,
                        index,
                        seq_len,
                        num_heads,
                        head_dim,
                        sin_value,
                        cos_value,
                        div_c);

#pragma unroll
    for (int iter = 0; iter < 3; iter++) {
      if (iter > num_inputs) break;
      const T* input = ins_data[iter] + index;
      VecType* out = reinterpret_cast<VecType*>(outs_data[iter] + index);

#pragma unroll
      for (int nx = 0; nx < kVectorsPerThread; ++nx) {
        int pr_index = nx * 2;
        int ls_index = pr_index + 1;

        MPType p0 = static_cast<MPType>(input[pr_index]);
        MPType p1 = static_cast<MPType>(input[ls_index]);

        if (sign == 1) {
          result[pr_index] = cos_value[pr_index] * p0;
          result[pr_index] -= sin_value[pr_index] * p1;

          result[ls_index] = sin_value[ls_index] * p0;
          result[ls_index] += cos_value[ls_index] * p1;
        } else if (sign == -1) {
          result[pr_index] =
              cos_value[pr_index] * p0 + sin_value[ls_index] * p1;
          result[ls_index] =
              cos_value[ls_index] * p1 - sin_value[pr_index] * p0;
        }

        store[pr_index] = static_cast<T>(result[pr_index]);
        store[ls_index] = static_cast<T>(result[ls_index]);
      }
      out[0] = *(reinterpret_cast<VecType*>(store));
    }
  }
}

template <typename T, typename MPType, int VecSize = 2>
__global__ void VectorizedFusedRopeWithRotateHalfKernel(
    phi::Array<const T*, 3> ins_data,
    phi::Array<const T*, 2> sin_cos_data,
    const int64_t* position_ids_data,
    bool flag_sin_cos,
    int sign,
    int64_t batch_size,
    int64_t seq_len,
    int64_t num_heads,
    int64_t head_dim,
    phi::Array<T*, 3> outs_data,
    int num_inputs,
    MPType div_c) {
  int64_t index =
      (static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
       threadIdx.x) *
      VecSize;
  int64_t stride = static_cast<int64_t>(gridDim.x) *
                   static_cast<int64_t>(blockDim.x) * VecSize;
  int64_t size = batch_size * seq_len * num_heads * head_dim;
  MPType sin_value[VecSize];
  MPType cos_value[VecSize];
  MPType result[VecSize];
  T store[VecSize];
  using VecType = phi::AlignedVector<T, VecSize>;
  constexpr int kVectorsPerThread = VecSize / 2;

  for (; index < size; index += stride) {
    VectorizedGetSinCos(sin_cos_data,
                        position_ids_data,
                        flag_sin_cos,
                        index,
                        seq_len,
                        num_heads,
                        head_dim,
                        sin_value,
                        cos_value,
                        div_c);

    // use rotate_half mode
    int stride_r = head_dim / 2;
#pragma unroll
    for (int iter = 0; iter < 3; iter++) {
      if (iter > num_inputs) break;
      // get value_index and rotate_half_index
      int index_v = index;
      int index_r = (index % head_dim) < stride_r ? (index + stride_r)
                                                  : (index - stride_r);
      MPType sign_r = (index % head_dim) < stride_r ? static_cast<MPType>(-1)
                                                    : static_cast<MPType>(1);
      const T* input_v = ins_data[iter] + index_v;
      const T* input_r = ins_data[iter] + index_r;
      VecType* out = reinterpret_cast<VecType*>(outs_data[iter] + index);

#pragma unroll
      for (int nx = 0; nx < VecSize; ++nx) {
        MPType p0 = static_cast<MPType>(input_v[nx]);
        MPType p1 = static_cast<MPType>(input_r[nx]);

        result[nx] = cos_value[nx] * p0 + sign * sign_r * sin_value[nx] * p1;

        store[nx] = static_cast<T>(result[nx]);
      }
      out[0] = *(reinterpret_cast<VecType*>(store));
    }
  }
}

#ifdef __MUSACC__
// grid(s/TILE_S, b, 1)
// block(d / (VLEN * 2), TILE_S, 1)
template <int32_t VLEN, bool BWD = false>
__global__ void fusedRopeInterleaved(
    half *p_output,                // [b, s, h, d]
    const half *p_input,           // [b, s, h, d]
    const half *p_cos,      // [1, s, 1, d]
    const half *p_sin,      // [1, s, 1, d]
    const int64_t *p_position_ids,  // [b, s]
    const int32_t b, const int32_t s, const int32_t h, const int32_t d,
    const int64_t out_stride_b, const int64_t out_stride_s,
    const int64_t out_stride_h, const int64_t in_stride_b,
    const int64_t in_stride_s, const int64_t in_stride_h,
    const int64_t pos_stride_b) {
  int64_t position_id = blockIdx.x * blockDim.y + threadIdx.y;
  int64_t sin_cos_seq_id = position_id;
  if (position_id >= s) return;
  if (p_position_ids != nullptr) {
    sin_cos_seq_id = p_position_ids[blockIdx.y * pos_stride_b + position_id];
  }

  static_assert(VLEN >= 1 && VLEN <= 8 && ((VLEN & (VLEN - 1)) == 0), "");

  v8f16_t reg_out_l;
  v8f16_t reg_out_r;
  v8f16_t reg_in_l;
  v8f16_t reg_in_r;
  v8f16_t reg_sin_l;
  v8f16_t reg_cos_l;
  v8f16_t reg_sin_r;
  v8f16_t reg_cos_r;

  const int32_t half_d = d >> 1;

  int64_t offset_sin_cos_l = sin_cos_seq_id * d + threadIdx.x * VLEN;
  auto ptr_cos = p_cos + offset_sin_cos_l;
  auto ptr_sin = p_sin + offset_sin_cos_l;


  // reg_cos_l = *(v8f16_t *)(ptr_cos);
  // reg_cos_r = *(v8f16_t *)(ptr_cos + half_d);
  // reg_sin_l = *(v8f16_t *)(ptr_sin);
  // reg_sin_r = *(v8f16_t *)(ptr_sin + half_d);
  asm volatile(
    "DMA.LD.B128 %0, %4, _  \n\t"
    "DMA.LD.B128 %1, %5, _  \n\t"
    "DMA.LD.B128 %2, %4, %6 \n\t"
    "DMA.LD.B128 %3, %5, %6  "
    : "=&R"(reg_cos_l), "=&R"(reg_sin_l), "=&R"(reg_cos_r), "=R"(reg_sin_r)
    : "R"(ptr_cos), "R"(ptr_sin), "R"(d)
  );

  int64_t offset_in_l = blockIdx.y * in_stride_b + position_id * in_stride_s + threadIdx.x * VLEN;
  int64_t offset_out_l = blockIdx.y * out_stride_b + position_id * out_stride_s + threadIdx.x * VLEN;
  auto ptr_in = p_input + offset_in_l;
  auto ptr_out = p_output + offset_out_l;
#pragma unroll 1
  for (int32_t head_j = 0; head_j < h; head_j ++) {
    // reg_in_l = *(v8f16_t *)(ptr_in);
    // reg_in_r = *(v8f16_t *)(ptr_in + half_d);
    asm volatile(
      "DMA.LD.B128 %0, %2, _  \n\t"
      "DMA.LD.B128 %1, %2, %3 "
      : "=&R"(reg_in_l), "=R"(reg_in_r)
      : "R"(ptr_in), "R"((int32_t)(d))
    );
    
#pragma unroll
    for (int32_t vlen_k = 0; vlen_k < VLEN; ++vlen_k) {
      if constexpr(BWD == false) {
        reg_out_l[vlen_k] = float(reg_in_l[vlen_k]) * float(reg_cos_l[vlen_k]) -
                            float(reg_in_r[vlen_k]) * float(reg_sin_l[vlen_k]);
        reg_out_r[vlen_k] = float(reg_in_r[vlen_k]) * float(reg_cos_r[vlen_k]) +
                            float(reg_in_l[vlen_k]) * float(reg_sin_r[vlen_k]);
      } else {
        reg_out_l[vlen_k] = float(reg_in_l[vlen_k]) * float(reg_cos_l[vlen_k]) +
                            float(reg_in_r[vlen_k]) * float(reg_sin_r[vlen_k]);
        reg_out_r[vlen_k] = float(reg_in_r[vlen_k]) * float(reg_cos_r[vlen_k]) -
                            float(reg_in_l[vlen_k]) * float(reg_sin_l[vlen_k]);
      }
    }

    *(v8f16_t *)(ptr_out) = reg_out_l;
    *(v8f16_t *)(ptr_out + half_d) = reg_out_r;

    ptr_in += in_stride_h;
    ptr_out += out_stride_h;
  }
}
#endif

}  // namespace fusion
}  // namespace phi
