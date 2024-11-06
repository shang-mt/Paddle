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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "paddle/phi/kernels/fusion/gpu/fused_rope_utils.h"

namespace phi {
namespace fusion {

template <typename T, typename Context>
void FusedRopeGradKernel(const Context& dev_ctx,
                         const paddle::optional<DenseTensor>& sin,
                         const paddle::optional<DenseTensor>& cos,
                         const paddle::optional<DenseTensor>& position_ids,
                         const DenseTensor& dout_q,
                         const paddle::optional<DenseTensor>& dout_k,
                         const paddle::optional<DenseTensor>& dout_v,
                         bool use_neox_rotary_style,
                         DenseTensor* dq,
                         DenseTensor* dk,
                         DenseTensor* dv) {
  int64_t numel = dout_q.numel();
  if (numel <= 0) return;
  dev_ctx.template Alloc<T>(dq);
  // small size for broadcast
  auto batch_size = dout_q.dims()[0];
  auto num_heads = dout_q.dims()[2];
  auto head_dim = dout_q.dims()[3];
  auto seq_len = dout_q.dims()[1];
  PADDLE_ENFORCE_NE(head_dim % 2,
                    1,
                    phi::errors::InvalidArgument(
                        "The head_dim of input must be a multiple of 2."));

  constexpr const int vec_size = 2;

  auto config =
      phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, numel, vec_size);

  int64_t grid = config.block_per_grid.x;
  int64_t block = config.thread_per_block.x;
  auto stream = dev_ctx.stream();

  phi::Array<T*, 3> outs_data;
  phi::Array<const T*, 3> ins_data;
  phi::Array<const T*, 2> sin_cos_data;
  const int64_t* position_ids_data = NULL;

  ins_data[0] = dout_q.data<T>();
  outs_data[0] = dq->data<T>();
  int num_inputs = 0;

  if (dout_k.get_ptr()) {
    dev_ctx.template Alloc<T>(dk);
    outs_data[1] = dk->data<T>();
    ins_data[1] = dout_k->data<T>();
    num_inputs++;
  }

  if (dout_v.get_ptr()) {
    dev_ctx.template Alloc<T>(dv);
    outs_data[2] = dv->data<T>();
    ins_data[2] = dout_v->data<T>();
    num_inputs++;
  }

  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  MPType div_c = static_cast<MPType>(1.0f / head_dim);

  bool flag_sin_cos = false;
  if (sin.get_ptr() && cos.get_ptr()) {
    sin_cos_data[0] = sin->data<T>();
    sin_cos_data[1] = cos->data<T>();

    flag_sin_cos = true;

    if (position_ids.get_ptr()) {
      position_ids_data = position_ids->data<int64_t>();
    }
  }

  int sign = -1;
  if (use_neox_rotary_style) {
    VectorizedFusedRopeWithRotateEveryTwoKernel<T, MPType, vec_size>
        <<<grid, block, 0, stream>>>(ins_data,
                                     sin_cos_data,
                                     position_ids_data,
                                     flag_sin_cos,
                                     sign,
                                     batch_size,
                                     seq_len,
                                     num_heads,
                                     head_dim,
                                     outs_data,
                                     num_inputs,
                                     div_c);
  } else {
#ifdef __MUSACC__
  int32_t musa_batch_size[3] = {0};
  int32_t musa_seq_len[3] = {0};
  int32_t musa_num_heads[3] = {0};
  int32_t musa_head_dim[3] = {0};
  musa_batch_size[0] = dout_q.dims()[0];
  musa_seq_len[0] = dout_q.dims()[1];
  musa_num_heads[0] = dout_q.dims()[2];
  musa_head_dim[0] = dout_q.dims()[3];
  if(dout_k.get_ptr()) {
    musa_batch_size[1] = dout_k->dims()[0];
    musa_num_heads[1] = dout_k->dims()[2];
    musa_seq_len[1] = dout_k->dims()[1];
    musa_head_dim[1] = dout_k->dims()[3];
  }
  if(dout_v.get_ptr()) {
    musa_batch_size[2] = dout_v->dims()[0];
    musa_num_heads[2] = dout_v->dims()[2];
    musa_seq_len[2] = dout_v->dims()[1];
    musa_head_dim[2] = dout_v->dims()[3];
  }
  if(flag_sin_cos && (std::is_same<T, float16>::value || std::is_same<T, half>::value) && musa_head_dim[0]%16 == 0 && musa_head_dim[1]%16 == 0 && musa_head_dim[2]%16 == 0) {
    const int32_t v_len = 8;

    for(int i = 0;i <= num_inputs; i++) {
      const int32_t block_dim_x = musa_head_dim[i] / (2*v_len);
      const int32_t tile_s = (512 + block_dim_x - 1) / block_dim_x;
      
      half *musa_input_data = (half *)ins_data[i];
      half *musa_output_data = (half *)outs_data[i];
      half *sin_data = (half *)sin_cos_data[0];
      half *cos_data =  (half *)sin_cos_data[1];
      const int64_t in_stride_b = musa_seq_len[i] * musa_num_heads[i] * musa_head_dim[i];
      const int64_t in_stride_s = musa_num_heads[i] * musa_head_dim[i];
      const int64_t in_stride_h = musa_head_dim[i];
      const int64_t pos_stride_b = musa_seq_len[i];

      dim3 musa_block(block_dim_x, tile_s, 1);
      dim3 musa_grid((musa_seq_len[i]+tile_s-1)/tile_s, musa_batch_size[i], 1);
      
      fusedRopeInterleaved<8, true>
            <<<musa_grid, musa_block, 0, stream>>>(musa_output_data,
                                                   musa_input_data, 
                                                   cos_data, 
                                                   sin_data, 
                                                   position_ids_data, 
                                                   musa_batch_size[i],
                                                   musa_seq_len[i],
                                                   musa_num_heads[i], 
                                                   musa_head_dim[i], 
                                                   in_stride_b, in_stride_s, in_stride_h, 
                                                   in_stride_b, in_stride_s, in_stride_h, 
                                                   pos_stride_b);
    }
  } else {
#endif
    VectorizedFusedRopeWithRotateHalfKernel<T, MPType, vec_size>
        <<<grid, block, 0, stream>>>(ins_data,
                                     sin_cos_data,
                                     position_ids_data,
                                     flag_sin_cos,
                                     sign,
                                     batch_size,
                                     seq_len,
                                     num_heads,
                                     head_dim,
                                     outs_data,
                                     num_inputs,
                                     div_c);
#ifdef __MUSACC__
  }
#endif
  }
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_rotary_position_embedding_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedRopeGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16){};
