#!/usr/bin/env python
# coding=utf-8
import paddle
from paddle.incubate.nn.functional import fused_rotary_position_embedding
import time

def rope():
    paddle.device.cuda.empty_cache()
    paddle.seed(1204)
    batch_size = 1
    seq_len = 2048
    num_head = 40
    head_dim = 128
    q = paddle.randn([batch_size, seq_len, num_head, head_dim], dtype='float16')
    k = paddle.randn([batch_size, seq_len, num_head, head_dim], dtype='float16')
    x = paddle.randn([1, seq_len, 1, head_dim], dtype='float16')
    y = paddle.randn([1, seq_len, 1, head_dim], dtype='float16')
    sin = paddle.sin(x)
    cos = paddle.cos(y)
    print(q)
    print(k)
    print(sin)
    print(cos)
    count = 1
    start = time.time()
    # for i in range(count):
    out_q, out_k, out_v = fused_rotary_position_embedding(q, k, v=None, sin=sin, cos=cos, position_ids=None, use_neox_rotary_style=False)
    print(time.time() - start)
    paddle.device.cuda.synchronize()
    print(out_q)
    print(out_k)
    # paddle.save(out_q, "out_q.gold.tensor")
    # paddle.save(out_k, "out_k.gold.tensor")
    # print(out_v)
    # musa_q = paddle.load("out_q.tensor")
    # musa_k = paddle.load("out_k.tensor")

    # cuda_q = paddle.load("out_q.gold.tensor")
    # cuda_k = paddle.load("out_k.gold.tensor")
    # print(cuda_q)
    # print(cuda_k)
    # print(cuda_q - out_q)
    # print(cuda_k - out_k)

    # print((musa_q - cuda_q).sum())
    # print((musa_k - cuda_k).sum())

if __name__ == "__main__":
    rope()
