# ruff: noqa
import torch
import tilelang
from tilelang import language as T
from utils import assert_tensors_similar


@tilelang.jit(       #* jit的部分会影响到最终生成的代码，out_idx的使用也会
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,        #! 禁用 TMA （topk稀疏读取kv不连续，如果使用TMA需要Padd）
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True, #! 禁用 warp specialization
    },
)
def sparse_mla_fwd(
    heads,
    dim,
    tail_dim,
    topk,
    kv_group=1,     #* GQA分组数，这是用MQA的方式，因此为 1
    sm_scale=None,
    is_causal=True,
    CP0=True,
    block_I=64,     #* tile分块大小
    num_stages=2,   #! 寄存器压力，为 2
    threads=256,
):
    assert dim == tilelang.math.next_power_of_2(dim), f"haven't check padding correctness yet, dim={dim}"
    assert tail_dim == tilelang.math.next_power_of_2(tail_dim), f"haven't check padding correctness yet, dim={tail_dim}"
    assert is_causal == True, "non-casual is not supported"
    assert topk % block_I == 0, "otherwise will load some index=0 thus causing wrong kv to be loaded"
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5 * 1.44269504  # log2(e)
    else:
        sm_scale = sm_scale * 1.44269504  # log2(e)

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    head_kv = heads // kv_group                              #! 统一定义shape
    q_shape = [batch, seq_len, heads, dim + tail_dim]           #* （输入） 512 + 64
    kv_shape = [batch, seq_len_kv, kv_group, dim + tail_dim]    #! （输入）注意使用 MQA 计算 Sparse Prefill (在softmax部分传入Mask实现稀疏化处理)
    o_shape = [batch, seq_len, heads, dim]                   #* （输出）ATTN-output
    indices_shape = [batch, seq_len, kv_group, topk]            #* （输入）已经得到的
    lse_shape = [batch, seq_len, heads]                      #*  (输出) LSE
    indices_dtype = T.int32
    dtype = T.bfloat16
    accum_dtype = T.float32
    #! Block配置
    G = kv_group
    H = head_kv
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    if padded_H != H:
        assert kv_group == 1, (
            "here we solve the H padding automatically, other wise you should handle Q copy and Output copy with your mask (when kv_group == 1, use g_i * padded_H:(g_i+1) * padded_H would be handled automatically)"
        )
    BI = block_I                        # 64，每次处理64个KV tokens
    NI = tilelang.cdiv(topk, block_I)   # 需要迭代的block数
    D = dim                             # 512
    D_tail = tail_dim                   # 64
    #! 头复制策略 (REPLICATE_H)：当 head_kv > 64 时，将头分布到多个线程块处理，每块处理 H_per_block = min(64, padded_H) 个头
    #! 利用 warp_group 的 分布式 SMEM 缓解反量化的 cuda core压力
    if head_kv > 64:
        assert head_kv % 64 == 0, "head_kv should be a multiple of 64"
        REPLICATE_H = head_kv // 64     #! 需要多少个64-head的block
    else:
        REPLICATE_H = 1

    H_per_block = padded_H if REPLICATE_H == 1 else 64

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),  # type: ignore   
        KV: T.Tensor(kv_shape, dtype),  # type: ignore
        Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
        Output: T.Tensor(o_shape, dtype),  # type: ignore
        Lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
    ):
        with T.Kernel(seq_len * REPLICATE_H, batch, kv_group, threads=threads) as (
            bx,  #! 序列位置 + 头复制数
            by,  #! Batch
            bz,  #! KV组索引 (MQA=1)
        ):  
            #! ===== 共享内存分配 =====
            Q_shared = T.alloc_shared([H_per_block, D], dtype)
            Q_tail_shared = T.alloc_shared([H_per_block, D_tail], dtype)

            KV_shared = T.alloc_shared([BI, D], dtype)
            K_tail_shared = T.alloc_shared([BI, D_tail], dtype)

            #* 输出缓冲区
            O_shared = T.alloc_shared([H_per_block, D], dtype)
            Lse_shared = T.alloc_shared([H_per_block], accum_dtype)

            #* Causal mask
            mask = T.alloc_fragment([BI], "bool")

            #! ===== 寄存器（Fragment）分配 =====
            #* 输出累加器
            acc_o = T.alloc_fragment([H_per_block, D], accum_dtype)

            #* Attention scores累加器
            acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
            S_shared = T.alloc_shared([H_per_block, BI], dtype)

            #! Online Softmax相关
            sumexp = T.alloc_fragment([H_per_block], accum_dtype)     #* 累积的exp和
            sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)   #* 当前block的exp和
            alpha = T.alloc_fragment([H_per_block], accum_dtype)      #* 重新归一化因子
            m_i = T.alloc_fragment([H_per_block], accum_dtype)        #* 当前最大值
            m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)   #* 之前的最大值

            #* 初始化
            T.fill(acc_o, 0)       #* 输出累加器清零
            T.fill(sumexp, 0)      #* exp和清零
            T.fill(m_i, -(2**30))  #* 最大值初始化为很小的负数 avoid -inf - inf to cause nan

            #! 计算当前block处理的索引
            b_i, g_i = by, bz
            s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)
            q_i = s_i
            max_kv_i = q_i  #* Causal mask: 只能attend到<=q_i的KV

            #! 计算当前block处理的head范围
            H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64)
            H1 = H0 + H_per_block
            # 例如：如果padded_H=128, REPLICATE_H=2
            #   第1个block: H0=0, H1=64
            #   第2个block: H0=64, H1=128

            T.copy(Q[b_i, s_i, H0:H1, :D], Q_shared)
            T.copy(Q[b_i, s_i, H0:H1, D:], Q_tail_shared)

            #! ===== 主循环：遍历top-k个tokens =====
                # i_i: 当前KV block索引（0 到 NI-1）
                # NI = ceil(topk / BI) = ceil(2048 / 64) = 32
                # num_stages=2: 使用2-stage pipeline
            for i_i in T.Pipelined(NI, num_stages=num_stages):

                #! === 步骤1：构建Causal Mask ===
                for bi_i in T.Parallel(BI):
                    # 检查Indices[b_i, s_i, g_i, i_i * BI + bi_i]是否满足causal约束
                    mask[bi_i] = Indices[b_i, s_i, g_i, i_i * BI + bi_i] <= max_kv_i
                    # mask[bi_i] = True if KV_index <= Query_index
                    # 例如：如果当前query在位置100，则只能attend到位置<=100的KV

                #! === 步骤2：加载KV到共享内存（稀疏访问）===
                for bi_i, d_i in T.Parallel(BI, D):
                    KV_shared[bi_i, d_i] = KV[b_i,         #* batch索引
                                             Indices[b_i, s_i, g_i, i_i * BI + bi_i], #* 从Indices读取KV位置
                                             g_i,          #* group索引
                                             d_i]          #* 维度索引
                        # 关键：这里的KV访问是稀疏的，由Indices指定
                        # 例如：Indices[0, 100, 0, :] = [99, 87, 95, 23, ...]
                        #      则依次加载KV[0, 99, 0, :], KV[0, 87, 0, :], ...
                for bi_i, d_i in T.Parallel(BI, D_tail):
                    K_tail_shared[bi_i, d_i] = KV[b_i,
                                                  Indices[b_i, s_i, g_i, i_i * BI + bi_i],
                                                  g_i, 
                                                  D + d_i] #* tail维度从D开始

                #! === 步骤3：初始化scores并应用mask ===
                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.if_then_else(mask[bi_i],   #* 如果mask=True
                                                      0,            #* 初始化为0（正常计算）
                                                      -T.infinity(acc_s.dtype)) #* 否则为-inf（softmax后为0)
                #! === 步骤4：计算Attention Scores（Q @ K^T）===
                    #* 主维度的GEMM
                T.gemm(
                    Q_shared,               # [H_per_block, D]
                    KV_shared,              # [BI, D]
                    acc_s,                  # [H_per_block, BI] (输出，累加模式)
                    transpose_B=True,       #* KV_shared转置为[D, BI]
                    policy=T.GemmWarpPolicy.FullRow,
                )                           #! FullCol: 每个warp处理完整的列维度

                    #* tail维度的GEMM（累加到acc_s）
                T.gemm(
                    Q_tail_shared,          # [H_per_block, D_tail]
                    K_tail_shared,          # [BI, D_tail]
                    acc_s,                  #! [H_per_block, BI] (继续累加)
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                #! 现在 acc_s[h, bi] = Q[h, :] @ K[bi, :]^T (完整的512+64维)

                #! === 步骤5：Online Softmax（FlashAttention核心）===
                    #* 5a. 保存之前的最大值
                T.copy(m_i, m_i_prev)  # m_i_prev[h]: 之前所有blocks的max(scores[h, :])

                    #* 5b. 计算当前block的最大值
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                                    # m_i[h] = max(m_i_prev[h], max(acc_s[h, :]))
                                    # clear=False: 不清空m_i，而是取max
                    #* 5c. 计算重新归一化因子
                for h_i in T.Parallel(H_per_block):
                    m_i[h_i] = T.max(m_i[h_i], m_i_prev[h_i])
                    
                    #* 5d. 计算当前block的softmax分子（exp(s - m)）
                for h_i in T.Parallel(H_per_block):
                    alpha[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    
                    acc_s[h_i, bi_i] = T.exp2(acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale)
                T.reduce_sum(acc_s, sumexp_i, dim=1)  # is this a accumulate operator?
                for h_i in T.Parallel(H_per_block):
                    sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                for h_i, d_i in T.Parallel(H_per_block, D):
                    acc_o[h_i, d_i] = acc_o[h_i, d_i] * alpha[h_i]

                T.copy(acc_s, S_shared)
                T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            # Rescale
            for h_i, d_i in T.Parallel(H_per_block, D):
                acc_o[h_i, d_i] /= sumexp[h_i]
            for h_i in T.Parallel(H_per_block):
                sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale

            T.copy(acc_o, O_shared)
            T.copy(acc_o, Output[b_i, s_i, H0:H1, :])
            T.copy(sumexp, Lse_shared)
            T.copy(sumexp, Lse[b_i, s_i, H0:H1])

    return main


def sparse_mla_fwd_interface(q, kv, indices, sm_scale=None, return_p_sum: bool = False, d_v=512, block_I=64, num_stages=2, threads=256):
    is_casual = True
    assert return_p_sum == False, "This kernel file is for fwd only"
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
    batch, seq_len, heads, dim_plus_tail_dim = q.shape
    _, seq_len_kv, kv_group, _ = kv.shape

    assert dim_plus_tail_dim == 576, "you should assign dim otherwise"
    dim = d_v

    assert kv.shape[-1] == dim_plus_tail_dim
    tail_dim = dim_plus_tail_dim - dim
    assert kv.shape[0] == batch
    _, _, _, topk = indices.shape
    assert indices.shape == (batch, seq_len, kv_group, topk)

    kernel = sparse_mla_fwd(
        heads, dim, tail_dim, topk, kv_group, sm_scale, is_casual, block_I=block_I, num_stages=num_stages, threads=threads
    )
    out, lse = kernel(q, kv, indices)
    return out, lse


def ref_sparse_mla_fwd_interface(q, kv, indices, sm_scale=None, is_casual=True):
    q = q.float()
    kv = kv.float()
    indices = indices.transpose(1, 2)
    b, sq, h, dim_q = q.shape
    b, sk, g, _ = kv.shape

    assert kv.shape[-1] == 576, "you should assign dim otherwise"
    dim = 512
    k = kv
    v = kv[..., :dim]

    b, _, _, dim_v = v.shape
    g_index = g
    h_index = h // g
    compressed_casual_mask = torch.arange(0, sq, dtype=torch.int32, device="cuda").view(-1, 1) >= torch.arange(
        1 - 1, sk * 1, 1, dtype=torch.int32, device="cuda"
    ).view(1, -1)

    mask = q.new_zeros(b, g_index, sq, sk + 1, dtype=torch.bool).scatter(3, indices.long(), 1)
    mask = mask[..., :-1]
    mask = mask & compressed_casual_mask.view(1, 1, sq, sk)
    mask[:, :, : 1 - 1, 0] = True
    mask = mask.view(b, g_index, 1, sq, sk)

    q = q.view(b, sq, g, -1, dim_q)
    score = torch.einsum("bmghd,bngd->bghmn", q, k)
    sm_scale = dim_q**-0.5 if sm_scale is None else sm_scale
    score = score.masked_fill(~mask, float("-inf")).mul(sm_scale)
    p = score.softmax(dim=-1)
    p = p.view(b, g_index, h_index, -1, sq, sk)
    p = p.view(b, g, -1, sq, sk)
    o = torch.einsum("bghmn,bngd->bmghd", p.type(v.dtype), v)
    o = o.reshape(b, sq, h, dim_v)
    return o.to(torch.bfloat16)


def test_sparse_mla_fwd(
    B=1,
    S=4096,
    SKV=8192,
    H=128,
    HKV=1,
    DQK=576,
    DV=512,
    topk=2048,
    dtype=torch.bfloat16,
    check_correctness=True,
    block_I=64,
    num_stages=2,
    threads=256,
):
    torch.random.manual_seed(0)
    q = torch.randn((B, S, H, DQK), dtype=dtype, device="cuda").requires_grad_(True)
    kv = torch.randn((B, SKV, HKV, DQK), dtype=dtype, device="cuda").requires_grad_(True)

    indices = torch.full((B, S, HKV, topk), SKV, dtype=torch.int32, device="cuda")
    for b in range(B):
        for t in range(S):
            for h in range(HKV):
                i_i = torch.randperm(max(1, t))[:topk]
                indices[b, t, h, : len(i_i)] = i_i

    tl_out, tl_lse = sparse_mla_fwd_interface(q, kv, indices, block_I=block_I, num_stages=num_stages, threads=threads)

    if check_correctness:
        # otherwise may cause out of memory
        ref_out = ref_sparse_mla_fwd_interface(q, kv, indices)
        assert_tensors_similar(tl_out, ref_out, eps=1e-2, name="out")
        print("assert_tensors_similar passed")

    def fn():
        return sparse_mla_fwd_interface(q, kv, indices, block_I=block_I, num_stages=num_stages, threads=threads)

    from tilelang.profiler import do_bench

    ms = do_bench(
        fn,
        rep=100,
        warmup=250,
    )
    print(f"Average time: {ms:.3f} ms")
    print("fwd io bandwidth = ", (B * S * DQK * topk * 2) / (ms * 1e-3) / 1e12)
    print("fwd tflops = ", (B * S * (DQK + DV) * topk * 2 * H) / (ms * 1e-3) / 1e12)


def run_regression_perf(
    B=1, S=4096, SKV=8192, H=128, HKV=1, DQK=576, DV=512, topk=2048, dtype=torch.bfloat16, block_I=64, num_stages=2, threads=256
):
    torch.random.manual_seed(0)
    q = torch.randn((B, S, H, DQK), dtype=dtype, device="cuda").requires_grad_(True)
    kv = torch.randn((B, SKV, HKV, DQK), dtype=dtype, device="cuda").requires_grad_(True)

    indices = torch.full((B, S, HKV, topk), SKV, dtype=torch.int32, device="cuda")
    for b in range(B):
        for t in range(S):
            for h in range(HKV):
                i_i = torch.randperm(max(1, t))[:topk]
                indices[b, t, h, : len(i_i)] = i_i

    is_casual = True
    _, _, heads, dim_plus_tail_dim = q.shape
    _, _, kv_group, _ = kv.shape
    dim = 512
    tail_dim = dim_plus_tail_dim - dim
    _, _, _, topk = indices.shape
    kernel = sparse_mla_fwd(heads, dim, tail_dim, topk, kv_group, None, is_casual, block_I=block_I, num_stages=num_stages, threads=threads)

    def run_kernel_only():
        kernel(q, kv, indices)

    from tilelang.profiler import do_bench

    return do_bench(run_kernel_only, backend="cupti")


if __name__ == "__main__":
    test_sparse_mla_fwd(
        B=1,
        S=4096,
        SKV=4096,
        H=128,
        HKV=1,
        DQK=576,
        DV=512,
        topk=2048,
        dtype=torch.bfloat16,
        check_correctness=True,
        block_I=64,
        num_stages=2,
        threads=256,
    )
