import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor, nn
from torch.utils.benchmark import Timer
from triton.language.extra import libdevice


@torch.no_grad()
def lstm_ref(x: Tensor, weights: list[Tensor]):
    # x: (L, B, D)
    (
        weight_ih,
        weight_hh,
        bias_ih,
        bias_hh,
        weight_ih_reverse,
        weight_hh_reverse,
        bias_ih_reverse,
        bias_hh_reverse,
    ) = weights

    hidden_size = weight_hh.shape[0] // 4
    L, B = x.shape[:2]
    out = torch.zeros(L, B, 2, hidden_size, device=x.device, dtype=x.dtype)

    # if we make a reverse copy of x, we can do this in 1 pass
    h = torch.zeros(B, hidden_size, device=x.device)
    c = torch.zeros(B, hidden_size, device=x.device)
    for t in range(L):
        # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        ifgo_x = F.linear(x[t], weight_ih, bias_ih)
        ifgo_h = F.linear(h, weight_hh, bias_hh)
        i, f, g, o = (ifgo_x + ifgo_h).chunk(4, dim=-1)
        c = f.sigmoid() * c + i.sigmoid() * g.tanh()
        h = o.sigmoid() * c.tanh()
        out[t, :, 0] = h

    # if we do 2 direction at the same time, h and h_reverse can be used in bmm
    h = torch.zeros(B, hidden_size, device=x.device)
    c = torch.zeros(B, hidden_size, device=x.device)
    for t in range(L - 1, -1, -1):
        ifgo_x = F.linear(x[t], weight_ih_reverse, bias_ih_reverse)
        ifgo_h = F.linear(h, weight_hh_reverse, bias_hh_reverse)
        i, f, g, o = (ifgo_x + ifgo_h).chunk(4, dim=-1)
        c = f.sigmoid() * c + i.sigmoid() * g.tanh()
        h = o.sigmoid() * c.tanh()
        out[t, :, 1] = h

    return out.view(L, B, -1)


# assume all inputs are contiguous
@triton.autotune(
    configs=[
        triton.Config(
            dict(TILE_N=TILE_N, TILE_K=TILE_K),
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for TILE_N in [16, 32, 64]
        for TILE_K in [32, 64, 128]
        for num_stages in [3, 4, 5, 6]
        for num_warps in [4, 8]
    ],
    key=[],
    reset_to_zero=["C_ptr"],
)
@triton.jit
def lstm_triton_kernel(
    X_ptr,  # (L, B, input_dim)
    C_ptr,  # (2, B, hidden_dim)
    Y_ptr,  # (L, B, hidden_dim * 2)
    weight_ih_ptr,  # (hidden_dim * 4, input_dim)
    weight_hh_ptr,  # (hidden_dim * 4, hidden_dim)
    bias_ih_ptr,  # (hidden_dim * 4)  -> these 2 can be combined
    bias_hh_ptr,  # (hidden_dim * 4)
    weight_ih_reverse_ptr,
    weight_hh_reverse_ptr,
    bias_ih_reverse_ptr,
    bias_hh_reverse_ptr,
    time: int,
    L: int,
    B: tl.constexpr,
    input_dim: tl.constexpr,
    hidden_dim: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
):
    tl.static_assert(hidden_dim % TILE_N == 0)
    tl.static_assert(input_dim % TILE_K == 0)
    tl.static_assert(hidden_dim % TILE_K == 0)

    # this impl is for small B (<4) -> use CUDA cores instead of Tensor Cores
    # forward and reverse can be done independently. use 1 kernel for both to reduce kernel launch overhead
    # and improve SM utilization w/o using separate CUDA streams.
    pid = tl.program_id(0)
    is_reverse = tl.program_id(1)

    # select data based on direction
    if is_reverse == 0:
        X_ptr += time * B * input_dim
        Y_ptr += time * B * hidden_dim * 2
        H_ptr = Y_ptr - B * hidden_dim * 2  # previous timestep
        wih_ptr = weight_ih_ptr
        whh_ptr = weight_hh_ptr
        bih_ptr = bias_ih_ptr
        bhh_ptr = bias_hh_ptr
    else:
        X_ptr += (L - 1 - time) * B * input_dim
        C_ptr += B * hidden_dim
        Y_ptr += (L - 1 - time) * B * hidden_dim * 2 + hidden_dim
        H_ptr = Y_ptr + B * hidden_dim * 2  # next timestep
        wih_ptr = weight_ih_reverse_ptr
        whh_ptr = weight_hh_reverse_ptr
        bih_ptr = bias_ih_reverse_ptr
        bhh_ptr = bias_hh_reverse_ptr

    i = tl.zeros((B, TILE_N), dtype=tl.float32)
    f = tl.zeros((B, TILE_N), dtype=tl.float32)
    g = tl.zeros((B, TILE_N), dtype=tl.float32)
    o = tl.zeros((B, TILE_N), dtype=tl.float32)

    offsets_m = tl.arange(0, B)[:, None, None]
    offsets_n = pid * TILE_N + tl.arange(0, TILE_N)[None, :, None]
    offsets_k = tl.arange(0, TILE_K)[None, None, :]
    X = X_ptr + (offsets_m * input_dim + offsets_k)  # (B, 1, TILE_K)
    Wih = wih_ptr + (offsets_n * input_dim + offsets_k)  # (1, TILE_N, TILE_K)

    # TODO: try interleave format for Wih -> do load and compute together for ifgo
    # not sure how we can calculate the final output afterwards
    for _ in range(input_dim, 0, -TILE_K):
        x = tl.load(X)
        i += tl.sum(x * tl.load(Wih + input_dim * hidden_dim * 0), axis=-1)
        f += tl.sum(x * tl.load(Wih + input_dim * hidden_dim * 1), axis=-1)
        g += tl.sum(x * tl.load(Wih + input_dim * hidden_dim * 2), axis=-1)
        o += tl.sum(x * tl.load(Wih + input_dim * hidden_dim * 3), axis=-1)
        X += TILE_K
        Wih += TILE_K

    if time > 0:
        offsets_m = tl.arange(0, B)[:, None, None]
        offsets_n = pid * TILE_N + tl.arange(0, TILE_N)[None, :, None]
        offsets_k = tl.arange(0, TILE_K)[None, None, :]
        H = H_ptr + (offsets_m * hidden_dim * 2 + offsets_k)  # (B, 1, TILE_K)
        Whh = whh_ptr + (offsets_n * hidden_dim + offsets_k)  # (1, TILE_N, TILE_K)

        for _ in range(hidden_dim, 0, -TILE_K):
            h = tl.load(H)
            i += tl.sum(h * tl.load(Whh + hidden_dim * hidden_dim * 0), axis=-1)
            f += tl.sum(h * tl.load(Whh + hidden_dim * hidden_dim * 1), axis=-1)
            g += tl.sum(h * tl.load(Whh + hidden_dim * hidden_dim * 2), axis=-1)
            o += tl.sum(h * tl.load(Whh + hidden_dim * hidden_dim * 3), axis=-1)
            H += TILE_K
            Whh += TILE_K

    offsets_n = (pid * TILE_N + tl.arange(0, TILE_N))[None, :]  # (1, TILE_N)
    Bih = bih_ptr + offsets_n
    Bhh = bhh_ptr + offsets_n
    i += tl.load(Bih + hidden_dim * 0) + tl.load(Bhh + hidden_dim * 0)
    f += tl.load(Bih + hidden_dim * 1) + tl.load(Bhh + hidden_dim * 1)
    g += tl.load(Bih + hidden_dim * 2) + tl.load(Bhh + hidden_dim * 2)
    o += tl.load(Bih + hidden_dim * 3) + tl.load(Bhh + hidden_dim * 3)

    offsets = tl.arange(0, B)[:, None] * hidden_dim + offsets_n  # (B, TILE_N)
    c = tl.load(C_ptr + offsets)
    c = tl.sigmoid(f) * c + tl.sigmoid(i) * libdevice.tanh(g)
    h = tl.sigmoid(o) * libdevice.tanh(c)
    tl.store(C_ptr + offsets, c)

    offsets = tl.arange(0, B)[:, None] * hidden_dim * 2 + offsets_n  # (B, TILE_N)
    tl.store(Y_ptr + offsets, h)


def lstm_triton(x: Tensor, weights: list[Tensor]):
    # x: (L, B, D)
    L, B, input_dim = x.shape
    hidden_dim = weights[1].shape[1]
    out = x.new_empty(L, B, hidden_dim * 2)
    c = x.new_empty(2, B, hidden_dim)  # triton pre-hook will zero this out

    def grid(meta):
        return (meta["hidden_dim"] // meta["TILE_N"], 2)

    for time in range(L):
        lstm_triton_kernel[grid](x, c, out, *weights, time, L, B, input_dim, hidden_dim)

    return out


class CUDAGraphWrapper:
    def __init__(self, x: Tensor, weights: Tensor):
        # x: (L, B, D)
        self.g = torch.cuda.CUDAGraph()
        self.x = torch.randn_like(x)

        s = torch.cuda.Stream()
        current_stream = torch.cuda.current_stream()

        # warmup
        s.wait_stream(current_stream)
        with torch.cuda.stream(s):
            for _ in range(3):
                lstm_triton(self.x, weights)
        current_stream.wait_stream(s)

        # capture graph
        with torch.cuda.graph(self.g):
            self.out = lstm_triton(self.x, weights)

        # x-forward: [max_L, 1, D]
        # x-reverse: [max_L, 1, D]
        # out-forward: [max_L, 1, D]
        # out-reverse: [max_L, 1, D]
        # copy into x-forward and x-reverse
        # copy out of out-forward and out-reverse
        #
        # replay: 0->128. need to specify start and stop

    def __call__(self, x: Tensor):
        self.x.copy_(x)
        self.g.replay()
        return self.out


if __name__ == "__main__":
    D = 512
    m = nn.LSTM(D, D // 2, bidirectional=True, device="cuda")
    m.flatten_parameters()
    lstm_ref_compiled = torch.compile(lstm_ref)

    # NOTE: PyTorch/CuDNN arranges weights in this order
    # weight_ih_l0
    # weight_hh_l0
    # weight_ih_l0_reverse
    # weight_hh_l0_reverse
    # bias_ih_l0
    # bias_hh_l0
    # bias_ih_l0_reverse
    # bias_hh_l0_reverse

    with torch.no_grad():
        inputs = torch.randn(1, 1, D, device="cuda")
        out_ref, _ = m(inputs)

        out = lstm_ref(inputs, m._flat_weights)
        print((out_ref - out).abs().mean().item())

        out = lstm_ref_compiled(inputs, m._flat_weights)
        print((out_ref - out).abs().mean().item())

        out = lstm_triton(inputs, m._flat_weights)
        print((out_ref - out).abs().mean().item())

    inputs = torch.randn(128, 1, D, device="cuda")
    lstm_triton_cudagraph = CUDAGraphWrapper(inputs, m._flat_weights)
    m0 = Timer("m(inputs)", globals=globals()).blocked_autorange()
    m1 = Timer("lstm_ref(inputs, m._flat_weights)", globals=globals()).blocked_autorange()
    m2 = Timer("lstm_ref_compiled(inputs, m._flat_weights)", globals=globals()).blocked_autorange()
    m3 = Timer("lstm_triton(inputs, m._flat_weights)", globals=globals()).blocked_autorange()
    m4 = Timer("lstm_triton_cudagraph(inputs)", globals=globals()).blocked_autorange()

    # TODO: CuDNN with CUDAGraph
    print(f"CuDNN: {m0.median * 1e3:.2f} ms")
    print(f"Reference: {m1.median * 1e3:.2f} ms")
    print(f"Reference (compiled): {m2.median * 1e3:.2f} ms")
    print(f"Triton: {m3.median * 1e3:.2f} ms")
    print(f"Triton CUDAGraph: {m4.median * 1e3:.2f} ms")

    with torch.profiler.profile() as prof:
        m(inputs)
    prof.export_chrome_trace("cudnn.json.gz")

    with torch.profiler.profile() as prof:
        lstm_ref_compiled(inputs, m._flat_weights)
    prof.export_chrome_trace("ref_compiled.json.gz")

    with torch.profiler.profile() as prof:
        lstm_triton(inputs, m._flat_weights)
    prof.export_chrome_trace("triton.json.gz")

    with torch.profiler.profile() as prof:
        lstm_triton_cudagraph(inputs)
    prof.export_chrome_trace("triton_cudagraph.json.gz")
