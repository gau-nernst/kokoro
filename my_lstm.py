import copy

import torch
import triton
import triton.language as tl
from torch import Tensor, nn
from torch.utils.benchmark import Timer
from triton.language.extra import libdevice


# assume all inputs are contiguous
@triton.autotune(
    configs=[
        triton.Config(
            dict(TILE_N=TILE_N, TILE_K=TILE_K),
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for TILE_N in [8, 16, 32]
        for TILE_K in [32, 64, 128]
        for num_stages in [3, 4, 5, 6]
        for num_warps in [4, 8]
    ],
    key=[],
    reset_to_zero=["C_ptr"],
    restore_value=["time_ptr"],
)
@triton.jit
def lstm_triton_kernel(
    X_ptr,  # (L, B, input_dim)
    C_ptr,  # (B, hidden_dim * 2)
    Y_ptr,  # (L, B, hidden_dim * 2)
    weight_ih_ptr,  # (hidden_dim * 4, input_dim)
    weight_hh_ptr,  # (hidden_dim * 4, hidden_dim)
    bias_ptr,  # (hidden_dim * 4)
    weight_ih_reverse_ptr,
    weight_hh_reverse_ptr,
    bias_reverse_ptr,
    time_ptr,
    L_ptr,
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

    # place in CUDA memory to use CUDA graph
    time = tl.load(time_ptr)
    L = tl.load(L_ptr)

    # select data based on direction
    if is_reverse == 0:
        X_ptr += time * B * input_dim
        Y_ptr += time * B * hidden_dim * 2
        H_ptr = Y_ptr - B * hidden_dim * 2  # previous timestep
        wih_ptr = weight_ih_ptr
        whh_ptr = weight_hh_ptr
        bias_ptr = bias_ptr
    else:
        X_ptr += (L - 1 - time) * B * input_dim
        C_ptr += hidden_dim
        Y_ptr += (L - 1 - time) * B * hidden_dim * 2 + hidden_dim
        H_ptr = Y_ptr + B * hidden_dim * 2  # next timestep
        wih_ptr = weight_ih_reverse_ptr
        whh_ptr = weight_hh_reverse_ptr
        bias_ptr = bias_reverse_ptr

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
    Bias = bias_ptr + offsets_n
    i += tl.load(Bias + hidden_dim * 0)
    f += tl.load(Bias + hidden_dim * 1)
    g += tl.load(Bias + hidden_dim * 2)
    o += tl.load(Bias + hidden_dim * 3)

    offsets = tl.arange(0, B)[:, None] * hidden_dim * 2 + offsets_n  # (B, TILE_N)
    c = tl.load(C_ptr + offsets)
    c = tl.sigmoid(f) * c + tl.sigmoid(i) * libdevice.tanh(g)
    h = tl.sigmoid(o) * libdevice.tanh(c)
    tl.store(C_ptr + offsets, c)
    tl.store(Y_ptr + offsets, h)

    # last block update time_ptr
    if pid == tl.num_programs(0) - 1 and tl.program_id(1) == 1:
        tl.store(time_ptr, time + 1)


def lstm_triton(x: Tensor, weights: list[Tensor], c: Tensor, time: Tensor, length: Tensor, out: Tensor, L: int):
    def grid(meta):
        return (meta["hidden_dim"] // meta["TILE_N"], 2)

    _, B, input_dim = x.shape
    hidden_dim = c.shape[1] // 2
    for _ in range(L):
        lstm_triton_kernel[grid](x, c, out, *weights, time, length, B, input_dim, hidden_dim)

    return out


class MyLSTM(nn.Module):
    def __init__(self, lstm: nn.LSTM, max_length: int = 512):
        super().__init__()
        assert lstm.num_layers == 1
        assert lstm.bidirectional
        self.max_length = max_length
        self.batch_first = lstm.batch_first

        self.weight_ih = nn.Parameter(lstm._flat_weights[0].detach().clone())
        self.weight_hh = nn.Parameter(lstm._flat_weights[1].detach().clone())
        self.bias = nn.Parameter(lstm._flat_weights[2].detach() + lstm._flat_weights[3].detach())
        self.weight_ih_reverse = nn.Parameter(lstm._flat_weights[4].detach().clone())
        self.weight_hh_reverse = nn.Parameter(lstm._flat_weights[5].detach().clone())
        self.bias_reverse = nn.Parameter(lstm._flat_weights[6].detach() + lstm._flat_weights[7].detach())

        self.weights = [
            self.weight_ih,
            self.weight_hh,
            self.bias,
            self.weight_ih_reverse,
            self.weight_hh_reverse,
            self.bias_reverse,
        ]

        # inputs/outputs to CUDA graph
        # NOTE: we use the same inputs/outputs buffers for different CUDA graphs
        B = 1
        input_dim = self.weight_ih.shape[1]
        hidden_dim = self.weight_hh.shape[1]
        self.x = torch.randn(max_length, B, input_dim, device="cuda")
        self.c = torch.randn(B, hidden_dim * 2, device="cuda")
        self.time = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.length = torch.full((1,), max_length, dtype=torch.int32, device="cuda")
        self.out = torch.randn(max_length, B, hidden_dim * 2, device="cuda")

        stream = torch.cuda.Stream()
        current_stream = torch.cuda.current_stream()
        self.graphs = dict()
        while max_length > 0:
            print(f"CUDA graph capture for length={max_length}")
            self.length.fill_(max_length)

            # warmup
            stream.wait_stream(current_stream)
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.time.fill_(0)
                    lstm_triton(self.x, self.weights, self.c, self.time, self.length, self.out, max_length)
            current_stream.wait_stream(stream)

            # graph capture
            self.time.fill_(0)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                lstm_triton(self.x, self.weights, self.c, self.time, self.length, self.out, max_length)

            self.graphs[max_length] = g
            max_length //= 2

    def forward(self, x: Tensor):
        if self.batch_first:
            x = x.transpose(0, 1)
        L, B, _ = x.shape
        assert B == 1
        assert L < self.max_length * 2
        self.x[:L] = x
        self.c.zero_()
        self.time.fill_(0)
        self.length.fill_(L)

        _L = L
        for length, g in self.graphs.items():
            if _L >= length:
                g.replay()
                _L -= length

        out = self.out[:L]
        if self.batch_first:
            out = out.transpose(0, 1)

        # partial signature compat with nn.LSTM
        return out, None


if __name__ == "__main__":
    D = 512
    m = nn.LSTM(D, D // 2, bidirectional=True, device="cuda")
    m.flatten_parameters()

    my_m = MyLSTM(m)

    with torch.no_grad():
        inputs = torch.randn(18, 1, D, device="cuda")
        out_ref, _ = m(inputs)

        out, _ = my_m(inputs)
        print((out_ref - out).abs().mean().item())

    inputs = torch.randn(200, 1, D, device="cuda")

    # input shape for m_graph can't be changed
    m_graph = copy.deepcopy(m)
    m_graph.flatten_parameters()
    torch.cuda.make_graphed_callables(m_graph, (inputs,))

    m0 = Timer("m(inputs)", globals=globals()).blocked_autorange()
    m1 = Timer("m_graph(inputs)", globals=globals()).blocked_autorange()
    m2 = Timer("my_m(inputs)", globals=globals()).blocked_autorange()

    print(f"CuDNN: {m0.median * 1e3:.2f} ms")
    print(f"CuDNN (CUDA graph): {m1.median * 1e3:.2f} ms")
    print(f"My LSTM: {m2.median * 1e3:.2f} ms")

    with torch.profiler.profile() as prof:
        m(inputs)
    prof.export_chrome_trace("cudnn.json.gz")

    with torch.profiler.profile() as prof:
        m_graph(inputs)
    prof.export_chrome_trace("cudnn_cudagraph.json.gz")

    with torch.profiler.profile() as prof:
        my_m(inputs)
    prof.export_chrome_trace("my_lstm.json.gz")
