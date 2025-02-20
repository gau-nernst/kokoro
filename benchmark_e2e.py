import time

import torch
from datasets import load_dataset
from torch import nn
from torch.utils.benchmark import Timer

from kokoro import KPipeline
from my_lstm import MyLSTM
from kokoro_onnx import Kokoro

# to avoid recompile in benchmarking later
torch.set_num_threads(1)

ds = load_dataset("fka/awesome-chatgpt-prompts", split="train")
prompts = ds["prompt"]

# kokoro = Kokoro("kokoro-v1.0.fp16-gpu.onnx", "voices-v1.0.bin")

pipe = KPipeline("a")
pipe.model.bert.compile()
pipe.model.decoder.compile()
pipe.model.text_encoder.lstm = MyLSTM(pipe.model.text_encoder.lstm)
for i, m in enumerate(pipe.model.predictor.text_encoder.lstms):
    if isinstance(m, nn.LSTM):
        pipe.model.predictor.text_encoder.lstms[i] = MyLSTM(m)
pipe.model.predictor.lstm = MyLSTM(pipe.model.predictor.lstm)
pipe.model.predictor.shared = MyLSTM(pipe.model.predictor.shared, max_length=512 * 8)


def predict(text):
    out = [x for _, _, x in pipe(text, voice="af_heart")]
    return out[0]
    # samples, _ = kokoro.create(text, voice="af_heart")
    # return samples


for i in range(5):
    text = prompts[i]
    text = text[: len(text) // 4]
    predict(text)

for i in range(5, 10):
    text = prompts[i]
    text = text[: len(text) // 4]
    t0 = time.perf_counter()
    audio = predict(text)
    latency = time.perf_counter() - t0
    duration = audio.shape[0] / 24_000
    print(f"{duration:.1f}: {latency * 1000:.2f} ms")

text = prompts[31]
text = text[: len(text) // 4]
print(text)
latency = Timer("predict(text)", globals=globals()).blocked_autorange(min_run_time=1.0).median
audio = predict(text)
duration = audio.shape[0] / 24_000
print(f"{duration:.1f}: {latency  * 1000:.2f} ms")

text = prompts[31]
print(text)
latency = Timer("predict(text)", globals=globals()).blocked_autorange(min_run_time=1.0).median
with torch.profiler.profile() as prof:
    audio = predict(text)
prof.export_chrome_trace("trace_custom_lsm.json.gz")
duration = audio.shape[0] / 24_000
print(f"{duration:.1f}: {latency  * 1000:.2f} ms")
