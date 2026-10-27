import modal

MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_DIR = "/models/Qwen3-0.6B"

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .apt_install("git", "build-essential")
    .pip_install("torch==2.6.0", "triton", "transformers>=4.56", "xxhash", "huggingface_hub", "ninja", "packaging", "wheel")
    .pip_install("flash-attn==2.7.4.post1", extra_options="--no-build-isolation")
    .add_local_python_source("nanovllm")
)

app = modal.App("nano-vllm", image=image)
volume = modal.Volume.from_name("nanovllm-models", create_if_missing=True)


@app.function(volumes={"/models": volume}, timeout=1800)
def download():
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL_ID, local_dir=MODEL_DIR)
    volume.commit()


@app.function(gpu="L4", volumes={"/models": volume}, timeout=1800)
def generate(prompts: list[str], max_tokens: int = 256):
    from transformers import AutoTokenizer
    from nanovllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    prompts = [
        tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]
    llm = LLM(MODEL_DIR, enforce_eager=False, tensor_parallel_size=1)
    outputs = llm.generate(prompts, SamplingParams(temperature=0.6, max_tokens=max_tokens))
    return [o["text"] for o in outputs]


@app.local_entrypoint()
def main():
    download.remote()
    for text in generate.remote(["Introduce yourself.", "List all prime numbers within 100."]):
        print(text, "\n" + "-" * 40)
