import modal

app = modal.App("iron-batch-real-backend")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "transformers",
        "accelerate",
        "torch",
        "fastapi",
    )
)

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
# Qwen2.5-3B max context is 32k tokens. Reject inputs that would exceed it.
MAX_CONTEXT_TOKENS = 32_000


@app.cls(gpu="A10", image=image, scaledown_window=300)
class Model:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.float16,
        )

    @modal.method()
    def generate_batch(self, token_ids: list[int], num_tokens: int) -> dict:
        """
        Generate up to num_tokens new tokens in a single call, stopping early
        on EOS. One HTTP call buys many decode steps instead of one.

        NOTE: this uses model.generate() which runs its own internal decode
        loop with KV cache reuse on the GPU side. Iron-batch's scheduler
        treats each token in the returned batch as one advance_token() call,
        but the actual GPU computation happened in one shot here — the
        scheduler's KV block accounting reflects logical token count, not
        GPU-side memory layout.
        """
        import torch
        from fastapi import HTTPException

        if len(token_ids) > MAX_CONTEXT_TOKENS:
            raise HTTPException(
                status_code=400,
                detail=f"token_ids length {len(token_ids)} exceeds max context {MAX_CONTEXT_TOKENS}"
            )

        if len(token_ids) == 0:
            raise HTTPException(status_code=400, detail="token_ids must not be empty")

        input_ids = torch.tensor([token_ids], device=self.model.device, dtype=torch.long)
        with torch.no_grad():
            out = self.model.generate(
                input_ids,
                max_new_tokens=num_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = out[0, input_ids.shape[1]:].tolist()
        hit_eos = self.tokenizer.eos_token_id in new_tokens
        if hit_eos:
            # Exclude the EOS token itself from the returned list — it's a
            # signal, not real output. RealBackend uses the eos flag to drain
            # the sequence cleanly without serving the EOS token id to the client.
            eos_idx = new_tokens.index(self.tokenizer.eos_token_id)
            new_tokens = new_tokens[:eos_idx]
        return {"token_ids": new_tokens, "eos": hit_eos}

    @modal.method()
    def warmup(self) -> dict:
        """Hit this once after deploy to pre-heat the container and avoid
        cold-start latency on the first real request."""
        result = self.generate_batch([1, 2, 3], num_tokens=1)
        return {"status": "warm", "eos": result["eos"]}


@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI
    from pydantic import BaseModel

    web_app = FastAPI()
    model = Model()

    class BatchRequest(BaseModel):
        token_ids: list[int]
        num_tokens: int = 32

    class BatchResponse(BaseModel):
        token_ids: list[int]
        eos: bool

    @web_app.post("/generate_batch", response_model=BatchResponse)
    def generate_batch(req: BatchRequest):
        result = model.generate_batch.remote(req.token_ids, req.num_tokens)
        return BatchResponse(**result)

    @web_app.get("/warmup")
    def warmup():
        return model.warmup.remote()

    @web_app.get("/health")
    def health():
        return {"status": "ok"}

    return web_app
