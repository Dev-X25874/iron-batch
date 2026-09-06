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


@app.cls(gpu="A10", image=image, scaledown_window=300)
class Model:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Float16, unquantized - matches the vLLM baseline precision so the
        # comparison isn't confounded by quantization differences anymore.
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.float16,
        )

    @modal.method()
    def next_token(self, token_ids: list[int]) -> dict:
        """Legacy single-step call. Kept for compatibility, no longer used
        by RealBackend on the hot path."""
        import torch

        input_ids = torch.tensor([token_ids], device=self.model.device, dtype=torch.long)
        with torch.no_grad():
            out = self.model(input_ids)
        next_id = int(torch.argmax(out.logits[0, -1, :]).item())
        eos = next_id == self.tokenizer.eos_token_id
        return {"token_id": next_id, "eos": eos}

    @modal.method()
    def generate_batch(self, token_ids: list[int], num_tokens: int) -> dict:
        """
        Generate up to num_tokens new tokens in a single call, stopping early
        on EOS. One HTTP call now buys many decode steps instead of one.
        """
        import torch

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
            new_tokens = new_tokens[: new_tokens.index(self.tokenizer.eos_token_id) + 1]
        return {"token_ids": new_tokens, "eos": hit_eos}


@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI
    from pydantic import BaseModel

    web_app = FastAPI()
    model = Model()

    class AdvanceRequest(BaseModel):
        token_ids: list[int]

    class AdvanceResponse(BaseModel):
        token_id: int
        eos: bool

    class BatchRequest(BaseModel):
        token_ids: list[int]
        num_tokens: int = 32

    class BatchResponse(BaseModel):
        token_ids: list[int]
        eos: bool

    @web_app.post("/advance", response_model=AdvanceResponse)
    def advance(req: AdvanceRequest):
        result = model.next_token.remote(req.token_ids)
        return AdvanceResponse(**result)

    @web_app.post("/generate_batch", response_model=BatchResponse)
    def generate_batch(req: BatchRequest):
        result = model.generate_batch.remote(req.token_ids, req.num_tokens)
        return BatchResponse(**result)

    @web_app.get("/health")
    def health():
        return {"status": "ok"}

    return web_app
