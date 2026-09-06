import modal

app = modal.App("vllm-compare")

image = (
    modal.Image.debian_slim()
    .pip_install("vllm", "fastapi")
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
)

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"


@app.cls(gpu="A10", image=image, scaledown_window=300)
class VLLMModel:
    @modal.enter()
    def load(self):
        from vllm import LLM
        self.llm = LLM(model=MODEL_NAME, dtype="float16")

    @modal.method()
    def generate(self, prompt: str, max_new_tokens: int) -> str:
        from vllm import SamplingParams
        params = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)
        out = self.llm.generate([prompt], params)
        return out[0].outputs[0].text


@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI
    from pydantic import BaseModel

    web_app = FastAPI()
    model = VLLMModel()

    class GenRequest(BaseModel):
        prompt: str
        max_new_tokens: int = 64

    @web_app.post("/generate")
    def generate(req: GenRequest):
        text = model.generate.remote(req.prompt, req.max_new_tokens)
        return {"text": text}

    return web_app
