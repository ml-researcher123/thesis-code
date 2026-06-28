"""Reader/generator backends."""

from __future__ import annotations

from dataclasses import dataclass

from .schema import RetrievalRun
from .text import lexical_overlap


@dataclass
class ExtractiveGenerator:
    """Offline reader that returns the most query-overlapping evidence sentence."""

    def answer(self, query: str, run: RetrievalRun) -> str:
        if not run.hits:
            return ""
        ranked = sorted(run.hits, key=lambda hit: lexical_overlap(query, hit.text), reverse=True)
        return ranked[0].text


class OpenAIGenerator:
    def __init__(self, model: str = "gpt-4.1-mini"):
        from openai import OpenAI

        self.model = model
        self.client = OpenAI()

    def answer(self, query: str, run: RetrievalRun) -> str:
        context = "\n\n".join(f"[{i + 1}] {hit.text}" for i, hit in enumerate(run.hits))
        prompt = (
            "Answer the question using only the cited evidence. If evidence conflicts, say so.\n\n"
            f"Question: {query}\n\nEvidence:\n{context}"
        )
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            temperature=0,
        )
        return response.output_text.strip()


class HuggingFaceGenerator:
    """Local open-weight reader for Kaggle/Colab experiments."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "cuda",
        batch_size: int = 4,
        max_new_tokens: int = 32,
        max_input_tokens: int = 2048,
        device_map: str | None = None,
        load_in_4bit: bool = False,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # Decoder-only models generate from the right edge of the prompt. Left
        # padding avoids generation shifts when batching variable-length prompts.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        model_kwargs = {"torch_dtype": dtype, "trust_remote_code": True}
        # Readers larger than ~3B do not fit a single 15GB T4 in fp16.
        # device_map="auto" shards the weights across all visible GPUs (e.g. the
        # 2xT4 Kaggle box); load_in_4bit quantizes to fit one GPU. With either,
        # accelerate/bitsandbytes place the weights, so we must NOT call
        # .to(device) afterwards (it raises for dispatched models).
        accelerate_placed = False
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["device_map"] = device_map or "auto"
            accelerate_placed = True
        elif device_map is not None:
            model_kwargs["device_map"] = device_map
            accelerate_placed = True

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if not accelerate_placed:
            self.model.to(device)
        self.model.eval()

        # Tensor entry point: with a sharded/quantized model the first
        # parameter's device (usually cuda:0) is where inputs go; accelerate
        # hooks then relay activations across shards during the forward pass.
        try:
            self.input_device = next(self.model.parameters()).device
        except StopIteration:
            self.input_device = torch.device(device)
        print(
            f"[reader] loaded {model_name} "
            f"(device_map={model_kwargs.get('device_map')}, 4bit={load_in_4bit}) "
            f"entry={self.input_device}",
            flush=True,
        )

    def answer(self, query: str, run: RetrievalRun) -> str:
        return self.answer_many([(query, run)])[0]

    def answer_many(self, items: list[tuple[str, RetrievalRun]]) -> list[str]:
        outputs: list[str] = []
        for start in range(0, len(items), self.batch_size):
            batch = items[start : start + self.batch_size]
            prompts = [self._format_prompt(query, run) for query, run in batch]
            outputs.extend(self._generate_batch(prompts))
            print(f"[reader] generated {min(start + len(batch), len(items))}/{len(items)}", flush=True)
        return outputs

    def _format_prompt(self, query: str, run: RetrievalRun) -> str:
        evidence = "\n".join(f"[{idx + 1}] {hit.text}" for idx, hit in enumerate(run.hits))
        messages = [
            {
                "role": "system",
                "content": (
                    "You answer questions using only the provided evidence. "
                    "Give the shortest correct answer phrase. Do not explain. "
                    "If the evidence is insufficient or conflicting, answer unknown."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {query}\n\nEvidence:\n{evidence}\n\nShort answer:",
            },
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return f"System: {messages[0]['content']}\nUser: {messages[1]['content']}\nAssistant:"

    def _generate_batch(self, prompts: list[str]) -> list[str]:
        import torch

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_tokens,
        )
        inputs = {key: value.to(self.input_device) for key, value in inputs.items()}
        input_width = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        decoded = self.tokenizer.batch_decode(generated[:, input_width:], skip_special_tokens=True)
        return [self._clean_answer(text) for text in decoded]

    @staticmethod
    def _clean_answer(text: str) -> str:
        text = text.strip()
        for marker in ("\n", "Question:", "Evidence:", "Short answer:"):
            if marker in text:
                text = text.split(marker, 1)[0].strip()
        return text.strip(" .")
