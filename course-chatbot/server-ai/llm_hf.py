# import gc
# import logging
# import threading
# import time
# from typing import Any, Dict, List, Optional

# import torch
# from transformers import Pipeline, pipeline

# from config import HF_MODEL_NAME

# logger = logging.getLogger(__name__)

# _generator: Optional[Pipeline] = None
# _generator_lock = threading.Lock()

# DEFAULT_MAX_NEW_TOKENS = 350
# DEFAULT_DO_SAMPLE = False
# DEFAULT_TEMPERATURE = 0.2
# DEFAULT_TOP_P = 0.9
# DEFAULT_REPETITION_PENALTY = 1.05


# def _build_pipeline() -> Pipeline:
#     """
#     โหลด Hugging Face pipeline ครั้งเดียว
#     """
#     logger.info("Loading Hugging Face pipeline: %s", HF_MODEL_NAME)

#     pipe = pipeline(
#         task="text-generation",
#         model=HF_MODEL_NAME,
#         device_map="auto",
#         torch_dtype="auto",
#     )

#     tokenizer = pipe.tokenizer
#     if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
#         tokenizer.pad_token_id = tokenizer.eos_token_id

#     logger.info("Hugging Face pipeline loaded successfully")
#     return pipe


# def get_generator() -> Pipeline:
#     global _generator

#     if _generator is None:
#         with _generator_lock:
#             if _generator is None:
#                 _generator = _build_pipeline()

#     return _generator


# def unload_generator() -> None:
#     global _generator

#     with _generator_lock:
#         _generator = None

#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()

#     logger.info("Hugging Face pipeline unloaded")


# def _build_messages(system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
#     system_prompt = (system_prompt or "").strip()
#     user_prompt = (user_prompt or "").strip()

#     if not user_prompt:
#         raise ValueError("user_prompt must not be empty")

#     messages: List[Dict[str, str]] = []
#     if system_prompt:
#         messages.append({"role": "system", "content": system_prompt})
#     messages.append({"role": "user", "content": user_prompt})
#     return messages


# def _build_prompt_text(generator: Pipeline, messages: List[Dict[str, str]]) -> str:
#     """
#     พยายามใช้ chat template ของ tokenizer ก่อน
#     ถ้าไม่มี ให้ fallback เป็นข้อความธรรมดา
#     """
#     tokenizer = generator.tokenizer

#     apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
#     if callable(apply_chat_template):
#         try:
#             return tokenizer.apply_chat_template(
#                 messages,
#                 tokenize=False,
#                 add_generation_prompt=True,
#             )
#         except Exception:
#             logger.exception("apply_chat_template failed, fallback to manual prompt")

#     system_parts = [m["content"] for m in messages if m.get("role") == "system" and m.get("content")]
#     user_parts = [m["content"] for m in messages if m.get("role") == "user" and m.get("content")]

#     system_text = "\n".join(system_parts).strip()
#     user_text = "\n".join(user_parts).strip()

#     if system_text:
#         return f"System:\n{system_text}\n\nUser:\n{user_text}\n\nAssistant:\n"
#     return f"User:\n{user_text}\n\nAssistant:\n"


# def _safe_token_count(generator: Pipeline, text: str) -> Optional[int]:
#     try:
#         tokenizer = generator.tokenizer
#         encoded = tokenizer(
#             text,
#             add_special_tokens=False,
#             return_attention_mask=False,
#             return_token_type_ids=False,
#         )
#         input_ids = encoded.get("input_ids")
#         if isinstance(input_ids, list):
#             return len(input_ids)
#         return None
#     except Exception:
#         return None


# def _extract_generated_text(outputs: Any) -> str:
#     """
#     รองรับ output ของ text-generation pipeline
#     """
#     if not outputs or not isinstance(outputs, list):
#         raise RuntimeError("Invalid HF output: expected non-empty list")

#     first = outputs[0]
#     if not isinstance(first, dict):
#         raise RuntimeError("Invalid HF output: expected dict item")

#     generated = first.get("generated_text")

#     if isinstance(generated, str):
#         text = generated.strip()
#         if not text:
#             raise RuntimeError("HF returned empty text")
#         return text

#     if isinstance(generated, list):
#         for item in reversed(generated):
#             if isinstance(item, dict) and item.get("role") == "assistant":
#                 content = str(item.get("content", "")).strip()
#                 if content:
#                     return content
#         raise RuntimeError("Assistant response not found")

#     raise RuntimeError(f"Unsupported HF output format: {type(generated)!r}")


# def chat_with_hf(
#     system_prompt: str,
#     user_prompt: str,
#     *,
#     max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
#     do_sample: bool = DEFAULT_DO_SAMPLE,
#     temperature: float = DEFAULT_TEMPERATURE,
#     top_p: float = DEFAULT_TOP_P,
#     repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
# ) -> Dict[str, Any]:
#     """
#     ส่งข้อความเข้า Hugging Face model แล้วคืนค่าแนวเดียวกับ Ollama:
#     {
#       "content": "...",
#       "model": "...",
#       "done": True,
#       "provider": "hf",
#       "elapsed_ms": ...,
#       "input_tokens": ...,
#       "output_tokens": ...
#     }
#     """
#     generator = get_generator()
#     messages = _build_messages(system_prompt, user_prompt)
#     prompt_text = _build_prompt_text(generator, messages)

#     input_tokens = _safe_token_count(generator, prompt_text)

#     generate_kwargs: Dict[str, Any] = {
#         "max_new_tokens": max_new_tokens,
#         "repetition_penalty": repetition_penalty,
#         "return_full_text": False,
#         "pad_token_id": getattr(generator.tokenizer, "pad_token_id", None),
#         "eos_token_id": getattr(generator.tokenizer, "eos_token_id", None),
#     }

#     if do_sample:
#         generate_kwargs.update(
#             {
#                 "do_sample": True,
#                 "temperature": temperature,
#                 "top_p": top_p,
#             }
#         )
#     else:
#         generate_kwargs["do_sample"] = False

#     started = time.perf_counter()

#     try:
#         outputs = generator(prompt_text, **generate_kwargs)
#         content = _extract_generated_text(outputs)
#         elapsed_ms = int((time.perf_counter() - started) * 1000)
#         output_tokens = _safe_token_count(generator, content)

#         return {
#             "content": content,
#             "model": HF_MODEL_NAME,
#             "done": True,
#             "provider": "hf",
#             "elapsed_ms": elapsed_ms,
#             "input_tokens": input_tokens,
#             "output_tokens": output_tokens,
#         }

#     except Exception as exc:
#         logger.exception("Hugging Face generation failed")
#         raise RuntimeError(f"Hugging Face generation failed: {exc}") from exc