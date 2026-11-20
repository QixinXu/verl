# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Token-in/token-out client for an external ``vllm serve`` process.

Lets the rollout-only harness run the *real model* without verl's training engine: start
``vllm serve <model>`` on a GPU and point ``+driver.llm=vllm_http +driver.base_url=...`` at it.
It implements the one method agent loops call on verl's ``LLMServerClient``: ``generate``, including
the ``prompt_logprobs`` request the influence measurement uses.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from verl.workers.rollout.replica import TokenOutput


class VLLMHttpClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        model_path: Optional[str] = None,
        max_concurrency: int = 64,
        timeout_s: float = 600.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.model_path = model_path or model
        self.max_concurrency = max_concurrency
        self.timeout_s = timeout_s
        self._session = None
        self._semaphore = None

    def _lazy_init(self):
        if self._session is None:
            import aiohttp

            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout_s))
            self._semaphore = asyncio.Semaphore(self.max_concurrency)

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        async with self._semaphore:
            for attempt in range(3):
                try:
                    async with self._session.post(f"{self.base_url}/completions", json=payload) as resp:
                        body = await resp.json()
                        if resp.status != 200:
                            raise RuntimeError(f"vllm server returned {resp.status}: {body}")
                    return body
                except Exception:
                    if attempt == 2:
                        raise
                    await asyncio.sleep(1.0 + attempt)

    async def _score(self, prompt_ids: list[int]) -> TokenOutput:
        """Log-probs of the prompt tokens themselves, in the layout of verl's vLLM server:
        ``prompt_logprobs[t - 1] = [log-prob of token t]`` plus one dummy row."""
        body = await self._post(
            {"model": self.model, "prompt": prompt_ids, "max_tokens": 1, "temperature": 1.0, "prompt_logprobs": 1}
        )
        rows = []
        for token_id, entry in zip(prompt_ids[1:], body["choices"][0]["prompt_logprobs"][1:], strict=True):
            # each entry maps token ids to their log-prob; the prompt token itself is always in it
            hit = entry.get(str(token_id)) or next(iter(entry.values()))
            rows.append([float(hit["logprob"])])
        return TokenOutput(token_ids=[], extra_fields={"prompt_logprobs": rows + [[0.0]]})

    async def generate(self, request_id, *, prompt_ids: list[int], sampling_params: dict[str, Any], **kwargs):
        self._lazy_init()
        if "prompt_logprobs" in sampling_params:
            return await self._score([int(t) for t in prompt_ids])
        payload = {
            "model": self.model,
            "prompt": [int(t) for t in prompt_ids],
            "max_tokens": int(sampling_params.get("max_tokens", 256)),
            "temperature": float(sampling_params.get("temperature", 1.0)),
            "top_p": float(sampling_params.get("top_p", 1.0)),
            # always sent: otherwise `vllm serve` falls back to the model's generation_config.json
            "top_k": int(sampling_params.get("top_k", -1) or -1),
            "repetition_penalty": float(sampling_params.get("repetition_penalty", 1.0)),
            "logprobs": 1,
            # tokens come back as "token_id:<id>", so nothing is ever re-encoded from text
            "return_tokens_as_token_ids": True,
            "skip_special_tokens": False,
        }
        body = await self._post(payload)
        choice = body["choices"][0]
        logprobs = choice["logprobs"]
        token_ids = [int(t.split(":")[1]) for t in logprobs["tokens"]]
        token_logprobs = [float(x) if x is not None else 0.0 for x in logprobs["token_logprobs"]]
        # If the server stripped the stop token, SkillAgentLoop closes the turn itself (as context,
        # mask 0); the client must not invent a generated token with a made-up logprob.
        return TokenOutput(
            token_ids=token_ids,
            log_probs=token_logprobs,
            stop_reason="completed" if choice.get("finish_reason") == "stop" else "length",
            num_preempted=0,
            extra_fields={"min_global_steps": 0, "max_global_steps": 0},
        )
