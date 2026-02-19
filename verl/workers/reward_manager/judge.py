import logging
import os
import re
import asyncio
from typing import Any, Optional
from uuid import uuid4

import torch
import numpy as np
from openai import AsyncOpenAI
from verl.workers.reward_manager.abstract import AbstractRewardManager, RawRewardFn
from verl.workers.reward_manager import register
from verl import DataProto
from verl.utils.tracking import Tracking

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, contains all the essential information from [correct_answer], is equivalent despite minor wording/order differences (such as name order, inclusion or omission of middle names/initials, common honorifics, standard shortenings of first names, inclusion/omission of non-contradictory date parts like year, minor articles like "a"/"the", extra descriptive context, non-essential descriptive prefixes/suffixes such as "Restaurant", "Inc.", "Ltd.", or sports suffixes like "FC", "CF", "SC", inclusion/omission of subtitles in titles, minor spacing/punctuation differences — including presence/absence of quotation marks, interchangeable punctuation such as ":" / "-" / "–", case-only differences, or presence/absence of diacritics), or is within a small margin of error for numerical problems. Answer 'no' only if the extracted answer is factually incorrect, missing essential identifying information, or contradicts the [correct_answer].

confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available.
""".strip()


@register("judge")
class JudgeRewardManager(AbstractRewardManager):
    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        **kwargs,
    ):
        super().__init__(tokenizer, num_examine, compute_score)
        self.tokenizer = tokenizer
        self.reward_fn_key = reward_fn_key

        self.judge_openai_api_key = kwargs.get("judge_openai_api_key", os.getenv("JUDGE_OPENAI_API_KEY"))
        self.judge_openai_base_url = kwargs.get("judge_openai_base_url", os.getenv("JUDGE_OPENAI_BASE_URL", "https://lonlie.plus7.plus/v1"))
        self.judge_openai_model = kwargs.get("judge_openai_model", os.getenv("JUDGE_OPENAI_MODEL", "gpt-4.1"))
        self.judge_openai_url = kwargs.get("judge_openai_url", os.getenv("JUDGE_OPENAI_URL", "https://lonlie.plus7.plus/v1/chat/completions"))

        self.max_retries = kwargs.get("max_retries", 3)
        self.timeout = kwargs.get("timeout", 30)
        self.semaphore_limit = kwargs.get("semaphore_limit", 50)

        self._clients = {}  # Store clients per event loop
        self._semaphores = {}  # Store semaphores per event loop

        logger.info(f"Initialized JudgeRewardManager with model={self.judge_openai_model}, timeout={self.timeout}s, semaphore_limit={self.semaphore_limit}")

    def _get_client(self):
        loop = asyncio.get_event_loop()
        loop_id = id(loop)
        
        if loop_id not in self._clients:
            self._clients[loop_id] = AsyncOpenAI(
                api_key=self.judge_openai_api_key,
                base_url=self.judge_openai_base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        return self._clients[loop_id]

    def _get_semaphore(self):
        loop = asyncio.get_event_loop()
        loop_id = id(loop)
        
        if loop_id not in self._semaphores:
            self._semaphores[loop_id] = asyncio.Semaphore(self.semaphore_limit)
        
        return self._semaphores[loop_id]

    async def _call_judge(self, prompt: str, max_retries: int = 3) -> dict:
        print(f'[DEBUG] _call_judge called with prompt length: {len(prompt)}')
        client = self._get_client()
        semaphore = self._get_semaphore()
        
        async with semaphore:
            for attempt in range(max_retries):
                try:
                    print(f'[DEBUG] Calling judge API, attempt {attempt + 1}/{max_retries}')
                    response = await client.chat.completions.create(
                        model=self.judge_openai_model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that evaluates responses."},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.0,
                        max_tokens=500,
                    )

                    content = response.choices[0].message.content
                    logger.debug(f"Judge response: {content[:200]}...")
                    print(f'[DEBUG] Judge API response received, content length: {len(content)}')
                    print(f'[DEBUG] Judge response (first 500 chars): {content[:500]}')

                    return {"success": True, "content": content}

                except asyncio.TimeoutError:
                    logger.warning(f"Judge API timeout on attempt {attempt + 1}/{max_retries}")
                    print(f'[DEBUG] Judge API timeout on attempt {attempt + 1}/{max_retries}')
                    if attempt == max_retries - 1:
                        return {"success": False, "error": "timeout", "content": ""}
                    await asyncio.sleep(2 ** attempt)

                except Exception as e:
                    logger.warning(f"Judge API error on attempt {attempt + 1}/{max_retries}: {e}")
                    print(f'[DEBUG] Judge API error on attempt {attempt + 1}/{max_retries}: {e}')
                    if attempt == max_retries - 1:
                        return {"success": False, "error": str(e), "content": ""}
                    await asyncio.sleep(2 ** attempt)

            print(f'[DEBUG] Judge API max retries exceeded')
            return {"success": False, "error": "max_retries_exceeded", "content": ""}

    def _parse_judge_response(self, content: str) -> dict:
        result = {
            "extracted_final_answer": None,
            "reasoning": None,
            "correct": None,
            "confidence": None,
            "parse_error": False
        }

        if not content:
            result["parse_error"] = True
            return result

        try:
            answer_match = re.search(r"\*\*extracted_final_answer:\*\*\s*(.*?)(?=\n|$)", content, re.IGNORECASE | re.DOTALL)
            if not answer_match:
                answer_match = re.search(r"\*\*extracted_final_answer\*\*:\s*(.*?)(?=\n|$)", content, re.IGNORECASE | re.DOTALL)
            if not answer_match:
                answer_match = re.search(r"extracted_final_answer:\s*(.*?)(?=\n|$)", content, re.IGNORECASE | re.DOTALL)
            if answer_match:
                result["extracted_final_answer"] = answer_match.group(1).strip()

            reasoning_match = re.search(r"\*\*reasoning:\*\*\s*(.*?)(?=\n\*\*correct:\*\*|\n\*\*correct\*\*:|\ncorrect:|$)", content, re.IGNORECASE | re.DOTALL)
            if not reasoning_match:
                reasoning_match = re.search(r"\*\*reasoning\*\*:\s*(.*?)(?=\n\*\*correct:\*\*|\n\*\*correct\*\*:|\ncorrect:|$)", content, re.IGNORECASE | re.DOTALL)
            if not reasoning_match:
                reasoning_match = re.search(r"reasoning:\s*(.*?)(?=\ncorrect:|$)", content, re.IGNORECASE | re.DOTALL)
            if reasoning_match:
                result["reasoning"] = reasoning_match.group(1).strip()

            correct_match = re.search(r"\*\*correct:\*\*\s*(yes|no)", content, re.IGNORECASE)
            if not correct_match:
                correct_match = re.search(r"\*\*correct\*\*:\s*(yes|no)", content, re.IGNORECASE)
            if not correct_match:
                correct_match = re.search(r"correct:\s*(yes|no)", content, re.IGNORECASE)
            if correct_match:
                result["correct"] = correct_match.group(1).lower()

            confidence_match = re.search(r"confidence:\s*(\d+(?:\.\d+)?)\s*%?", content, re.IGNORECASE)
            if confidence_match:
                result["confidence"] = float(confidence_match.group(1))
            else:
                result["confidence"] = 100.0

        except Exception as e:
            logger.warning(f"Error parsing judge response: {e}")
            result["parse_error"] = True

        return result

    async def _process_single_item(self, data_item: dict, gen_uid: str) -> dict:
        prompt = data_item.get("prompt", "")
        response = data_item.get("response", "")
        ground_truth = data_item.get("ground_truth", "")

        print(f'[DEBUG] _process_single_item called for gen_uid: {gen_uid}')
        print(f'[DEBUG] Prompt length: {len(prompt)}, Response length: {len(response)}, Ground truth length: {len(ground_truth)}')
        print(f'[DEBUG] Prompt (first 200 chars): {prompt[:200]}')
        print(f'[DEBUG] Response (first 200 chars): {response[:200]}')
        print(f'[DEBUG] Ground truth (first 200 chars): {ground_truth[:200] if ground_truth else "EMPTY"}')

        judge_prompt = GRADER_TEMPLATE.format(
            question=prompt,
            response=response,
            correct_answer=ground_truth
        )

        print(f'[DEBUG] Judge prompt length: {len(judge_prompt)}')
        result = await self._call_judge(judge_prompt, max_retries=self.max_retries)

        print(f'[DEBUG] Judge API result success: {result["success"]}, error: {result.get("error")}')

        if result["success"]:
            parsed = self._parse_judge_response(result["content"])
            print(f'[DEBUG] Parsed result: correct={parsed["correct"]}, parse_error={parsed["parse_error"]}')
            print(f'[DEBUG] Extracted final answer: {parsed.get("extracted_final_answer")}')
            print(f'[DEBUG] Judge reasoning: {parsed.get("reasoning")}')
            print(f'[DEBUG] Judge confidence: {parsed.get("confidence")}')
            if parsed["parse_error"]:
                score = 0.0
                print(f'[DEBUG] Parse error, score set to 0.0')
            else:
                score = 1.0 if parsed["correct"] == "yes" else 0.0
                print(f'[DEBUG] Score calculated: {score} (correct={parsed["correct"]})')
        else:
            score = 0.0
            parsed = {
                "extracted_final_answer": None,
                "reasoning": f"Error: {result['error']}",
                "correct": "no",
                "confidence": 0.0,
                "parse_error": True,
            }
            print(f'[DEBUG] Judge API failed, score set to 0.0, error: {result["error"]}')

        print(f'[DEBUG] Returning result for gen_uid {gen_uid}: score={score}, correct={parsed.get("correct")}, judge_error={result.get("error")}')
        return {
            "gen_uid": gen_uid,
            "score": score,
            "extracted_final_answer": parsed.get("extracted_final_answer"),
            "correct": parsed.get("correct", "no"),
            "reasoning": parsed.get("reasoning"),
            "confidence": parsed.get("confidence", 0.0),
            "judge_error": result.get("error"),
        }

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        print('[DEBUG] JudgeRewardManager.__call__ called')
        prompts = data.batch.get("prompts", None)
        responses = data.batch.get("responses", None)
        ground_truths = data.batch.get("ground_truths", None)
        gen_uids = data.batch.get("gen_uids", None)

        if prompts is None or responses is None:
            logger.error("Missing required fields in data batch")
            return torch.zeros(len(data), dtype=torch.float32)

        batch_size = len(data)
        logger.info(f"Processing batch of {batch_size} items")
        print(f'[DEBUG] Batch size: {batch_size}, return_dict: {return_dict}')

        items = []
        for i in range(batch_size):
            gen_uid = gen_uids[i] if gen_uids is not None else str(uuid4())
            
            prompt_text = prompts[i] if prompts is not None else ""
            response_text = responses[i] if responses is not None else ""
            ground_truth_text = ground_truths[i] if ground_truths is not None else ""
            
            if isinstance(prompt_text, torch.Tensor):
                prompt_text = self.tokenizer.decode(prompt_text, skip_special_tokens=True)
            if isinstance(response_text, torch.Tensor):
                response_text = self.tokenizer.decode(response_text, skip_special_tokens=True)
            if isinstance(ground_truth_text, torch.Tensor):
                ground_truth_text = self.tokenizer.decode(ground_truth_text, skip_special_tokens=True)
            
            items.append({
                "prompt": prompt_text,
                "response": response_text,
                "ground_truth": ground_truth_text,
                "gen_uid": gen_uid,
            })

        print(f'[DEBUG] Created {len(items)} items for processing')

        async def process_batch(items):
            tasks = [self._process_single_item(item, item["gen_uid"]) for item in items]
            results = await asyncio.gather(*tasks)
            return results

        print('[DEBUG] Creating new event loop for thread pool executor...')
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            print('[DEBUG] Running async process_batch...')
            results = loop.run_until_complete(process_batch(items))
            print(f'[DEBUG] process_batch completed, got {len(results)} results')
        finally:
            loop.close()
            print('[DEBUG] Event loop closed')

        scores = torch.tensor([r["score"] for r in results], dtype=torch.float32)

        unique_gen_uids = set(r["gen_uid"] for r in results)
        avg_score = scores.mean().item()
        std_score = scores.std().item() if len(scores) > 1 else 0.0
        min_score = scores.min().item()
        max_score = scores.max().item()

        correct_count = sum(1 for r in results if r["correct"] == "yes")
        incorrect_count = sum(1 for r in results if r["correct"] == "no")
        error_count = sum(1 for r in results if r["judge_error"] is not None)

        metrics = {
            "avg_score": avg_score,
            "std_score": std_score,
            "min_score": min_score,
            "max_score": max_score,
            "num_unique_gen_uids": len(unique_gen_uids),
            "correct_count": correct_count,
            "incorrect_count": incorrect_count,
            "error_count": error_count,
            "batch_size": batch_size,
        }

        logger.info(f"Batch metrics: {metrics}")

        if return_dict:
            return {
                "scores": scores,
                "results": results,
                "metrics": metrics,
            }
        else:
            Tracking.log_metrics(metrics)
            return scores

    def verify(self, data: DataProto) -> torch.Tensor:
        return self.__call__(data, return_dict=False)
