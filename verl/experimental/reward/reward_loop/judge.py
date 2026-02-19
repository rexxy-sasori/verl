import logging
import os
from typing import Any

from omegaconf import DictConfig
from transformers import AutoTokenizer

from verl import DataProto
from verl.experimental.reward.reward_loop import register
from verl.experimental.reward.reward_loop.base import RewardLoopManagerBase
from verl.workers.reward_manager import get_reward_manager_cls

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("judge")
class JudgeRewardLoopManager(RewardLoopManagerBase):
    """Reward loop manager that uses JudgeRewardManager for remote judge-based reward computation."""

    def __init__(
        self,
        config: DictConfig,
        tokenizer: AutoTokenizer,
        compute_score=None,
        reward_router_address=None,
        reward_model_tokenizer=None,
    ):
        print('[DEBUG] JudgeRewardLoopManager.__init__ called')
        super().__init__(config, tokenizer)
        
        print('[DEBUG] Getting JudgeRewardManager from reward manager registry...')
        reward_manager_cls = get_reward_manager_cls("judge")
        print(f'[DEBUG] Got reward manager class: {reward_manager_cls}')
        
        self.reward_manager = reward_manager_cls(
            tokenizer=tokenizer,
            num_examine=config.reward_model.get("num_examine", 1),
            compute_score=compute_score,
            reward_fn_key=config.reward_model.get("reward_fn_key", "data_source"),
            **config.reward_model,
        )

        logger.info("Initialized JudgeRewardLoopManager")

    async def run_single(self, data: DataProto) -> dict:
        print('[DEBUG] JudgeRewardLoopManager.run_single called')
        assert len(data) == 1, "Only support single data item"
        
        print('[DEBUG] Calling JudgeRewardManager in thread pool executor...')
        result = await self.loop.run_in_executor(
            None,
            lambda: self.reward_manager(data, return_dict=True),
        )
        print(f'[DEBUG] Got result from JudgeRewardManager: {type(result)}')
        
        if isinstance(result, dict):
            scores = result.get("scores")
            results = result.get("results", [])
            metrics = result.get("metrics", {})
            
            print(f'[DEBUG] Scores: {scores}, Results count: {len(results)}, Metrics: {list(metrics.keys())}')
            
            if scores is not None and len(scores) > 0:
                reward_score = scores[0].item()
                reward_extra_info = {}
                
                if len(results) > 0:
                    item_result = results[0]
                    reward_extra_info["extracted_final_answer"] = item_result.get("extracted_final_answer")
                    reward_extra_info["correct"] = item_result.get("correct")
                    reward_extra_info["reasoning"] = item_result.get("reasoning")
                    reward_extra_info["confidence"] = item_result.get("confidence")
                    reward_extra_info["judge_error"] = item_result.get("judge_error")
                
                reward_extra_info.update(metrics)
                
                print(f'[DEBUG] Final reward_score: {reward_score}, extra_info keys: {list(reward_extra_info.keys())}')
                return {"reward_score": reward_score, "reward_extra_info": reward_extra_info}
        
        print('[DEBUG] No valid result, returning default score 0.0')
        return {"reward_score": 0.0, "reward_extra_info": {}}
