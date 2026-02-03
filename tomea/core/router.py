from typing import Type
from tomea.harnesses.base import BaseHarness
from tomea.harnesses.sft import SFTHarness
from tomea.harnesses.rl import RLHarness

class ResearchRouter:
    
    # Keyword sets separated by category for maintainability.
    # Adding a new domain = adding a new list here.
    
    RL_KEYWORDS = [
        # Standard RL terminology
        "reinforcement learning", "ppo", "dpo", "policy optimization", "reward",
        "grpo", "ipo", "kto",
        # Financial RL (QuantEval, trading papers)
        "sharpe ratio", "reward function", "reward optimization", "trading strategy",
        "portfolio optimization", "profit and loss", "p&l", "alpha generation",
        "reward signal", "value function", "q-learning", "actor-critic",
        # Preference learning (broader DPO family)
        "preference learning", "preference optimization", "rlhf",
        "reward model", "human feedback",
        # Mirror / advanced DPO variants
        "mirror descent", "gdpo",
    ]
    
    SFT_KEYWORDS = [
        # If we ever need to force SFT routing on ambiguous papers
        "supervised fine-tuning", "instruction tuning", "sft",
    ]

    @staticmethod
    def get_harness(mode: str = "auto", abstract: str = "") -> Type[BaseHarness]:
        # 1. Manual Override — always wins
        if mode == "sft": return SFTHarness
        if mode == "rl": return RLHarness
        
        # 2. Auto-Detection
        abstract_lower = abstract.lower()
        
        # Check RL first — it's the more specific signal.
        # If any RL keyword matches, route to RL.
        if any(kw in abstract_lower for kw in ResearchRouter.RL_KEYWORDS):
            return RLHarness
        
        # Check explicit SFT keywords (catches "supervised fine-tuning" papers
        # that might also mention "reward" in passing)
        if any(kw in abstract_lower for kw in ResearchRouter.SFT_KEYWORDS):
            return SFTHarness
            
        # Default: SFT. This is the safer fallback — SFT template is more
        # generic and less likely to produce broken code on unknown papers.
        return SFTHarness