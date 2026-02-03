import pandas as pd
import logging
from datasets import load_dataset, Dataset
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class DataEngine:
    """
    The Universal Data Adapter.
    Analyses data schemas and provides context to the Synthesizer.
    """
    
    def __init__(self):
        # Fallbacks to prevent "Mock Data" if user provides nothing
        self.defaults = {
            "sft": "tatsu-lab/alpaca",
            "dpo": "Anthropic/hh-rlhf",
            "ppo": "Anthropic/hh-rlhf", 
            "grpo": "Anthropic/hh-rlhf",
            "rl": "Anthropic/hh-rlhf"  # Generic RL fallback
        }

    def load_and_inspect(self, source: Optional[str] = None, mode: str = "sft") -> Dict[str, Any]:
        """
        Loads the dataset and returns a schema summary for the Synthesizer.
        """
        dataset = None
        source_name = source
        
        try:
            # 1. Load Data (User provided or Default)
            if not source:
                source_name = self.defaults.get(mode.lower(), "tatsu-lab/alpaca")
                logger.info(f"   No data provided. Using Standard Benchmark: {source_name}")
                # Load tiny slice just for introspection
                dataset = load_dataset(source_name, split="train[:10]") 
            elif source.endswith(".csv"):
                dataset = Dataset.from_pandas(pd.read_csv(source))
            elif source.endswith(".json") or source.endswith(".jsonl"):
                dataset = load_dataset("json", data_files=source, split="train[:10]")
            else:
                # Assume HF ID
                dataset = load_dataset(source, split="train[:10]")

        except Exception as e:
            logger.error(f"Data Load Error: {e}")
            return {"error": str(e)}

        # 2. Inspect Schema
        if len(dataset) == 0:
             return {"error": "Dataset is empty."}
             
        sample = dataset[0]
        columns = list(sample.keys())
        
        # 3. Construct Schema Context
        # We truncate the sample to avoid blowing up LLM context
        schema_info = {
            "source_name": source_name,
            "columns": columns,
            "sample_row": str(sample)[:500], 
            "recommended_map": self._recommend_mapping(columns, mode)
        }
        
        return schema_info

    def _recommend_mapping(self, columns: list, mode: str) -> str:
        """
        Heuristic logic to guide the LLM on how to map columns.
        Covers: standard LLM formats, preference pairs, financial/time-series data.
        """
        mode = mode.lower()
        cols = [c.lower() for c in columns]
        
        mapping_hint = ""
        
        # ===================================================================
        # PREFERENCE PAIR FORMATS (DPO / RL with preference data)
        # ===================================================================
        if "dpo" in mode or "rl" in mode:
            if "chosen" in cols and "rejected" in cols:
                mapping_hint = "Columns 'chosen' and 'rejected' detected. Map these directly to DPO format."
            elif "response_j" in cols and "response_k" in cols:
                mapping_hint = "Map 'response_j' to chosen, 'response_k' to rejected."
            
            # ===============================================================
            # FINANCIAL / TIME-SERIES FORMATS (RL with reward functions)
            # These need special handling: no chosen/rejected pairs, instead
            # the reward function computes signal from the data directly.
            # ===============================================================
            
            # OHLCV price data (QuantEval, trading strategies)
            ohlcv_cols = {"open", "high", "low", "close", "volume"}
            if ohlcv_cols.issubset(set(cols)):
                mapping_hint = (
                    "OHLCV time-series data detected. This is NOT preference-pair data. "
                    "Do NOT use chosen/rejected format. Instead: "
                    "Map 'close' prices as the primary signal for reward computation. "
                    "The reward_function should compute returns (e.g., Sharpe Ratio or P&L) "
                    "from price sequences. Use 'open'/'close' for entry/exit calculations. "
                    "Format as sequential windows for the model input. "
                    "Requires: pandas for rolling calculations, numpy for array ops."
                )
            
            # Financial sentiment / news impact (FinGPT style)
            elif any(kw in cols for kw in ["news", "sentiment", "impact", "market_news"]):
                # Determine if we have a signal column for reward
                has_reward_signal = any(kw in cols for kw in ["impact", "return", "price_change", "label", "score"])
                
                if has_reward_signal:
                    # Can construct reward from the signal column
                    signal_col = next(kw for kw in ["impact", "return", "price_change", "label", "score"] if kw in cols)
                    text_col = next(kw for kw in ["news", "market_news", "text", "headline"] if kw in cols)
                    mapping_hint = (
                        f"Financial sentiment data detected. Map '{text_col}' as the prompt input. "
                        f"'{signal_col}' is a numeric reward signal — use it directly in reward_function "
                        f"to score model completions (higher {signal_col} = better reward). "
                        f"Do NOT use chosen/rejected format. This is a reward-based RL setup. "
                        f"If 'ticker' column exists, include it as context in the prompt."
                    )
                else:
                    # News without a reward signal — treat as SFT-style text generation
                    text_col = next(kw for kw in ["news", "market_news", "text", "headline"] if kw in cols)
                    mapping_hint = (
                        f"Financial text data detected but no reward signal column found. "
                        f"Treating as text generation task. Map '{text_col}' as input. "
                        f"If ticker/date columns exist, prepend them as context."
                    )
            
            # Generic ticker/stock data without clear signal
            elif "ticker" in cols or "symbol" in cols:
                mapping_hint = (
                    "Stock/ticker data detected. Check if numeric columns represent price or return signals. "
                    "If so, use them for reward computation in RL. If not, treat as structured text input."
                )
        
        # ===================================================================
        # SFT FORMATS
        # ===================================================================
        elif "sft" in mode:
            if "instruction" in cols and "output" in cols:
                mapping_hint = "Alpaca format detected. Map `instruction` + `input` to prompt, `output` to response."
            elif "text" in cols:
                mapping_hint = "Standard text column detected. Use 'text' directly."
            # Financial text for SFT (e.g., news summarization)
            elif any(kw in cols for kw in ["news", "market_news", "headline"]):
                text_col = next(kw for kw in ["news", "market_news", "headline"] if kw in cols)
                mapping_hint = (
                    f"Financial text detected in SFT mode. Map '{text_col}' as input. "
                    f"If a summary or output column exists, use it as the target. "
                    f"Otherwise, this may need manual target definition."
                )
        
        return mapping_hint