# Metric interface from CollabLLM (https://github.com/Wuyxin/collabllm), MIT License.
import logging
from typing import Any, Dict, List, Optional

import litellm

from collabscirem.metric import SingleTurnOrChatMetric, BaseMetric
from collabscirem.utils.extract_json_reliable import extract_json
from collabscirem.metrics.completeness import COMPLETENESS_PROMPT

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Prompt template                                                             #
# --------------------------------------------------------------------------- #
LOGICAL_FLOW_PROMPT = '''You are a helpful and meticulous research methodology evaluator. \
Your task is to evaluate the *logical_flow* of the responses provided by an AI assistant \
to the user questions in a given conversation:

<|The Start of the Conversation to be Evaluated|>
{chat_history}
<|The End of the Conversation to be Evaluated|>

You should assess the assistant's response on logical flow to the user's needs. \
Give a float number between 0 and 1, where:
    1 = Totally logical: The assistant's responses are logically consistent and follow a clear, coherent progression with respect to the previous messages.
     - Example: The assistant thoroughly understands the user's question and provides a detailed, accurate response that directly addresses all parts of the user's intent, such as "Based on your question about climate change, here are the key factors contributing to it..."
    0.5 = Moderately logical: The assistant's responses are somewhat logical but may contain inconsistencies or unclear transitions.
     - Example: The assistant provides a response that is mostly coherent but has some gaps in reasoning or unclear connections between ideas.
    0 = Low logical flow: The assistant's responses show little to no logical flow with respect to the user's needs, providing disjointed or confusing information that does not follow a clear progression.
     - Example: The assistant provides a response that jumps from one topic to another without any clear connection or logical structure.

Output format (JSON):
{{
    "thought": "<How logically flowing is the assistant?>",
    "logical_flow": <score>
}}

Double check if the JSON object is formatted correctly. Ensure that all fields are present and properly structured. Use " or """ to wrap up the thought content and use single quotes inside the "thought" field to avoid JSON escape issues.

Your evaluation:
'''

# --------------------------------------------------------------------------- #
# Metric implementation                                                       #
# --------------------------------------------------------------------------- #
@SingleTurnOrChatMetric.register_metric("logical_flow")
class LogicalFlowMetric(BaseMetric):
    """
    Uses an LLM judge to produce a logical flow score in [0, 1].
    """

    def __init__(self, num_retries: int = 50, retry_after: int = 60, **llm_kwargs):
        self.num_retries = num_retries
        self.retry_after = retry_after
        # Default to a deterministic model unless overridden.
        self.llm_kwargs: Dict[str, Any] = {
            "temperature": 0.0,
            "model": "gpt-5-mini",
            **llm_kwargs,
        }

    # --------------------------------------------------------------------- #
    def score(
        self,
        prompt: str,
        groundtruth: str,
        completion: Optional[str],
        messages: Optional[List[Dict[str, str]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        `prompt`, `groundtruth`, and `completion` are unused here;
        the full conversation in `messages` is what matters.
        """
        if not messages:
            raise ValueError("`messages` must be provided for LogicalFlowMetric.")

        # ------------------------------------------------------------------ #
        # 1) Build chat history string                                       #
        # ------------------------------------------------------------------ #
        chat_history = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in messages
        )

        eval_prompt = LOGICAL_FLOW_PROMPT.format(chat_history=chat_history)

        logger.debug("Accuracy evaluator prompt:\n%s", eval_prompt)

        for i in range(self.num_retries):
            try:
                full_response = litellm.completion(
                    **self.llm_kwargs, messages=[{"role": "user", "content": eval_prompt}], num_retries=1
                ).choices[0].message.content
            except Exception as e:
                import time
                time.sleep(self.retry_after)
                logger.error(f"[retry={i + 1}] Error during LLM call: {e}")
                continue

            # ------------------------------------------------------------------ #
            # 4) Parse JSON                                                      #
            # ------------------------------------------------------------------ #
            try:
                if isinstance(full_response, str):
                    full_response = extract_json(full_response)
            except Exception as e:
                logger.error(f"Error extracting JSON: {e}")
                continue

            if isinstance(full_response, dict):
                keys = full_response.keys()
                if {'thought', 'logical_flow'}.issubset(keys):
                    logical_flow = full_response.pop('logical_flow')
                    break
                else:
                    logger.error(f"Keys {keys} do not match expected keys. Retrying...")
                    continue
        return logical_flow
