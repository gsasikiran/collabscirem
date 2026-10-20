# Adapted from CollabLLM (https://github.com/Wuyxin/collabllm), MIT License;
# extended here for the science-aware reward decomposition.
# Reward metrics of CollabSciReM (paper Table 3; prompts in Appendix F.3).
# Importing a module registers its metric under the name used in --metric_names.
from .accuracy import *       # R_gen
from .completeness import *
from .clarity import *
from .interactivity import *
from .justification import *  # R_sci
from .logical_flow import *
from .novelty import *
from .format import *         # R_struct
from .efficiency import *     # token_amount
