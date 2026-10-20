# Adapted from CollabLLM (https://github.com/Wuyxin/collabllm), MIT License;
# extended here for the science-aware reward decomposition.
from .summarized_papers import SummarizeScientificPapers


datasets_info = {
    'scientific-papers-summarization': {
        'task_desc': 'scientific workflow generation',
        'class': SummarizeScientificPapers
    },
}
