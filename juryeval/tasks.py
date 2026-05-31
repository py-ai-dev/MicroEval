import warnings

warnings.warn(
    "juryeval.tasks is deprecated. Use juryeval.metrics instead.",
    DeprecationWarning,
    stacklevel=2,
)

from juryeval.metrics.classification import eval_classification  # noqa: F401, E402
from juryeval.metrics.translation import eval_translation  # noqa: F401, E402
from juryeval.metrics.summarization import eval_summarization  # noqa: F401, E402
