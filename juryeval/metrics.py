import warnings

warnings.warn(
    "juryeval.metrics is deprecated. Use juryeval.metrics.fluency directly.",
    DeprecationWarning,
    stacklevel=2,
)

from juryeval.metrics.fluency import perplexity, flesch_kincaid  # noqa: F401, E402
