"""Which foundation models can use exogenous covariates zero-shot, and how (Notebook 05a and the README)."""
# --- cell: capabilities ---
import pandas as pd

# One row per model run in this chapter. "past" = past-only covariates (observed up to the
# origin), "future" = known-future covariates (also known over the horizon), "mv" = several
# target series forecast jointly. Every flag was checked against the paper or model card AND
# the library version pinned in requirements/ (the adapter flags in ch03_zeroshot.models
# must agree; tests/test_architectures_and_capabilities.py checks that).
ZERO_SHOT_MODELS = [
    # key, model, organisation, backbone, mv, past, future, mechanism, source
    ("chronos2", "Chronos-2", "Amazon", "encoder-only transformer", True, True, True,
     "group attention across target and covariate tokens", "Ansari et al. (2025), arXiv:2510.15821"),
    ("chronos-bolt", "Chronos-Bolt (small)", "Amazon", "T5 encoder-decoder", False, False, False,
     "univariate only", "Ansari et al. (2024), arXiv:2403.07815; model card"),
    ("timesfm25", "TimesFM-2.5", "Google", "decoder-only transformer", False, False, True,
     "XReg: in-context linear regression on the covariates, fitted per series", "Das et al. (2024), arXiv:2310.10688; timesfm 2.x README"),
    ("timesfm3", "TimesFM-3", "Google", "decoder-only transformer with variate attention", True, True, True,
     "variate attention over target and covariate channels", "model card google/timesfm-3.0-pytorch (no paper yet)"),
    ("tirex", "TiRex", "NX-AI", "xLSTM (sLSTM) decoder", False, False, False,
     "univariate only", "Auer et al. (2025), arXiv:2505.23719"),
    ("tirex2", "TiRex-2", "NX-AI", "xLSTM time mixer + variate mixer", True, True, True,
     "asymmetric group attention; reverse xLSTM pass for known-future inputs", "Podest, Pichler et al. (2026), arXiv:2607.01204"),
    ("timestral", "Timestral-209M", "Mistral AI", "causal transformer, decoupled horizon tokens", True, True, True,
     "variate attention every 4th block; covariates cannot read targets", "Mistral AI (2026), arXiv:2610.37585"),
    ("timestral-5m", "Timestral-5M", "Mistral AI", "causal transformer, decoupled horizon tokens", True, True, True,
     "as Timestral-209M", "Mistral AI (2026), arXiv:2610.37585"),
    ("ttm", "TTM-r2", "IBM", "TSMixer (MLP mixer)", False, False, False,
     "exogenous mixer exists but only after fine-tuning; zero-shot is univariate", "Ekambaram et al. (2024), arXiv:2401.03955"),
    ("flowstate", "FlowState-r1.1", "IBM", "S5 state-space encoder + functional basis decoder", False, False, False,
     "univariate only", "Graf et al. (2025), arXiv:2508.05287"),
    ("patchtst-fm", "PatchTST-FM-r2", "IBM", "Conformer-style patch transformer", False, False, False,
     "univariate only", "model card ibm-granite/granite-timeseries-patchtst-fm-r2"),
    ("t0", "t0-gamma", "The Forecasting Company", "decoder-style patch transformer, time + variate attention", True, True, True,
     "alternating time and variate attention; known-future rows read bidirectionally", "Meyer et al. (2026), arXiv:2609.24559"),
    ("tabpfn-ts", "TabPFN-TS (v2 weights)", "Prior Labs", "tabular prior-fitted network", False, False, True,
     "known-future covariates become extra feature columns of the regression table", "Hoo et al. (2025), arXiv:2501.02945"),
    ("moirai11", "Moirai-1.1 (small)", "Salesforce", "masked encoder, any-variate attention", True, True, True,
     "all variates flattened into one sequence; binary variate attention bias", "Woo et al. (2024), arXiv:2402.02592"),
    ("moirai-moe", "Moirai-MoE (small)", "Salesforce", "decoder-only sparse MoE", True, True, True,
     "same packing and attention bias as Moirai-1.1", "Liu, Liu et al. (2024), arXiv:2410.10469"),
    ("moirai2", "Moirai-2.0 (small)", "Salesforce", "decoder-only transformer", False, False, False,
     "univariate: the paper dropped multivariate and covariate support", "Liu, Aksu et al. (2025), arXiv:2511.11698, Sections 3.1 and 6"),
    ("toto2", "Toto-2.0 (22m)", "Datadog", "decoder-only, time + variate attention", True, False, False,
     "joint targets via variate attention; exogenous inputs not yet supported", "Khwaja et al. (2026), arXiv:2605.20119; toto-2 README"),
    ("toto1", "Toto-1.0", "Datadog", "decoder-only, proportional factorised attention", True, False, True,
     "known-future covariates passed as extra variates (exogenous API); their future values reach only steps 65 onward",
     "Cohen et al. (2025), arXiv:2505.14766; toto-ts 0.2.0"),
    ("sundial", "Sundial (base)", "Tsinghua THUML", "decoder-only transformer + flow-matching head", False, False, False,
     "univariate only", "Liu, Qin et al. (2025), arXiv:2502.00816"),
    ("timer", "Timer (base)", "Tsinghua THUML", "decoder-only transformer", False, False, False,
     "univariate checkpoint (Timer-XL adds covariates only when trained for them)", "Liu et al. (2024), arXiv:2402.02368; model card"),
    ("timemoe", "Time-MoE (50M)", "Princeton, Squirrel Ai, Griffith et al.", "decoder-only sparse MoE, point tokens", False, False, False,
     "univariate only", "Shi et al. (2025), arXiv:2409.16040"),
    ("kairos", "Kairos (50M)", "ShanghaiTech and Ant Group", "T5-style encoder-decoder with mixture-of-size patching", False, False, False,
     "univariate only", "Feng et al. (2025), arXiv:2509.25826"),
    ("lag-llama", "Lag-Llama", "Morgan Stanley, ServiceNow, Mila et al.", "LLaMA-style decoder on lag features", False, False, False,
     "univariate only (calendar features are built from timestamps, not passed in)", "Rasul et al. (2023), arXiv:2310.08278"),
]

COLUMNS = ["key", "model", "organisation", "backbone", "multivariate", "past_covariates",
           "future_covariates", "how", "source"]

# Considered and left out of the runs (the reasons are in the README).
EXCLUDED = [
    ("MOMENT", "no zero-shot forecasting head: the forecasting head must be trained first"),
    ("YingLong", "the released inference path needs a CUDA GPU (FlashAttention)"),
    ("Timer-S1", "GPU-only inference in the released code"),
    ("TempoPFN", "GPU-only inference in the released code"),
    ("TabPFN-TS default weights (TabPFN-2.5)", "gated: needs a Prior Labs account and token; the ungated v2 weights are used"),
]


def capability_table():
    """The classification table with a plain-language class for each model."""
    df = pd.DataFrame(ZERO_SHOT_MODELS, columns=COLUMNS)

    def klass(r):
        if r.past_covariates and r.future_covariates:
            return "past + known-future covariates"
        if r.future_covariates:
            return "known-future covariates only"
        if r.past_covariates:
            return "past covariates only"
        return "none (target only)"

    df["exogenous_class"] = df.apply(klass, axis=1)
    return df
