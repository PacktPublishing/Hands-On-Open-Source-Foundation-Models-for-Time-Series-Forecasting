"""Architecture figures, short overviews and compute notes for every model in Notebooks 05 to 10.

Each entry was checked three ways (see ``FACT_CHECK`` and tests/test_architectures.py):
1. against the paper (section or equation named in ``facts``);
2. against the checkpoint's config.json (the numbers in ``config``, snapshotted in
   data/model_configs/);
3. against the module tree of the loaded network (``print(model)``, layer counts and widths).
"""
# --- cell: architectures ---
import ch03_zeroshot.diagrams as _dg
from ch03_zeroshot.diagrams import Block, Box, Column, Fan, Link, Note, Res, Spec, draw, draw_h

ARCH = {}

# The expressions written under the layers of the figures, with a plain explanation of each (shown in the notebooks
# under each figure). A layer gets one when its label names the operation; each expression is written once per
# figure. Each entry is (expression, sentence that introduces it, sentence that follows it).
MATH = {
    r"asinh": (r"$\tilde{x}_t=\mathrm{arcsinh}\left(\frac{x_t-\mu}{\sigma}\right)$",
              r"The input is first **standardised** with the mean $\mu$ and standard deviation $\sigma$ of the "
              r"context (in the causal versions, of the values seen so far), and then compressed with the **inverse "
              r"hyperbolic sine** (arcsinh):",
              "The arcsinh is close to a straight line near zero and grows like a logarithm for large values, so a "
              "few spikes cannot dominate the input."),
    r"std": (r"$\tilde{x}_t=\frac{x_t-\mu}{\sigma}$",
            r"**Instance normalisation** puts every series on the same scale, using the mean $\mu$ and standard "
            "deviation $\\sigma$ of its own context (in the causal versions, of the values seen so far):",
            r"The forecast is mapped back to the original units with $x=\mu+\sigma\tilde{x}$."),
    r"patch": (r"$p_i=(\tilde{x}_{(i-1)P+1},\dots,\tilde{x}_{iP})$",
              r"**Patching** cuts the series into windows of $P$ steps, and each window becomes one token:",
              r"A context of $T$ steps then gives $T/P$ tokens, so attention costs $(T/P)^2$ instead of $T^2$."),
    r"embed": (r"$h_i=\mathrm{MLP}(p_i)$",
              r"The **patch embedding** turns each patch (with its missing-value mask, where the model has one) into "
              r"a vector of width $d$, the token the transformer works on:",
              "The map is a small fully connected network, a multilayer perceptron (MLP)."),
    r"linemb": (r"$h_i=Wp_i+b$",
               r"The **linear embedding** maps each input token with one learned matrix $W$ to a vector of width "
               r"$d$:",
               "In most models the input token is a patch; in FlowState it is a single time step with its "
               "missing-value flag."),
    r"pointemb": (r"$h_t=\mathrm{SwiGLU}(x_t)$",
                 "The **point-wise embedding** makes every single value its own token, through a small gated "
                 "network:",
                 "There are no patches, so the sequence of tokens is as long as the context."),
    r"maxpool": (r"$\tilde{y}=\mathrm{MaxPool}_{k_\ell}(y_{t-L+1:t})$",
                r"In **multi-rate sampling**, each stack sees the input after max-pooling with its own kernel size "
                r"$k_\ell$:",
                "A large kernel keeps only the slow movements; a small one keeps the detail."),
    r"interp": (r"$\hat{y}_{\tau}=\theta_a+\frac{\tau-\tau_a}{\tau_b-\tau_a}(\theta_b-\theta_a)$",
               r"With **hierarchical interpolation**, a block predicts only a few knots $\theta$ of the future, and "
               r"the steps $\tau$ between two knots $a$ and $b$ are filled in along a straight line:",
               "Long horizons therefore cost little."),
    r"causal": (r"$\mathrm{softmax}\left(\frac{QK^{\top}}{\sqrt{d_k}}+M\right)V$",
               r"**Masked self-attention** compares each token's query $Q$ with the keys $K$, turns the scores into "
               r"weights with a softmax, and uses the weights to mix the values $V$:",
               r"The mask $M$ is $-\infty$ wherever a token may not read another: a later time step in causal "
               "attention, or an excluded pair in variate attention (covariates that may not read targets, or rows "
               r"that belong to different series). Dividing by $\sqrt{d_k}$, the square root of the key width, "
               "keeps the scores in a range where the softmax does not saturate."),
    r"t5bias": (r"$\mathrm{softmax}\left(QK^{\top}+B\right)V$",
               r"**T5 attention** compares each query with every key and adds a learned bias $B$ that depends on the "
               "(bucketed) distance between the two tokens:",
               "Position therefore enters as a bias rather than inside the vectors. As in the T5 language model, "
               r"the scores are not divided by $\sqrt{d_k}$; that scale is folded into the initial weights. The "
               "decoder's self-attention and cross-attention use the same form."),
    r"t5rope": (r"$\mathrm{softmax}\left(\tilde{Q}\tilde{K}^{\top}\right)V$",
               r"**T5-style attention with rotary positions** (RoPE) rotates the queries and keys by an angle "
               r"proportional to their position before comparing them; the rotated versions are written "
               r"$\tilde{Q}$ and $\tilde{K}$:",
               "Each score then depends on the distance between the two tokens. As in T5, the scores are not "
               r"divided by $\sqrt{d_k}$. The group attention in the figure uses the same unscaled form, without "
               "the rotation."),
    r"drope": (r"$\mathrm{softmax}\left(\tilde{Q}\tilde{K}^{\top}\right)V$",
              r"**Attention with dynamic rotary positions** (DRoPE) rotates the queries and keys by an angle "
              r"proportional to their position before comparing them, written $\tilde{Q}$ and $\tilde{K}$; the "
              "rotation frequencies adapt to each series:",
              "Each score then depends on the distance between the two tokens. As in T5, the scores are not "
              r"divided by $\sqrt{d_k}$, and the decoder's attention layers use the same unscaled form."),
    r"pdsattn": (r"$\mathrm{softmax}\left(\frac{(s\odot Q)K^{\top}}{\sqrt{d_k}}+M\right)V$",
                r"TimesFM's **attention with a per-dimension query scale** multiplies every dimension of the "
                r"queries by a learned vector $s$ (PerDimScale) before the scores are computed:",
                "The queries and keys are also RMS-normalised (QK-norm) and rotated by RoPE. $M$ is $-\\infty$ "
                "above the diagonal, so a patch never reads a later one. TimesFM-3's variate attention uses the "
                "same form without the causal mask."),
    r"gatedattn": (r"$\sigma(hW_g)\odot\mathrm{softmax}\left(\frac{QK^{\top}}{\sqrt{d_k}}+M\right)V$",
                  "**Gated attention** is scaled attention whose output is multiplied, element by element, by a "
                  r"sigmoid gate computed from the layer input $h$:",
                  "Each head can thereby switch its own contribution off. The mask $M$ is causal in time attention "
                  "and follows the role rules in variate attention: known-future covariates read only each other, "
                  "and past-only covariates never read targets."),
    r"toto2attn": (r"$\mathrm{softmax}\left(\frac{(s\odot Q)K^{\top}}{d_k}+M\right)V$",
                  r"Toto-2.0's **attention** multiplies the queries by a learned per-dimension scale $s$ "
                  r"(PerDimScale) and divides the scores by $d_k$ rather than $\sqrt{d_k}$:",
                  r"This is the scaling of μP (maximal update parametrisation), which keeps the scores stable as the network "
                  r"gets wider. $M$ is "
                  "causal in time attention; variate attention has no mask."),
    r"anyvar": (r"$q_{i,m}^{\top}R_{i-j}\,k_{j,n}+u_{mn}$",
               r"**Any-variate attention** scores the query at time $i$ of variate $m$ against the key at time $j$ "
               r"of variate $n$ as",
               r"The rotary matrix $R_{i-j}$ (RoPE) encodes the distance in time, and the learned bias $u_{mn}$, which is "
               r"$u_1$ when $m=n$ and $u_2$ otherwise, says whether the two tokens belong to the same variate or not. "
               r"The variates are told apart "
               "without putting them in any order. A softmax over the keys turns the scores into weights (in "
               "Moirai-MoE, only over earlier time steps)."),
    r"attn": (r"$\mathrm{softmax}\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V$",
             r"**Scaled dot-product attention** compares each token's query $Q$ with the keys $K$ of the tokens it "
             r"may read, and a softmax turns the scores into weights that mix their values $V$:",
             "Every token is thus rebuilt from the tokens that matter for it."),
    r"rms": (r"$\frac{h}{\sqrt{\frac{1}{d}\sum_j h_j^2+\epsilon}}\odot g$",
            r"**RMSNorm** divides each token vector $h$ by its root mean square and multiplies the result by a "
            r"learned gain $g$:",
            r"Here $d$ is the width of the vector and $\epsilon$ a small constant that avoids dividing by zero. "
            "The normalisation keeps the size of the activations stable through many layers."),
    r"ln": (r"$\frac{h-\bar{h}}{s_h}\odot g+b$",
           r"**LayerNorm** centres each token vector $h$ on its mean $\bar{h}$, divides it by its standard "
           r"deviation $s_h$, and then rescales and shifts it with the learned $g$ and $b$:",
           ""),
    r"relu": (r"$W_2\,\max(0,\,W_1h)$",
             r"The **feed-forward network** widens each token with $W_1$, keeps the positive part (ReLU) and "
             r"projects it back with $W_2$:",
             "It works on every token separately, after attention has mixed them."),
    r"gelu": (r"$W_2\,\mathrm{GELU}(W_1h)$",
             "The **feed-forward network** applies the smooth GELU activation between two projections, to every "
             "token separately:",
             ""),
    r"silu": (r"$W_2\,\mathrm{SiLU}(W_1h)$",
             r"The **feed-forward network** applies the SiLU activation, $z\,\sigma(z)$, between two projections, to "
             "every token separately:",
             ""),
    r"swiglu": (r"$W_2\left(\mathrm{SiLU}(W_1h)\odot W_3h\right)$",
               "The **SwiGLU feed-forward network** passes one projection of the token through SiLU and uses it to "
               r"gate a second projection, element by element ($\odot$), before projecting back:",
               "It trains better than a plain ReLU network of the same size."),
    r"quantile": (r"$\rho_\tau(u)=\max(\tau u,\,(\tau-1)u)$",
                 r"The **quantile head** has one output per quantile level $\tau$. Training minimises the pinball "
                 r"loss $\rho_\tau$ of the error $u=y-\hat{q}_\tau$:",
                 r"Minimising this loss makes each output the $\tau$-quantile of the forecast distribution. The "
                 r"median ($\tau=0.5$) is the point forecast scored in this chapter."),
    r"moeshared": (r"$\sum_{k\in S(h)}g_k(h)\,E_k(h)+\sigma(w^{\top}h)\,E_{\mathrm{sh}}(h)$",
                  "The **mixture of experts with a shared expert** lets a router keep the top 2 of the 8 experts "
                  r"for each point, the set $S(h)$, and weight them with its gates $g_k$; a shared expert "
                  r"$E_{\mathrm{sh}}$ processes every point, weighted by a sigmoid gate:",
                  ""),
    r"gatedemb": (r"$h_i=\mathrm{SiLU}(W_gp_i)\odot W_ep_i$",
                 "The **gated patch embedding** multiplies two linear maps of the patch element by element, one of "
                 "them passed through SiLU:",
                 r"The Timer paper writes a plain linear map $W_ep_i$; the released checkpoint adds the gate."),
    r"robust": (r"$\tilde{x}_t=\frac{x_t-\mathrm{Med}(x)}{\mathrm{IQR}(x)}$",
               "**Robust standardisation** subtracts the median of the context and divides by its interquartile "
               "range:",
               "A few outliers then cannot distort the scale."),
    r"moe": (r"$\sum_{k\in S(h)}g_k(h)\,E_k(h)$",
            r"In a **mixture of experts**, a router scores the experts $E_k$ for each token, keeps the top few, the "
            r"set $S(h)$, and adds their outputs weighted by its gates $g_k$:",
            "Only those experts run, so the active model is much smaller than the total."),
    r"slstm": (r"$c_t=f_t\,c_{t-1}+i_t\,z_t,\;\;h_t=o_t\,c_t/n_t$",
              r"The **sLSTM cell** keeps a gated memory of the past in its cell state $c_t$, with a forget gate "
              r"$f_t$, an input gate $i_t$ and a candidate value $z_t$; $n_t$ normalises the state and the output "
              r"gate $o_t$ decides what is passed on:",
              "Unlike attention, the cost grows only linearly with the context."),
    r"ssm": (r"$s_k=\bar{A}s_{k-1}+\bar{B}u_k,\;\;y_k=Cs_k+Du_k$",
            r"A **state-space layer** updates a hidden state $s_k$ linearly from the previous state and the new "
            r"input $u_k$, and reads its output $y_k$ from the state:",
            r"The discretised matrices $\bar{A}$ and $\bar{B}$ depend on the step size, which FlowState scales "
            "with the sampling rate."),
    r"legendre": (r"$\hat{y}(t)=\sum_{n}c_n\,P_n(t)$",
                 "The **functional decoder** outputs a smooth function of time, a weighted sum of Legendre "
                 r"polynomials $P_n$ whose coefficients $c_n$ come from the network:",
                 "The forecast can therefore be read at any horizon."),
    r"mixture": (r"$p(y)=\sum_{c}w_c\,p_c(y)$",
                r"The **mixture distribution** head outputs weights $w_c$ and parameters for several distributions "
                r"$p_c$; their weighted sum is the forecast density:",
                "Sampling from it gives forecast paths."),
    r"flow": (r"$\frac{dx}{ds}=v_\theta(x,s\mid h)$",
             r"**Flow matching** trains a network to learn a velocity field $v_\theta$ that moves random noise $x$ "
             r"towards a forecast as $s$ runs from 0 to 1, given the summary $h$ of the context:",
             "Following the field from different noise draws gives different sampled futures."),
    r"lags": (r"$[x_{t-\ell_1},\dots,x_{t-\ell_K}]$",
             r"With **lag features**, each token holds the values at fixed lags $\ell_k$ before its time step:",
             "With lags such as one day, one week and one year, even a short sequence of tokens sees the "
             "seasonal past."),
    r"studentt": (r"$y\sim t_\nu(\mu,\sigma)$",
                 r"The **Student-t output** predicts the location $\mu$, the scale $\sigma$ and the degrees of "
                 r"freedom $\nu$ of the next value's distribution:",
                 "Its heavy tails make it robust to outliers."),
    r"decomp": (r"$x=x^{\mathrm{trend}}+x^{\mathrm{rem}}$",
               "The **decomposition** splits the input into a trend, computed as a moving average, and the "
               "remainder:",
               "Each part gets its own linear map from the last $L$ values to the next $H$."),
    r"resblock": (r"$\mathrm{RB}(x)=\mathrm{Dropout}\left(W_2\,\mathrm{ReLU}(W_1x)\right)+W_sx$",
                 "TiDE's **residual block** is a one-hidden-layer ReLU network plus a linear skip $W_sx$ (a matrix, "
                 "because the input and output widths differ):",
                 "Every part of TiDE (feature projection, encoder, decoder and temporal decoder) is built from it. "
                 "The TiDE paper adds a layer norm on the output, which the Darts implementation leaves off by "
                 "default."),
    r"linskip": (r"$\hat{y}_{L+1:L+H}=\mathrm{TD}(\cdot)+A\,y_{1:L}$",
                 r"The **global linear skip** adds a single linear map $A$, of size $H \times L$, from the look-back "
                 r"to the horizon to the output $\mathrm{TD}(\cdot)$ of the temporal decoder:",
                 "TiDE therefore contains a purely linear forecaster, like DLinear, and the fully connected network "
                 "only has to learn what the linear map misses."),
    r"boost": (r"$\hat{y}=\sum_{m=1}^{M}\eta\,f_m(x)$",
              r"In **gradient boosting**, the forecast is the sum of $M$ small regression trees $f_m$, each fitted "
              r"to what the trees before it got wrong and shrunk by the learning rate $\eta$:",
              ""),
}

_RULES = [  # (key, test on the lower-case label), checked in this order
    ("asinh", lambda s: "sinh" in s or "asinh" in s),
    ("robust", lambda s: s.startswith("robust scaling")),
    ("std", lambda s: any(w in s for w in ("standardis", "normalis", "revin", "running (causal) mean",
                                           "patch-wise scaling", "z-score"))),
    ("gatedemb", lambda s: "gated embedding" in s),
    ("pointemb", lambda s: "point-wise" in s and "embedding" in s),
    ("linemb", lambda s: "linear" in s and "embedding" in s),
    ("embed", lambda s: ("embedding" in s and not any(w in s for w in ("role", "[mask]"))) or "tokenizer" in s),
    ("patch", lambda s: "patch" in s and not any(w in s for w in ("head", "decoder", "output", "merge", "router", "fuse",
                                                                    "learnable"))),
    ("maxpool", lambda s: "maxpool" in s),
    ("interp", lambda s: s.startswith("interpolate")),
    ("slstm", lambda s: "slstm" in s),
    ("ssm", lambda s: "state-space" in s),
    ("causal", lambda s: "attention" in s and "causal" in s),
    ("attn", lambda s: "attention" in s),
    ("rms", lambda s: s.startswith("rmsnorm")),
    ("ln", lambda s: s.startswith("layernorm")),
    ("swiglu", lambda s: "swiglu" in s or "gated silu" in s),
    ("relu", lambda s: "feed forward (relu" in s),
    ("gelu", lambda s: "feed forward (gelu" in s),
    ("silu", lambda s: "feed forward (silu" in s),
    ("legendre", lambda s: "legendre" in s),
    ("mixture", lambda s: "mixture" in s),
    ("flow", lambda s: "flow-matching" in s or "flow matching" in s or "timeflow" in s),
    ("boost", lambda s: "sum of trees" in s),
    ("studentt", lambda s: "student-t" in s),
    ("lags", lambda s: "lag" in s and "feature" in s),
    ("decomp", lambda s: "moving average" in s or "decompos" in s),
    ("quantile", lambda s: "quantile" in s and any(w in s for w in ("head", "output"))),
]


def _assign_math(spec):
    """Give the first layer that names each operation its expression (an explicit ``math`` is kept)."""
    seen = set()

    def walk(items):
        for it in items:
            if isinstance(it, Box):
                if it.math:
                    seen.add(it.math)
                    continue
                s = it.text.lower().replace("\n", " ")
                for key, test in _RULES:
                    if test(s):
                        if key not in seen:
                            it.math = key
                            seen.add(key)
                        break
            elif isinstance(it, Fan):
                if it.math:
                    seen.add(it.math)
                elif it.labels and it.labels[0].startswith("E") and "moe" not in seen:
                    it.math = "moe"
                    seen.add("moe")
            elif hasattr(it, "items"):
                walk(it.items)

    for col in spec.columns:
        walk(col.items)


_dg.MATH_LOOKUP = lambda key: (MATH[key][0] if key in MATH else (key if key and key.startswith("$") else None))


def _add(key, spec, overview, complexity, facts, config=None):
    ARCH[key] = {"spec": spec, "overview": overview, "complexity": complexity, "facts": facts,
                 "config": config or {}}


def figure(key, path=None, show=True, vertical=False, style="explainer"):
    """Draw the architecture of model ``key``, saved to ``path`` (.png and .pdf) when given.

    The default ``style="explainer"`` draws the step-by-step explainer of ``ch03_zeroshot.explainers``: four
    numbered panels from the data the model reads to the forecast it returns. ``style="classic"`` draws the
    layer diagram of ``ARCH[key]["spec"]`` left to right (``vertical=True``: bottom to top)."""
    if style == "explainer" and not vertical:
        from ch03_zeroshot.explainers import SPECS, figure as explainer_figure
        if key in SPECS:
            return explainer_figure(key, path=path, show=show)
    spec = ARCH[key]["spec"]
    if vertical:
        return draw(spec, path=path, show=show)
    _assign_math(spec)
    return draw_h(spec, path=path, show=show)


def math_notes(key):
    """Markdown paragraphs that explain the expressions written in the figure of ``key``, each in display maths."""
    spec = ARCH[key]["spec"]
    _assign_math(spec)
    keys = []

    def walk(items):
        for it in items:
            m = getattr(it, "math", None)
            if m and m in MATH and m not in keys:
                keys.append(m)
            if hasattr(it, "items"):
                walk(it.items)

    blocks = []
    for col in spec.columns:                 # main rows first, as the figure reads
        walk([it for it in col.items if not isinstance(it, Block)])
        blocks += [it for it in col.items if isinstance(it, Block)]
    walk(blocks)                             # then the opened-up blocks (panels A, B)
    out = []
    for k in keys:
        expr, before, after = MATH[k]
        out.append(f"{before}\n\n${expr}$\n\n{after}".rstrip())   # display maths: $$...$$
    return "\n\n".join(out)


def _attn_block(n, mixer="Causal self-attention (RoPE)", ffn="Feed forward (SwiGLU)", norm="RMSNorm",
                accent=None, name="block", extra=None):
    items = [Res([Box(norm), Box(mixer, "accent" if accent == "mixer" else "dark", name=name + "_mix")])]
    if extra:
        items += extra
    items.append(Res([Box(norm), Box(ffn, "accent" if accent == "ffn" else "mid", name=name + "_ffn")]))
    return Block(items, repeat=str(n), name=name)


# ----------------------------------------------------------------------------- Chronos-2
_add("chronos2", Spec("chronos2", "Chronos-2 (Amazon, 120M): encoder-only, group attention", [Column(
    title="Chronos-2", inp="targets + past-only + known-future covariates, one row each",
    out="21 quantiles per future step (point forecast = median)", items=[
        Box("Normalise each row (context mean, sd),\nthen arcsinh", name="scale"),
        Box("Non-overlapping patches, P = 16\n+ relative time index + observed mask", name="patch"),
        Box("Residual MLP patch embedding (3P = 48 → 768)"),
        Box("[context patches]  REG  [future patches]", name="reg"),
        Block([Res([Box("RMSNorm"), Box("Time attention (RoPE)", "dark", math="t5rope")]),
               Res([Box("RMSNorm"), Box("Group attention\n(across the rows of a group, no position)", "accent",
                                        name="group", math="-")]),
               Res([Box("RMSNorm"), Box("Feed forward (ReLU, 3072)", "mid")])], repeat="12", name="block"),
        Box("Final RMSNorm"),
        Box("Residual MLP quantile head on future patches\n(768 → 16 steps × 21 quantiles)", name="head"),
    ])], notes=[
        Note("scale", "Mean and sd from the context;\narcsinh compresses large values\nbut is linear near zero", "left"),
        Note("reg", "Known-future covariates fill the\nfuture patches; targets and\npast-only rows are masked there", "right"),
        Note("group", "At each patch index, a row reads\nthe other rows of its group:\nthis is how covariates and\nother targets inform a forecast", "right"),
        Note("block", "T5-style encoder:\nd_model 768, 12 heads", "left", dy=-0.8),
        Note("head", "Direct multi-step: all future\npatches in one forward pass", "right")]),
    overview=("Chronos-2 is an encoder-only transformer. It lays out every series of a forecasting problem, targets "
              "and covariates alike, as separate rows of patches, and its blocks alternate between two kinds of "
              "attention: time attention, which looks along one row, and group attention, which looks across the rows"
              " at the same time step. Thanks to this layout, the same weights handle a single series, several series"
              " together, or a series with covariates, all in context and without fine-tuning. The future is part of "
              "the input too: the patches after the forecast origin carry the known-future covariates and a mask for "
              "everything still unknown, and a residual output head turns them into 21 quantiles in a single pass."),
    complexity=("Each block costs about O(V · N² · d) for time attention, with V rows of N = (T + H)/16 patches, plus"
                " O(N · V² · d) for group attention. One forward pass covers the whole horizon, up to 64 output "
                "patches (1,024 steps)."),
    facts=[("encoder-only, T5 encoder design", "Ansari et al. 2025, Sec. 3.2"),
           ("standardisation, then arcsinh", "Sec. 3.1, Eq. 1-2"),
           ("patch + time index + mask, residual network from 3P inputs to width D", "Sec. 3.1, Eq. 3"),
           ("REG token between context and future patches", "Sec. 3.1"),
           ("time attention uses RoPE; group attention has no positional embedding", "Sec. 3.2"),
           ("21 quantiles 0.01..0.99", "Sec. 3.2 Quantile Head"),
           ("block = TimeSelfAttention, GroupSelfAttention, FeedForward, each pre-norm + residual",
            "chronos 2.3.2 chronos2/model.py Chronos2EncoderBlock"),
           ("output residual block 768 -> 336 = 16 x 21", "loaded module tree")],
    config={"num_layers": 12, "d_model": 768, "num_heads": 12, "d_ff": 3072,
            "chronos_config.input_patch_size": 16, "chronos_config.context_length": 8192})

# ----------------------------------------------------------------------------- Chronos-Bolt
_add("chronos-bolt", Spec("chronos-bolt", "Chronos-Bolt small (Amazon, 48M): T5 encoder-decoder, direct quantiles", [
    Column(title="Encoder", inp="one univariate context (up to 2,048 steps)", out="", items=[
        Box("Instance normalisation (context mean, sd)", name="scale"),
        Box("Patches, P = 16, + observed mask"),
        Box("Residual MLP patch embedding (32 → 512)"),
        Box("[context patches]  REG"),
        Block([Res([Box("RMSNorm"), Box("Self-attention\n(T5 relative position bias)", "dark", math="t5bias")]),
               Res([Box("RMSNorm"), Box("Feed forward (ReLU, 2048)", "mid")])], repeat="6", name="enc"),
        Box("Final RMSNorm", name="enc_out")]),
    Column(title="Decoder", inp="one learned start token", out="64 steps × 9 quantiles", items=[
        Box("Token embedding (a single decoder token)", name="tok"),
        Block([Res([Box("RMSNorm"), Box("Self-attention", "dark", math="-")]),
               Res([Box("RMSNorm"), Box("Cross-attention to\nencoder states", "dark", name="xattn", math="-")]),
               Res([Box("RMSNorm"), Box("Feed forward (ReLU, 2048)", "mid")])], repeat="6", second=True,
              name="dec"),
        Box("Final RMSNorm"),
        Box("Residual MLP head → 64 × 9\nquantiles at once", "accent", name="head")])],
    links=[Link("enc_out", "xattn")],
    notes=[Note("scale", "Chronos (2024) tokenised values\ninto bins; Bolt keeps real values\nin patches instead", "left"),
           Note("head", "Above 64 steps: all 9 quantile\npaths are fed back, and the 81\nresults cut back to 9 quantiles", "right")]),
    overview=("Chronos-Bolt is the faster successor of the original Chronos. The original model scaled each value "
              "into one of 4,096 bins, treated the bins as words and generated the forecast one token at a time, like"
              " a language model. Chronos-Bolt instead feeds patches of real values to a T5 encoder, and its decoder "
              "writes 64 steps of 9 quantiles in a single step. It is univariate: each series is forecast on its own."),
    complexity=("The encoder's self-attention costs O(N² · d) for N = T/16 patches (at most 128), and the decoder "
                "reads them once for every 64-step chunk of the forecast. It is among the fastest models here on a "
                "CPU."),
    facts=[("patch-based T5 encoder-decoder, direct multi-step quantiles", "chronos-bolt model card; Ansari et al. 2024"),
           ("context 2048, patch 16, prediction length 64, 9 quantiles, REG token", "config.json chronos_config"),
           ("standardisation with nanmean / nanstd", "chronos 2.3.2 chronos_bolt.py InstanceNorm"),
           ("decoder input is a single start token; output block 512 -> 576 = 64 x 9", "chronos_bolt.py decode; module tree")],
    config={"num_layers": 6, "num_decoder_layers": 6, "d_model": 512, "num_heads": 8, "d_ff": 2048,
            "chronos_config.input_patch_size": 16, "chronos_config.prediction_length": 64})

# ----------------------------------------------------------------------------- TimesFM-2.5
_add("timesfm25", Spec("timesfm25", "TimesFM-2.5 (Google, 231M): decoder-only patch transformer", [Column(
    title="TimesFM-2.5", inp="one univariate context (up to 16,384 steps)",
    out="mean + 9 quantiles per step; longer horizons continue autoregressively", items=[
        Box("Input patches, P = 32, + padding mask", name="patch"),
        Box("Causal normalisation per patch\n(running mean, sd)", name="revin"),
        Box("Residual MLP tokenizer (64 → 1280)"),
        Block([Res([Box("RMSNorm"), Box("Causal self-attention\n(RoPE, QK-norm, per-dim scale)", "dark",
                                        math="pdsattn"),
                    Box("RMSNorm (post)")]),
               Res([Box("RMSNorm"), Box("Feed forward (SiLU, 1280)", "mid"), Box("RMSNorm (post)")])],
              repeat="20", name="block"),
        Fan(["point head: 128 steps", "quantile head: 1,024 steps"], "accent", name="head",
            caption="two residual heads, 10 outputs per step"),
    ])], notes=[
        Note("revin", "Statistics use only past patches,\nso nothing leaks from the future", "left"),
        Note("block", "16 heads of width 80;\nKV cache during decoding", "left", dy=-0.8),
        Note("head", "Output patch (128) is 4 × the input patch (32): fewer autoregressive steps; a "
                     "second head gives the quantiles for up to 1,024 steps in one pass", "right")]),
    overview=("TimesFM treats a series the way a language model treats text. Each patch of 32 steps is a token, a "
              "causal (decoder-only) transformer reads the tokens in time order, and the output at the last token "
              "predicts the next 128 steps. Version 2.5 adds a running normalisation that uses only past values, "
              "normalisation of the queries and keys inside attention (QK-norm) and a quantile head. Covariates are "
              "not part of the network: the library's XReg option fits a linear regression on the known-future "
              "covariates in context and combines it with the network's forecast."),
    complexity=("The transformer costs O(L · N² · d) for N = T/32 tokens and L = 20 layers. A horizon longer than 128"
                " steps needs one extra decoding pass for every further 128 steps."),
    facts=[("decoder-only, input patch 32, output patch 128", "Das et al. 2024, Sec. 4 and Fig. 1"),
           ("20 layers, hidden 1280, 16 heads of 80, context 16384", "config.json"),
           ("tokenizer residual block on [values, mask] 64 -> 1280", "module tree; timesfm_2p5_torch.py forward"),
           ("pre and post RMSNorm around attention and FFN; QK RMSNorm; RoPE; PerDimScale", "module tree"),
           ("running per-patch mean/sd normalisation (revin with running stats)", "timesfm_2p5_torch.py decode"),
           ("XReg covariates = in-context linear regression", "timesfm README, covariates section")],
    config={"num_hidden_layers": 20, "hidden_size": 1280, "num_attention_heads": 16, "patch_length": 32,
            "horizon_length": 128, "context_length": 16384})

# ----------------------------------------------------------------------------- TimesFM-3
_add("timesfm3", Spec("timesfm3", "TimesFM-3 (Google, 331M): decoder-only, sequence + variate attention", [Column(
    title="TimesFM-3", inp="targets + past-only + known-future covariates (up to 32 rows)",
    out="64 steps × 9 quantiles per target row", items=[
        Box("Input patches, P = 32, per row; running mean, sd\n(iterative RevIN), optional linear detrending",
            name="norm"),
        Box("Token = this patch (32) + next 64 steps of the same row\n(future values visible for known-future rows only)",
            "accent", name="token"),
        Box("Residual MLP (values + masks, 192 → 1280)"),
        Block([Res([Box("RMSNorm"), Box("Causal sequence attention\n(RoPE, QK-norm, per-dim scale)", "dark",
                                        math="pdsattn"), Box("RMSNorm (post)")]),
               Res([Box("RMSNorm"), Box("Variate attention (across rows)", "accent", name="var", math="-"),
                    Box("RMSNorm (post)")]),
               Res([Box("RMSNorm"), Box("Feed forward (ReLU, 1280)", "mid"), Box("RMSNorm (post)")])],
              repeat="20", name="block"),
        Box("Linear head (1280 → 64 × 9)"),
    ])], notes=[
        Note("token", "How known-future covariates\nenter: each token also sees\nthe covariate's next 64 values", "right"),
        Note("var", "Targets read covariates and\neach other at the same patch", "right"),
        Note("norm", "Causal statistics: patch k is\nnormalised with data up to k", "left")]),
    overview=("TimesFM-3 keeps the decoder-only patch design of TimesFM and adds variate attention, so one model "
              "reads several targets and covariates together. Each token carries its own input patch and, for a "
              "known-future covariate, also that covariate's values over the next output patch. No paper describes "
              "the model yet, so the description here comes from the model card and the released code (timesfm "
              "3.0.2)."),
    complexity=("Each layer costs O(V · N² · d) for attention along time plus O(N · V² · d) for variate attention, "
                "for up to V = 32 rows."),
    facts=[("stacked mixing transformer with variate attention and CPM iterative RevIN", "model card google/timesfm-3.0-pytorch"),
           ("input patch 32, output patch 64, 9 quantiles, max 32 variates", "config.json; timesfm3/torch/configs.py"),
           ("20 MixingTransformer layers: seq_attn (RoPE), var_attn (no RoPE), FFN, pre+post RMSNorm", "module tree"),
           ("token = [patch values, rolled next-64 values] + masks = 192 inputs", "timesfm3/torch/model.py _preprocess"),
           ("head Linear 1280 -> 576 = 64 x 9", "module tree")],
    config={"input_patch_len": 32, "output_patch_len": 64, "use_variate_attention": True})

# ----------------------------------------------------------------------------- TiRex
_add("tirex", Spec("tirex", "TiRex (NX-AI, 35M): xLSTM (sLSTM) decoder with state tracking", [Column(
    title="TiRex", inp="one univariate context", out="32 steps × 9 quantiles per patch", items=[
        Box("Instance normalisation (context mean, sd)"),
        Box("Patches, P = 32, + missing mask"),
        Box("Residual MLP patch embedding (64 → 512)"),
        Block([Res([Box("RMSNorm"), Box("sLSTM layer (4 heads,\nrecurrent state, exponential gates)", "accent",
                                        name="slstm")]),
               Res([Box("RMSNorm"), Box("Feed forward (SwiGLU, 1408)", "mid")])], repeat="12", name="block"),
        Box("Final RMSNorm"),
        Box("Residual MLP head (512 → 32 × 9)", name="head"),
    ])], notes=[
        Note("slstm", "A recurrent cell instead of\nattention: memory grows\nlinearly with the context", "right"),
        Note("head", "Longer horizons: future patches\nare fed in as 'missing' (CPM)\nand predicted in place", "right")]),
    overview=("TiRex has no attention at all. It is built on sLSTM, the scalar-memory recurrent cell of the xLSTM "
              "family, whose running state tracks a series well over both short and long horizons. It is trained with"
              " contiguous patch masking (CPM): future patches are treated as missing inputs, so the model learns to "
              "forecast several patches at once instead of feeding its own forecasts back in."),
    complexity=("The cost grows only linearly with the context: O(L · N · d²) for N = T/32 patches, with constant "
                "memory per step."),
    facts=[("xLSTM blocks with sLSTM, pre-RMSNorm residual, FFN, final RMSNorm", "Auer et al. 2025, Sec. 2 and Fig. 2"),
           ("z-score scaling; patch embedding: values + mask through a residual block", "Sec. 2"),
           ("contiguous patch masking (CPM) for multi-patch forecasts", "Sec. 2.1 and Fig. 3"),
           ("gated feed forward SiLU(W_g x) ⊙ (W_u x)", "tirex 1.x models/slstm/block.py FeedForward"),
           ("12 sLSTMBlocks, embedding 512, FFN 1408, head 512 -> 288 = 32 x 9", "module tree")],
    config={})

# ----------------------------------------------------------------------------- TiRex-2
_add("tirex2", Spec("tirex2", "TiRex-2 (NX-AI, 83M): xLSTM time mixer + attention variate mixer", [Column(
    title="TiRex-2", inp="targets + past-only + known-future covariates",
    out="32 steps × 9 quantiles per target row (up to 320 steps)", items=[
        Box("Normalise (context mean, sd), then arcsinh;\nbinary rows bypassed"),
        Box("Patches, P = 32, + mask → residual MLP (64 → 512)"),
        Block([Res([Box("RMSNorm"), Box("Time mixer: forward xLSTM over the past;\nreverse xLSTM for known-future rows",
                                        "accent", name="time"),
                    Box("Linear fusion of both directions")]),
               Res([Box("RMSNorm"), Box("Feed forward (SwiGLU)", "mid")]),
               Res([Box("RMSNorm"), Box("Variate mixer: group attention\n(covariates never read targets)",
                                        "dark", name="var", math="causal")]),
               Res([Box("RMSNorm"), Box("Feed forward (SiLU, 2048)", "mid")])], repeat="12", name="block",
              label="time + variate block"),
        Box("Final norm (T5-style RMS, no mean),\nresidual MLP head (512 → 32 × 9)"),
    ])], notes=[
        Note("time", "Blocks alternate mLSTM and\nsLSTM cells; the reverse pass\nlets future-known inputs flow\nback to the forecast", "right"),
        Note("var", "Asymmetric mask: targets attend\nto covariates, not the reverse", "right"),
        Note("block", "12 blocks, width 512", "left")]),
    overview=("TiRex-2 extends TiRex to several series and to covariates. Each block first mixes information along "
              "time with xLSTM, using a forward pass over the past and a reverse pass with the same weights over the "
              "known-future rows, and then mixes across series with attention whose mask stops covariates from "
              "reading the targets. The model can also update a forecast as each new value arrives, at a constant "
              "cost per step."),
    complexity="The time mixer is linear in length, O(V · N · d²); the variate mixer costs O(N · V² · d).",
    facts=[("alternating time mixer and variate mixer", "Podest, Pichler et al. 2026, Sec. 3.2 and Fig. 2"),
           ("standardise + arcsinh with binary bypass", "Sec. 3.2, App. C; config scaler_cfg"),
           ("reverse, weight-tied xLSTM pass for future-known covariates, fused linearly", "Sec. 3.2"),
           ("asymmetric group attention", "Sec. 3.2, App. B"),
           ("time-mixer FFN gated (SwiGLU); variate-mixer FFN SiLU 4 x 512; output norm T5-style (no bias, no mean subtraction)",
            "tirex2 model/component bi_xlstm.py, attention_block.py; model-config.yaml act_func"),
           ("12 MultivariateBlocks alternating mLSTM / sLSTM, emb 512, patch 32", "config.json; module tree"),
           ("maximum prediction length 320", "config future_len; tirex-2 0.3.1 warning")],
    config={"num_blocks": 12, "embedding_dim": 512, "input_patch_size": 32, "future_len": 320})

# ----------------------------------------------------------------------------- Timestral
_TIMESTRAL_NOTES = [
    Note("dhf", "Decoupled Horizon Forecasting:\ncontext tokens are causal; learned\nprediction tokens read the context\nbut never each other's future", "right"),
    Note("var", "Every 4th block; known-future rows\nattend only to each other, past-only\ncovariates never read targets", "right"),
    Note("norm", "Causal normalisation + arcsinh;\nknown-future rows use the full window", "left")]
for _key, _title, _n, _d, _h, _ff in [("timestral", "Timestral-209M", 24, 768, 12, 2048),
                                      ("timestral-5m", "Timestral-5M", 8, 192, 3, 512)]:
    _add(_key, Spec(_key, f"Timestral (Mistral AI, {_title.split('-')[1]}): causal transformer with decoupled horizon tokens", [Column(
        title=_title, inp="targets + past-only + known-future covariates",
        out="21 quantiles per step, up to 1,024 steps in one pass", items=[
            Box("Causal normalisation (running mean, sd),\nthen arcsinh, per row", name="norm"),
            Box(f"Patches, P = 32, values + missing flags\n→ residual MLP (64 → {_d})"),
            Box("[context tokens] + [learned prediction tokens]", "accent", name="dhf"),
            Block([Block([Res([Box("RMSNorm"), Box("Causal time attention\n(RoPE on half the head dims, sigmoid gate)",
                                                   "dark", math="gatedattn")]),
                          Res([Box("RMSNorm"), Box(f"Feed forward (SwiGLU, {_ff})", "mid")])], repeat="3",
                         label="time blocks"),
                   Block([Res([Box("RMSNorm"), Box("Causal time attention\n(same as in the time blocks)", "dark", math="-")]),
                          Res([Box("RMSNorm"), Box("Variate attention (no position, sigmoid gate)", "accent",
                                                   name="var", math="-")]),
                          Res([Box("RMSNorm"), Box(f"Feed forward (SwiGLU, {_ff})", "mid")])], repeat="1",
                         second=True, label="time + variate block")], repeat=str(_n // 4), name="block"),
            Box("Final RMSNorm"),
            Box(f"Residual MLP head → 32 steps × 21 quantiles\n(median + softplus increments)"),
        ])], notes=_TIMESTRAL_NOTES + [Note("block", f"{_n} blocks in all, d_model {_d},\n{_h} heads", "left", dy=-0.8)]),
        overview=("Timestral reads the context patches causally and forecasts through separate learned prediction "
                  "tokens, a design its authors call Decoupled Horizon Forecasting, so a forecast of up to 32 patches"
                  " needs a single forward pass. Every fourth block adds attention across series, with masks that "
                  "keep covariates from seeing the targets. The quantiles are built as a median plus positive "
                  "increments, so they can never cross."),
        complexity=(f"Time attention costs O(V · N² · d) over N = (T + H)/32 tokens; attention across series "
                    f"runs in {_n // 4} of the {_n} blocks."),
        facts=[("causal encoder with Decoupled Horizon Forecasting (learned prediction tokens)",
                "Mistral AI 2026, Sec. 3.1 and Fig. 5"),
               ("variate attention every 4th block, masking rules for covariates", "Sec. 3.2"),
               ("patch 32 with missing indicators, residual MLP embedding; causal standardisation + asinh",
                "Sec. 3.3"),
               ("pre-norm RMSNorm, SwiGLU, sigmoid-gated attention, RoPE on half head dims", "Sec. 3.3; config"),
               ("21 quantiles: median plus softplus increments, 32 steps per token", "Sec. 3.3"),
               (f"d_model {_d}, {_n} blocks, {_h} heads, SwiGLU width {_ff}", "Table 1; config.json")],
        config={"patch_size": 32, "d_model": _d, "num_heads": _h, "mlp_hidden_dim": _ff})

# ----------------------------------------------------------------------------- TTM
_add("ttm", Spec("ttm", "TTM-r2 (IBM, 1-5M): Tiny Time Mixer, an MLP-mixer without self-attention", [Column(
    title="TTM-r2", inp="one context of 512 steps per channel", out="96 steps (cut to H): point forecast", items=[
        Box("Instance normalisation (RevIN)"),
        Box("Patches, P = 64 (8 patches)"),
        Box("Linear patch embedding (+ frequency token\nin r2.1 checkpoints)", name="emb"),
        Block([Box("TSMixer block: MLP across patches and\nfeatures + simple gated attention\n(a softmax feature gate)", "accent",
                   name="mixer"),
               Box("Adaptive patching: merge or split patches\nper level (3 levels)", "plain")], repeat="2",
              name="backbone", label="backbone"),
        Block([Box("Small TSMixer decoder (d = 128)", "mid"),
               Box("Exogenous mixer (fine-tuning only)", "plain", name="exog")], repeat="2", second=True,
              name="decoder", label="decoder"),
        Box("Linear forecast head (→ 96 steps)"),
    ])], notes=[
        Note("mixer", "No self-attention, no recurrence:\nsmall MLPs and gates, so it runs\nin milliseconds on a CPU", "right"),
        Note("exog", "Channel and exogenous mixing\nexist but need fine-tuning;\nzero-shot TTM is univariate", "right"),
        Note("emb", "Resolution prefix: a learned token\nfor the sampling frequency", "left")]),
    overview=("Tiny Time Mixers (TTM) have no attention and only one to a few million parameters. They are MLP-Mixer "
              "models: small fully connected networks mix the patches, first across time and then across features, "
              "and adaptive patching lets the model look at the context at several resolutions. IBM releases many "
              "checkpoints, each tied to one context length and horizon; the adapter picks the one for a 512-step "
              "context and the next horizon up."),
    complexity="Each mixer layer costs O(N · d²) with N = 8 patches, which makes TTM the cheapest model here.",
    facts=[("TSMixer backbone, adaptive patching, resolution prefix tuning, small decoder, linear head",
            "Ekambaram et al. 2024, Sec. 2-3.1 and Fig. 2"),
           ("exogenous mixer used in fine-tuning", "Sec. 3.2"),
           ("context 512, patch 64, d_model 192, 2 layers, adaptive levels 3, decoder d_model 128, 2 layers",
            "config.json (main revision)"),
           ("revision chosen per horizon, e.g. 512-48-ft-r2.1", "granite-tsfm 0.3.10 get_model log")],
    config={"context_length": 512, "patch_length": 64, "d_model": 192, "num_layers": 2,
            "adaptive_patching_levels": 3, "decoder_d_model": 128, "decoder_num_layers": 2})

# ----------------------------------------------------------------------------- FlowState
_add("flowstate", Spec("flowstate", "FlowState r1.1 (IBM, 18.5M): state-space encoder + functional decoder", [Column(
    title="FlowState", inp="one univariate context, no patching", out="9 quantiles at any horizon", items=[
        Box("Causal normalisation (running mean, sd)"),
        Box("Linear embedding of every time step\n(value + missing flag, 2 → 512)", name="emb"),
        Block([Res([Box("S5 state-space layer (HiPPO init),\nstep size scaled by sampling rate", "accent",
                        name="ssm"),
                    Box("Output gate (sigmoid)"),
                    Res([Box("RMSNorm"), Box("Feed forward (SwiGLU, 1024)", "mid")]),
                    Box("LayerNorm")])], repeat="6", name="enc", label="S5 layer"),
        Box("Functional basis decoder: coefficients of\nLegendre polynomials (256 per quantile)", "accent",
            name="dec"),
        Box("Evaluate the continuous forecast on the\ntarget grid, undo normalisation"),
    ])], notes=[
        Note("ssm", "SSM state evolves in continuous\ntime: rescaling its step size adapts\nthe same weights to any sampling rate", "right"),
        Note("dec", "Forecast is a smooth function,\nsampled at the step you need", "right")]),
    overview=("FlowState replaces the transformer with an S5 state-space encoder that runs over every time step, and "
              "its decoder outputs the coefficients of a smooth function of time (a sum of Legendre polynomials) "
              "instead of values at fixed steps. Scaling the step size of the state-space layer and the decoder's "
              "sampling step to the data makes the model sampling-rate equivariant: one set of weights serves hourly,"
              " daily or monthly data."),
    complexity=("The cost is linear in the context length, O(L · T · d), computed with parallel scans; there is no "
                "attention."),
    facts=[("linear embedding without patching, S5 encoder, functional basis decoder", "Graf et al. 2025, Sec. 3 and Fig. 1"),
           ("causal normalisation", "Sec. 5.1"),
           ("sampling-rate scale factors for encoder and decoder", "Sec. 3.3"),
           ("S5 block with sigmoid output gate, then a two-layer gated MLP (expansion 2) and LayerNorm, all inside a skip",
            "Sec. 3.1, App. (18.6M variant); granite-tsfm FlowStateS5Layer"),
           ("6 S5 layers, state 512, embedding 512, Legendre ('legs') decoder 256, 9 quantiles", "config.json; module tree"),
           ("decoder Linear 512 -> 2304 = 256 x 9", "module tree")],
    config={"encoder_num_layers": 6, "encoder_state_dim": 512, "embedding_feature_dim": 512, "decoder_dim": 256,
            "decoder_type": "legs"})

# ----------------------------------------------------------------------------- PatchTST-FM
_add("patchtst-fm", Spec("patchtst-fm", "PatchTST-FM r2 (IBM, 385M): Conformer encoder over overlapping patches", [Column(
    title="PatchTST-FM", inp="one univariate context (up to 8,192 steps)", out="99 quantiles per step", items=[
        Box("Instance normalisation (RevIN)"),
        Box("Overlapping patches: length 16, stride 8;\nfuture patches are masked", name="patch"),
        Box("Residual MLP embedding (values + mask)\n+ learned position embedding"),
        Block([Res([Box("LayerNorm"), Box("½ feed forward (GELU)", "mid")]),
               Res([Box("LayerNorm"), Box("Multi-head self-attention\n(bidirectional, 16 heads)", "dark")]),
               Res([Box("LayerNorm"), Box("Depthwise convolution\n(kernel 5 or 3)", "accent", name="conv")]),
               Res([Box("LayerNorm"), Box("½ feed forward (GELU)", "mid")])], repeat="30", name="block",
              label="Conformer block"),
        Box("Pre-head LayerNorm, residual MLP head"),
        Box("Overlap-add of patch outputs\n(Hamming-window weighting)", name="ola"),
    ])], notes=[
        Note("conv", "Conformer: convolution adds\nlocal smoothing between\nneighbouring patches", "right"),
        Note("patch", "Forecasting = filling in\nmasked future patches", "left"),
        Note("ola", "Overlapping patches are\nblended back into one series", "right")]),
    overview=("PatchTST-FM scales up the PatchTST idea (patches as tokens, one series at a time) into a foundation "
              "model. Its Conformer blocks place attention and a depthwise convolution between two half-step feed-"
              "forward layers, and it forecasts by reconstructing masked future patches. Overlapping patches are "
              "blended back together with a Hamming window."),
    complexity=("O(L · N² · d) with N ≈ (T + H)/8 overlapping patches and L = 30 wide layers, which makes it the "
                "heaviest model here."),
    facts=[("conformer blocks: half FFN, MHSA, conv, half FFN", "model card ibm-granite/granite-timeseries-patchtst-fm-r2"),
           ("30 layers, d_model 1024, 16 heads, patch 16 stride 8, 99 quantiles, context 8192", "config.json"),
           ("non-causal, Hamming patch-loss windowing, pre-head norm", "config.json"),
           ("depthwise Conv2d kernels (1,5)/(1,3), RevIN, residual in/out layers", "module tree")],
    config={"n_layer": 30, "d_model": 1024, "n_head": 16, "d_patch": 16, "patch_stride": 8, "num_quantile": 99})

# ----------------------------------------------------------------------------- t0
_add("t0", Spec("t0", "t0-gamma (The Forecasting Company, 256M): time + variate attention", [Column(
    title="t0-gamma", inp="targets + past-only + known-future covariates",
    out="21 quantiles per step (single pass up to 1,024 steps)", items=[
        Box("Causal normalisation (running mean, sd),\nthen arcsinh; known-future rows use\nwhole-window statistics", name="norm"),
        Box("Patches, P = 32: values + within-patch time\n+ validity mask → residual MLP"),
        Box("+ role embedding (target / past / known-future)", "accent", name="role"),
        Block([Block([Res([Box("RMSNorm"), Box("Time attention (RoPE + xPos decay;\ncausal, bidirectional for known-future)",
                                               "dark")]),
                      Res([Box("RMSNorm"), Box("Feed forward (SwiGLU)", "mid")])], repeat="2", label="time layers"),
               Block([Res([Box("RMSNorm"), Box("Variate attention (asymmetric:\nknown-future rows do not read back)",
                                               "accent", name="var", math="-")]),
                      Res([Box("RMSNorm"), Box("Feed forward (SwiGLU)", "mid")])], repeat="1", second=True,
                     label="variate layer")], repeat="8", name="block"),
        Box("Final RMSNorm"),
        Box("Residual MLP decoder: next patch, quantiles\nmonotone by construction (softplus steps)"),
    ])], notes=[
        Note("var", "Asymmetry stops a two-hop leak\nfrom targets into earlier steps\nthrough future covariates", "right"),
        Note("norm", "Input step t is scaled with\nstatistics up to t; the target\nwith statistics frozen at T", "left"),
        Note("block", "24 layers = 16 time + 8 variate;\nwidth 1024, 8 heads", "left", dy=-1.2)]),
    overview=("t0 is a decoder-style patch transformer whose layers repeat a pattern of two time-attention layers and"
              " one variate-attention layer. A learned role embedding tells the model whether a row is a target, a "
              "past-only covariate or a known-future covariate, and the attention masks follow those roles. The t0 "
              "report describes the alpha and beta checkpoints; t0-gamma is a larger, later member of the family."),
    complexity="The time layers cost O(V · N² · d) and the variate layers O(N · V² · d), with N = (T + H)/32.",
    facts=[("decoder-style patch transformer, decoupled time and variate attention", "Meyer et al. 2026, Sec. 4.1"),
           ("causal scaling + arcsinh; known-future scaled over the whole span", "Sec. 4.1 Eq. 2"),
           ("embedding: values, within-patch time index, mask; role embedding", "Sec. 4.1"),
           ("pattern 2 time layers then 1 variate layer; asymmetric variate attention", "Sec. 4.1"),
           ("monotone quantiles via softplus increments; single pass up to 1024 steps", "Sec. 4.1, 4.3"),
           ("RoPE with xPos decay by patch position; pre-norm RMSNorm, QK-norm, SwiGLU", "Sec. 4.1, Table 2"),
           ("the t0 report details t0-alpha (width 512, 102M); t0-gamma is the larger release: 24 layers, embed 1024, "
            "8 heads, FFN 2048, patch 32, group_every_n 3, 21 quantiles", "config.json; module tree")],
    config={"num_layers": 24, "embed_dim": 1024, "num_heads": 8, "patch_size": 32, "group_every_n": 3})

# ----------------------------------------------------------------------------- TabPFN-TS
_add("tabpfn-ts", Spec("tabpfn-ts", "TabPFN-TS (Prior Labs): forecasting as tabular regression with TabPFN-v2", [Column(
    title="TabPFN-TS", inp="context rows (time, value) + known-future covariates",
    out="predictive distribution per future row → median", items=[
        Box("Build a table: one row per time step", name="table"),
        Box("Features: running index,\ncalendar sin/cos, seasonal\nFourier terms (auto-detected),\nknown-future covariates",
            "accent", name="feat", width=2.8),
        Box("Train rows = context (features, y);\ntest rows = horizon (features only)"),
        Block([Res([Box("Attention across features (per row)", "dark")]),
               Res([Box("Attention across rows\n(test rows read train rows only)", "dark", name="rows")]),
               Res([Box("Feed forward", "mid")])], repeat="12", name="block", label="TabPFN-v2 layer"),
        Box("Output: histogram over y (discretised distribution)"),
    ])], notes=[
        Note("feat", "No lags: the forecast for a\nfuture row depends only on its\nfeatures, so all rows are\npredicted at once", "right"),
        Note("rows", "In-context learning: the\n'training set' is just more\ninput to one forward pass", "right"),
        Note("table", "Each target series is its own\ntable; covariates are extra\nfeature columns", "left")]),
    overview=("TabPFN-TS turns forecasting into filling in a table. Each time step becomes a row with time features "
              "(a running index, calendar terms and automatically detected seasonal terms) and any known-future "
              "covariates. TabPFN-v2, a transformer pre-trained on synthetic tables, then predicts the target in the "
              "horizon rows from the context rows, in one pass."),
    complexity=("Attention over the rows costs O(R² · d) for R = T + H rows, multiplied by the number of features for"
                " the attention across features. That makes it the slowest model on a CPU here, so its context is "
                "capped."),
    facts=[("features: running index, calendar, auto seasonal; no lag features", "Hoo et al. 2025, Sec. 3"),
           ("TabPFN-v2 two-way attention (features, samples), outputs a discretised distribution",
            "Hollmann et al. 2025 (Nature), Methods"),
           ("ungated v2 regressor weights used", "tabpfn-time-series 1.3.0; adapter note"),
           ("12 layers in TabPFN-v2 regressor", "Hollmann et al. 2025, Methods")],
    config={})

# ----------------------------------------------------------------------------- Moirai 1.1
_add("moirai11", Spec("moirai11", "Moirai-1.1 small (Salesforce, 14M): masked encoder, any-variate attention", [Column(
    title="Moirai-1.1", inp="all variates (targets + covariates) flattened into one sequence",
    out="mixture distribution → 100 samples → median", items=[
        Box("Instance normalisation per variate\n(context mean, sd)"),
        Box("Patch size chosen by frequency: one of\n8, 16, 32, 64, 128, each with its own projection",
            "accent", name="patch"),
        Box("Future patches → learned [mask] embedding"),
        Block([Res([Box("RMSNorm"), Box("Any-variate attention: RoPE in time\n+ learned bias: same / other variate",
                                        "accent", name="anyvar", math="anyvar")]),
               Res([Box("RMSNorm"), Box("Feed forward (SwiGLU)", "mid")])], repeat="6", name="block"),
        Box("Final RMSNorm"),
        Box("Mixture head: Student-t, normal, negative\nbinomial, log-normal (per patch size)", name="head"),
    ])], notes=[
        Note("anyvar", "Variates are told apart by two learned biases, not by an order or an embedding", "right"),
        Note("patch", "Large patches for high-frequency\ndata, small for low frequency", "left"),
        Note("head", "Probabilistic by design;\nwe score the sample median", "right")]),
    overview=("Moirai is a masked encoder. All series, targets and covariates alike, are flattened into one sequence "
              "of tokens. Any-variate attention tells the series apart with a learned bias that says whether two "
              "tokens belong to the same series, while rotary embeddings encode time, so the model accepts any number"
              " of targets and covariates. The future patches enter as mask tokens, and their outputs give the "
              "parameters of a mixture distribution."),
    complexity=("Full attention over V · N tokens costs O((V · N)² · d): quadratic in the number of series as well as"
                " in time."),
    facts=[("masked encoder, multi patch-size input/output projections", "Woo et al. 2024, Sec. 3.1.1"),
           ("any-variate attention: RoPE + binary attention bias", "Sec. 3.1.2, Eq. 2-3"),
           ("pre-norm RMSNorm, QK-norm, SwiGLU, no biases; learnable [mask] embedding", "Sec. 3.1 and Fig. 2"),
           ("mixture: Student-t, negative binomial, log-normal, low-variance normal", "Sec. 3.1.3"),
           ("6 layers, d_model 384, 6 heads, patch sizes 8-128", "config.json; module tree")],
    config={"d_model": 384, "num_layers": 6})

# ----------------------------------------------------------------------------- Moirai-MoE
_add("moirai-moe", Spec("moirai-moe", "Moirai-MoE small (Salesforce, 117M total, 11M active): decoder-only sparse MoE", [Column(
    title="Moirai-MoE", inp="all variates flattened into one sequence", out="mixture distribution per next patch",
    items=[
        Box("Normalise (observed context mean, sd);\npatches, P = 16 (one size)", name="norm"),
        Box("Residual MLP patch projection (16 → 384)"),
        Block([Res([Box("RMSNorm"), Box("Causal any-variate attention\n(RoPE + variate bias)", "dark",
                                        math="anyvar")]),
               Res([Box("RMSNorm"), Box("Gate: distance to token clusters\nof a pre-trained Moirai", "accent",
                                        name="gate"),
                    Fan(["E1", "E2", "…", "E32"], "mid", name="experts", caption="top-2 of 32")])],
              repeat="6", name="block"),
        Box("Final RMSNorm"),
        Box("Mixture head (Student-t, normal,\nnegative binomial, log-normal)"),
    ])], notes=[
        Note("gate", "Experts specialise by token,\nnot by frequency: replaces\nMoirai's 5 patch-size layers", "right"),
        Note("experts", "Each expert is a SwiGLU\nfeed forward (512)", "right")]),
    overview=("Moirai-MoE replaces Moirai's separate patch layers for each frequency with a single patch size, and "
              "lets a sparse mixture of experts specialise on different token patterns instead. It is decoder-only "
              "(each patch predicts the next), and its router is built from k-means centroids of the token "
              "representations of a pre-trained Moirai."),
    complexity=("Attention costs O((V · N)² · d). Each token runs only 2 of the 32 expert feed-forward networks, so "
                "the compute per token matches a dense model of about 11M parameters."),
    facts=[("decoder-only, single projection, sparse MoE replacing FFN", "Liu, Liu et al. 2024, Sec. 3"),
           ("token-cluster gating (k-means centroids of pretrained Moirai)", "Sec. 3.2.1, Eq. 6"),
           ("top-2 of 32 experts; small: 6 layers, d 384, d_ff 512, 11M active / 117M total", "Sec. 4.1, Table 1"),
           ("normaliser statistics from the first part of the sequence in training (Sec. 3.1); at inference the "
            "whole observed context", "uni2ts moirai_moe/module.py PackedStdScaler"),
           ("residual MLP projection", "Sec. 3.1"),
           ("patch size 16; 32 GatedLinearUnit experts", "config.json; module tree")],
    config={"d_model": 384, "num_layers": 6, "d_ff": 512, "patch_sizes": [16]})

# ----------------------------------------------------------------------------- Moirai 2.0
_add("moirai2", Spec("moirai2", "Moirai-2.0 small (Salesforce, 11M): decoder-only, quantile loss", [Column(
    title="Moirai-2.0", inp="one univariate context", out="9 quantiles; long horizons by autoregressive decoding",
    items=[
        Box("Instance normalisation (context mean, sd)"),
        Box("Patches, P = 16, + missing indicator\n→ residual block (32 → 384)", width=2.7),
        Block([Res([Box("RMSNorm"), Box("Causal self-attention (RoPE)", "dark")]),
               Res([Box("RMSNorm"), Box("Feed forward (SwiGLU, 1024)", "mid")])], repeat="6", name="block"),
        Box("Final RMSNorm"),
        Box("Residual head: 4 future patches × 9 quantiles\n(multi-token prediction)", "accent", name="head"),
        Box("Autoregressive multi-quantile decoding:\nexpand 9 × 9 paths, collapse to 9", "accent", name="ar"),
    ])], notes=[
        Note("head", "Each token predicts 4 patches\n(64 steps) at once", "right"),
        Note("ar", "Feeds back quantiles, not just\nthe median, to keep uncertainty", "right"),
        Note("block", "Univariate: the paper dropped\nmultivariate and covariate support", "left")]),
    overview=("Moirai 2.0 simplifies Moirai: a decoder-only transformer with one patch size, trained with a quantile "
              "(pinball) loss instead of a mixture likelihood, that predicts four patches at each step. It is 30 "
              "times smaller than Moirai-1.0-Large and univariate."),
    complexity="O(L · N² · d) for N = T/16, with one decoding round per 64 steps.",
    facts=[("decoder-only, single patch, quantile loss, multi-token prediction", "Liu, Aksu et al. 2025, Sec. 3.1"),
           ("patch + missing indicator, residual block from 2p inputs to width d", "Sec. 3.1 Eq. 1"),
           ("autoregressive multi-quantile decoding (expand, collapse)", "Sec. 3.2, Algorithm 1"),
           ("univariate; covariates dropped", "Sec. 3.1 and 6"),
           ("6 layers, d 384, d_ff 1024, patch 16, 4 predict tokens, 9 quantiles; head 384 -> 576", "config.json; module tree")],
    config={"d_model": 384, "num_layers": 6, "d_ff": 1024, "patch_size": 16, "num_predict_token": 4})

# ----------------------------------------------------------------------------- Toto 2.0
_add("toto2", Spec("toto2", "Toto-2.0 (Datadog, 22M): decoder-only, time and variate attention", [Column(
    title="Toto-2.0", inp="targets of one problem (variates of one group)", out="32 steps × 9 quantiles per patch",
    items=[
        Box("Robust causal scaler: running mean,\nsd, then arcsinh"),
        Box("Patches, P = 32, + mask; contiguous patch\nmasking (CPM) blanks the future patches", "accent",
            name="cpm"),
        Box("Residual MLP patch embedding (64 → 512)"),
        Block([Res([Box("RMSNorm"), Box("Causal time attention (RoPE + xPos,\nper-dim scale)", "dark",
                                        math="toto2attn")]),
               Res([Box("RMSNorm"), Box("Feed forward (SwiGLU, 1368)", "mid")])], repeat="5", name="block",
              label="time block"),
        Block([Res([Box("RMSNorm"), Box("Variate attention (full, across rows)", "accent", name="var",
                                        math="-")]),
               Res([Box("RMSNorm"), Box("Feed forward (SwiGLU, 1368)", "mid")])], repeat="1", second=True,
              label="variate block"),
        Box("Final RMSNorm"),
        Box("Residual MLP quantile head (512 → 32 × 9)"),
    ])], notes=[
        Note("cpm", "One pass forecasts many\npatches (stable to ~768 steps)", "left"),
        Note("var", "Exogenous inputs are planned\nbut not in the released 2.0", "right")]),
    overview=("Toto 2.0 keeps the decoder of Toto, with layers that alternate attention along time and across series."
              " It replaces Toto's step-by-step sampling with contiguous patch masking, a nine-quantile output head "
              "and an arcsinh input scaler. The 22m checkpoint has five time layers and one variate layer."),
    complexity="The time layers cost O(V · N² · d) and the variate layer O(N · V² · d), with N = T/32.",
    facts=[("decoder-only, alternating time (causal) and variate (full) attention", "Khwaja et al. 2026, Sec. 2"),
           ("contiguous patch masking, quantile head, arcsinh scaler, residual MLP projections, PerDimScale",
            "Sec. 2.1-2.4"),
           ("patch 32", "Sec. 2.4; config.json"),
           ("22m: 6 layers = 5 time + 1 variate, d_model 512, 8 heads, d_ff 1368", "config.json; module tree")],
    config={"num_layers": 6, "d_model": 512, "num_heads": 8, "patch_size": 32, "d_ff": 1368})

# ----------------------------------------------------------------------------- Toto 1.0
_add("toto1", Spec("toto1", "Toto-1.0 (Datadog, 151M): proportional factorised attention, Student-t mixture", [Column(
    title="Toto-1.0", inp="targets (+ known-future covariates as extra variates)",
    out="32 samples from a mixture of 24 Student-t → median", items=[
        Box("Causal normalisation per patch\n(Welford running mean, sd)", "accent", name="scale"),
        Box("Patches, P = 64 → linear embedding (768)"),
        Block([Res([Box("RMSNorm"), Box("Time-wise attention (causal,\nRoPE + xPos)", "dark")]),
               Res([Box("RMSNorm"), Box("Feed forward (SwiGLU, 3072)", "mid")])], repeat="11", name="block",
              label="time block"),
        Block([Res([Box("RMSNorm"), Box("Variate-wise attention\n(bidirectional, ID mask)", "accent", name="var",
                                        math="causal")]),
               Res([Box("RMSNorm"), Box("Feed forward (SwiGLU, 3072)", "mid")])], repeat="1", second=True,
              label="variate block"),
        Box("Unembed: each token → 64 step vectors"),
        Box("Student-t mixture head (24 components)\nfor every step of the next patch", name="head"),
    ])], notes=[
        Note("var", "11:1 ratio of time to variate\nlayers: cheaper than full\nspace-time attention", "right"),
        Note("scale", "Each patch scaled with data\nup to that patch only", "left"),
        Note("head", "Autoregressive: sample, append,\nrepeat; we use 32 samples", "right")]),
    overview=("Toto 1.0 was built for observability data, the metrics that monitor computer systems. Its decoder uses"
              " what its authors call proportional factorised attention: eleven layers that attend along time for "
              "every layer that attends across series. The inputs are scaled causally, patch by patch, the output is "
              "a mixture of Student-t distributions, and forecasts are sampled autoregressively, each new chunk fed "
              "back as input. Known-future covariates can be added as extra series."),
    complexity=("O(V · N² · d) for the time layers plus O(N · V² · d) for the variate layer; sampling repeats for "
                "every 64 steps of the horizon."),
    facts=[("patch 64, embedding 768, decoder-only, pre-norm RMSNorm, SwiGLU, RoPE+XPOS", "Cohen et al. 2025, Sec. 3.1"),
           ("proportional factorised attention 11:1", "Sec. 3.1, Fig. 2C"),
           ("causal patch-based instance normalisation (Welford)", "Sec. 3.1; App. A.1"),
           ("Student-T mixture head", "Sec. 3.1; App. A.3"),
           ("12 layers (11 time-wise + 1 variate-wise last), 12 heads, MLP 3072, 24 components; "
            "no final norm; unembed 768 → 64 × 768", "config.json; loaded weights (151.3M parameters)")],
    config={"num_layers": 12, "embed_dim": 768, "num_heads": 12, "patch_size": 64, "spacewise_every_n_layers": 12})

# ----------------------------------------------------------------------------- Sundial
_add("sundial", Spec("sundial", "Sundial base (Tsinghua, 128M): decoder-only transformer + flow-matching head", [Column(
    title="Sundial", inp="one univariate context (up to 2,880 steps)",
    out="20 sampled futures (720 steps each) → median", items=[
        Box("Instance normalisation (context mean, sd)"),
        Box("Patches, P = 16, + padding mask\n→ MLP embedding (32 → 768)"),
        Block([Res([Box("LayerNorm (pre)"), Box("Causal self-attention (RoPE,\nFlashAttention, KV cache)", "dark")]),
               Res([Box("LayerNorm (pre)"), Box("Feed forward (SwiGLU, 3072)", "mid")])], repeat="12",
              name="block"),
        Box("Final LayerNorm"),
        Box("TimeFlow head: MLP with 3 AdaLN residual\nblocks turns noise into a 720-step future,\nconditioned on the token",
            "accent", name="flow"),
    ])], notes=[
        Note("flow", "Flow matching: start from\nGaussian noise and integrate a\nlearned velocity in 50 steps", "right"),
        Note("block", "Continuous tokens: no value\nbins, no parametric density", "left")]),
    overview=("Sundial is a decoder-only transformer whose output layer is a generative flow-matching network called "
              "TimeFlow. Instead of predicting a mean or a fixed set of quantiles, it draws whole future paths from "
              "random noise, guided by the transformer's summary of the context. We draw 20 paths, as the model card "
              "does, and use their median as the point forecast."),
    complexity="O(L · N² · d) for N = T/16 tokens, plus 50 steps of a small network for each sampled path.",
    facts=[("patch embedding of [x_i, m_i] via shared MLP", "Liu, Qin et al. 2025, Sec. 4.1.1 Eq. 4"),
           ("decoder-only, Pre-LN, causal attention with RoPE, FlashAttention, KV cache", "Sec. 4.1.2 Eq. 5"),
           ("TimeFlow loss, multi-patch prediction F > P, sampling algorithm", "Sec. 4.1.3, Alg. 1"),
           ("hidden 768, 12 layers, 12 heads, FFN 3072, token 16, output 720, flow depth 3, 50 sampling steps",
            "config.json; loaded weights (128.3M parameters)")],
    config={"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12, "input_token_len": 16,
            "num_sampling_steps": 50})

# ----------------------------------------------------------------------------- Timer
_add("timer", Spec("timer", "Timer base (Tsinghua, 84M): GPT-style next-patch prediction", [Column(
    title="Timer", inp="one univariate context (up to 2,880 steps)", out="next 96 steps; repeat to extend", items=[
        Box("Instance normalisation (context mean, sd)"),
        Box("Non-overlapping patch\ntokens, P = 96", name="tok"),
        Box("Gated embedding (96 → 1024)", name="emb"),
        Block([Res([Box("Causal self-attention (RoPE)", "dark")]), Box("LayerNorm (post)"),
               Res([Box("Feed forward (SwiGLU, 2048)", "mid")]), Box("LayerNorm (post)")], repeat="8",
              name="block"),
        Box("Final LayerNorm"),
        Box("Linear projection to the next patch (1024 → 96)", "accent", name="head"),
    ])], notes=[
        Note("tok", "Single-series sequence format:\nevery series is one sequence\nof patch tokens", "left"),
        Note("head", "Trained like a language model:\npredict token i+1 from tokens 1..i\n(MSE loss)", "right"),
        Note("emb", "The released checkpoint gates the embedding and normalises after each residual "
                    "add (post-LayerNorm); the paper writes a plain linear embedding", "right")]),
    overview=("Timer applies the GPT recipe to time series. Patches of 96 steps are the tokens, a decoder-only "
              "transformer is trained to predict the next token, and long horizons come from generating one token "
              "after another. The 84M checkpoint is univariate and returns a point forecast."),
    complexity=("O(L · N² · d) with N = T/96, only 30 tokens for a context of 2,880 steps, so attention is cheap and "
                "most of the cost is in the wide layers."),
    facts=[("next token prediction on non-overlapping patch tokens, decoder-only", "Liu et al. 2024, Sec. 3.3 Eq. 3"),
           ("MSE loss; linear output W_d", "Sec. 3.3 Eq. 3-4"),
           ("checkpoint: gated embedding SiLU(W_g s) ⊙ (W_e s), post-LayerNorm blocks, final LayerNorm, gated SiLU FFN",
            "modeling_timer.py (TimerPointEmbedding, TimerDecoderLayer, TimerModel)"),
           ("hidden 1024, 8 layers, 8 heads, FFN 2048, token 96, RoPE",
            "config.json; loaded weights (84.1M parameters)")],
    config={"hidden_size": 1024, "num_hidden_layers": 8, "num_attention_heads": 8, "input_token_len": 96})

# ----------------------------------------------------------------------------- Time-MoE
_add("timemoe", Spec("timemoe", "Time-MoE base (Shi et al., 113M total, 50M active): decoder-only, point tokens, sparse experts", [Column(
    title="Time-MoE", inp="one univariate context (context + horizon ≤ 4,096)",
    out="multi-resolution heads: 1, 8, 32 or 64 steps per pass", items=[
        Box("Instance normalisation (context mean, sd)"),
        Box("Point-wise tokens: SwiGLU embedding of\neach value (1 → 384)", "accent", name="tok"),
        Block([Res([Box("RMSNorm"), Box("Causal self-attention (RoPE)", "dark")]),
               Res([Box("RMSNorm"), Fan(["E1", "E2", "…", "E8", "shared"], "mid", name="moe",
                                         caption="top-2 of 8 experts (768) + shared (1536)", math="moeshared")])],
              repeat="12",
              name="block"),
        Box("Final RMSNorm"),
        Fan(["1", "8", "32", "64"], "plain", name="heads", caption="horizon heads"),
    ])], notes=[
        Note("tok", "No patches: attention over\nevery single time step", "left"),
        Note("moe", "Each point goes to 2 of 8 experts;\na shared expert (sigmoid gate)\nsees every point", "right"),
        Note("heads", "Greedy schedule combines heads\nto reach any horizon", "right")]),
    overview=("Time-MoE is a decoder-only transformer that makes every single time step a token, and replaces the "
              "feed-forward layer with a sparse mixture of experts plus one shared expert. Several output heads "
              "forecast 1, 8, 32 or 64 steps ahead, and they are combined greedily to reach the requested horizon. "
              "The released '50M' checkpoint has 113M parameters in total, of which about 50M are active for each "
              "time step."),
    complexity=("O(L · T² · d), because every point is a token. Long contexts are therefore expensive, so the context"
                " plus the horizon is capped at 4,096 steps."),
    facts=[("point-wise tokenisation with SwiGLU embedding", "Shi et al. 2025, Sec. 3.1 Eq. 1"),
           ("RMSNorm, RoPE, causal attention, MoE with shared expert (sigmoid gate)", "Sec. 3.1 Eq. 2-8"),
           ("multi-resolution forecasting heads", "Sec. 3.1"),
           ("hidden 384, 12 layers, 12 heads, 8 experts (768) top-2 + shared expert (1536), heads [1, 8, 32, 64], "
            "max 4,096 positions", "config.json; loaded weights (113.4M parameters)")],
    config={"hidden_size": 384, "num_hidden_layers": 12, "num_attention_heads": 12, "num_experts": 8,
            "num_experts_per_tok": 2, "horizon_lengths": [1, 8, 32, 64]})

# ----------------------------------------------------------------------------- Kairos
_add("kairos", Spec("kairos", "Kairos (ShanghaiTech and Ant Group, 50M): mixture-of-size patching, encoder-decoder", [
    Column(title="Encoder", inp="one univariate context (up to 2,048 steps)", out="", items=[
        Box("Instance normalisation (context mean, sd)"),
        Box("Coarse segments of 128 steps"),
        Box("Router per segment: picks 3 of 5 experts\n(3 patch sizes + 2 'null' experts that add nothing)", "accent",
            name="router"),
        Fan(["32", "64", "128", "null", "null"], "plain", name="gran", caption="top-3 of 5: patch sizes or null"),
        Box("Fuse: expand to the finest patch, weighted sum"),
        Block([Res([Box("RMSNorm"), Box("Self-attention with DRoPE\n(instance-adaptive frequencies)", "accent",
                                        name="drope", math="drope")]),
               Res([Box("RMSNorm"), Box("Feed forward (ReLU, 2048)", "mid")])], repeat="6", name="enc"),
        Box("Encoder states", name="enc_out")]),
    Column(title="Multi-patch decoder", inp="(no input values: the tokens are learned)",
           out="2 patches × 64 steps × 9 quantiles per pass", items=[
        Box("2 learnable forecast tokens, one per future patch", name="ftok"),
        Block([Res([Box("RMSNorm"), Box("Self-attention (DRoPE)", "dark", math="-")]),
               Res([Box("RMSNorm"), Box("Cross-attention to\nencoder states", "dark", name="xattn", math="-")]),
               Res([Box("RMSNorm"), Box("Feed forward (ReLU, 2048)", "mid")])], repeat="6", second=True, name="dec"),
        Box("Residual MLP head (512 → 2048 → 64 × 9)")])],
    links=[Link("enc_out", "xattn")],
    notes=[Note("router", "Smooth stretches get coarse\npatches, busy ones fine patches", "left"),
           Note("drope", "Rotary frequencies scaled by an\nMLP of the series' spectrum;\npositions count real time steps", "left")]),
    overview=("Kairos adapts the way it cuts a series into tokens to each series. A router decides, segment by "
              "segment, which patch sizes to use, and 'null' experts let it use fewer of them. Its dynamic rotary "
              "embeddings (DRoPE) adapt their frequencies to the series' spectrum and to the mix of patch sizes. A "
              "decoder with learnable forecast tokens predicts several future patches in parallel."),
    complexity=("Encoder attention costs O(N² · d) over the fused tokens, and the decoder's cross-attention O(J · N ·"
                " d) for J forecast tokens."),
    facts=[("Mixture-of-Size encoder with top-K granularity routing and null experts", "Feng et al. 2025, Sec. 3.2"),
           ("DRoPE: instance-adaptive frequencies from FFT features + granularity-aware positions", "Sec. 3.3 Eq. 5-6"),
           ("multi-patch decoder with learnable forecast tokens via cross-attention", "Sec. 3.4"),
           ("T5-style 6 + 6 layers, d_model 512, 8 heads, FFN 2048 (ReLU), segment 128, patch sizes 128/64/32, "
            "2 null experts, top-3 routing, 2 decoder segments of 64, 9 quantiles",
            "config.json; loaded weights (50.1M parameters)")],
    config={"num_layers": 6, "num_decoder_layers": 6, "d_model": 512, "num_heads": 8, "input_patch_size": 128,
            "levels": 3, "n_null_experts": 2, "n_activated_experts": 3, "num_decoder_segments": 2,
            "prediction_length": 64})

# ----------------------------------------------------------------------------- Lag-Llama
_add("lag-llama", Spec("lag-llama", "Lag-Llama (Rasul et al., 2.5M): LLaMA-style decoder on lagged values", [Column(
    title="Lag-Llama", inp="one univariate context (32 tokens + the lags they need)",
    out="Student-t parameters for the next step; 100 sampled paths → median", items=[
        Box("Robust scaling (median, IQR)"),
        Box("Token for step t: 84 lagged values y(t−l)\n+ 6 date-time features + 2 scale statistics",
            "accent", name="tok"),
        Box("Linear projection (92 → 144)"),
        Block([Res([Box("RMSNorm"), Box("Causal self-attention (9 heads, RoPE)", "dark")]),
               Res([Box("RMSNorm"), Box("Feed forward (SwiGLU, 512)", "mid")])], repeat="8", name="block"),
        Box("Final RMSNorm"),
        Box("Distribution head: Student-t (df, loc, scale)", name="head"),
    ])], notes=[
        Note("tok", "Lags from 1 to 1,093 steps,\nchosen to cover daily, weekly,\nmonthly, quarterly and yearly\ncycles", "left"),
        Note("head", "One step at a time:\nsample, append, repeat", "right")]),
    overview=("Lag-Llama is one of the first open foundation models for forecasting. Each token is built from lagged "
              "values of the series, with lags chosen to cover several seasonal frequencies, plus date features and "
              "summary statistics. A small decoder in the style of the LLaMA language models reads the tokens and "
              "outputs a Student-t distribution for the next step."),
    complexity=("O(L · C² · d) for a context of C tokens, but the model needs one pass per forecast step, so H passes"
                " for each sampled path."),
    facts=[("lag-feature tokenisation, date-time covariates, summary statistics", "Rasul et al. 2023, Sec. 4.1"),
           ("LLaMA decoder: pre-RMSNorm, RoPE, causal attention", "Sec. 4.2"),
           ("Student-t distribution head", "Sec. 4.3"),
           ("8 layers, 9 heads × 16 = width 144, 84 lags, context 32, SwiGLU 512, final RMSNorm",
            "checkpoint hyper-parameters and weights (2.45M parameters)")],
    config={"n_layer": 8, "n_head": 9, "n_embd_per_head": 16, "context_length": 32, "n_lags": 84})


# ============================================================================ deep-learning baselines
_add("nhits", Spec("nhits", "N-HiTS (Challu et al., 2023): hierarchical interpolation MLP", [Column(
    title="N-HiTS", inp="last L values (input chunk)", out="H future steps = sum of stack outputs", items=[
        Block([Box("MaxPool with kernel kₗ (multi-rate sampling)", "accent", name="pool"),
               Box("MLP (ReLU)", "mid"),
               Box("Backcast + coarse forecast coefficients"),
               Box("Interpolate coefficients to H steps", "accent", name="interp"),
               Box("Subtract backcast from the block's input")], repeat="3", name="stack",
              label="stack"),
        Box("Sum the forecasts of all blocks"),
    ])], notes=[
        Note("pool", "One block per stack; early stacks see a heavily\npooled input (slow trends), later stacks finer detail",
             "right"),
        Note("interp", "Few coefficients, interpolated:\ncheap long horizons", "right")]),
    overview=("N-HiTS stacks small fully connected blocks. Each block looks at the input at a different sampling rate"
              " and forecasts a coarse version of the future, which is then interpolated to full length. Adding up "
              "the blocks rebuilds the forecast from the slow components to the fast ones."),
    complexity=("Linear in the input length and the horizon: a few small networks that train in seconds to minutes on"
                " a CPU."),
    facts=[("multi-rate pooling, hierarchical interpolation, residual stacking", "Challu et al. 2023 (AAAI), Sec. 3")])

_add("tft", Spec("tft", "Temporal Fusion Transformer (Lim et al., 2021): LSTM encoder-decoder + interpretable attention", [Column(
    title="TFT", inp="past target + past covariates + known-future covariates", out="H steps (one point forecast per step)",
    items=[
        Box("Variable selection networks (past, future, static)", "accent", name="vsn"),
        Box("LSTM encoder (past) → LSTM decoder (future)"),
        Box("Gated residual network + static enrichment"),
        Res([Box("Interpretable multi-head\nself-attention (masked)", "dark", math="causal")]),
        Box("Gated residual network (position-wise)"),
        Box("Linear output: one value per step\n(trained with squared error)"),
    ])], notes=[
        Note("vsn", "Learns a weight for every input\nvariable: built-in covariate\nimportance", "right")]),
    overview=("The Temporal Fusion Transformer (TFT) combines recurrent layers for local patterns with attention for "
              "long-range ones. Its variable selection networks learn a weight for each covariate, which makes it a "
              "natural supervised benchmark for the datasets with covariates."),
    complexity=("The LSTM costs O(T · d²) and attention O(T² · d) over the encoder and decoder steps; it is the "
                "slowest baseline to train."),
    facts=[("variable selection, LSTM encoder-decoder, static enrichment, interpretable attention",
            "Lim et al. 2021 (IJF), Sec. 4 and Fig. 2"),
           ("point output trained with mean squared error (TFT's quantile outputs are switched off here)",
            "baselines.py (likelihood=None); Darts TFTModel")])

_add("dlinear", Spec("dlinear", "DLinear (Zeng et al., 2023): decomposition + two linear layers", [Column(
    title="DLinear", inp="last L values", out="H future steps", items=[
        Box("Moving-average decomposition", "accent", name="dec"),
        Fan(["trend part", "remainder"], "plain", name="parts"),
        Fan(["Linear L → H", "Linear L → H"], "mid", name="lin"),
        Box("Sum"),
    ])], notes=[
        Note("dec", "Trend = moving average;\nremainder = series - trend", "right"),
        Note("lin", "One weight per (input step,\noutput step): no hidden layers", "right")]),
    overview=("DLinear asks how far a single linear map can go. The input is split into a trend (a moving average) "
              "and a remainder, each part is mapped to the horizon by one linear layer, and the two outputs are "
              "added."),
    complexity="O(L · H) multiply-adds per forecast, the cheapest neural baseline.",
    facts=[("series decomposition then two one-layer linear networks", "Zeng et al. 2023 (AAAI), Sec. 4")])

_add("tide", Spec("tide", "TiDE (Das et al., 2023): dense MLP encoder-decoder with a linear skip", [Column(
    title="TiDE", inp="look-back y(1..L) of every target + covariates over the look-back (and the horizon if known)",
    out="H future steps of every target", items=[
        Box("Feature projection: residual block per\ntime step (covariates → width 4)", "accent", name="proj"),
        Box("Flatten and concatenate: look-back y(1..L)\n+ projected covariates over 1..L+H", name="cat"),
        Box("Dense encoder: residual block (ReLU)\n→ embedding e (hidden size)", "mid", name="enc", math="resblock"),
        Box("Dense decoder: residual block\n→ H vectors d(t) of width p = 16", "mid", name="dec"),
        Box("Temporal decoder, one step at a time:\nresidual block on [d(t), projected covariates(t)]", "accent",
            name="tdec"),
        Box("Add the global linear skip:\nlook-back y(1..L) → H steps", name="skip", math="linskip"),
    ])], notes=[
        Note("proj", "Known-future covariates are projected\nover the horizon as well; past-only\ncovariates (a Darts "
             "extension) only\nover the look-back", "right"),
        Note("tdec", "Each future step sees its own\ncovariate values: a 'highway' from\nthe covariates to the "
             "forecast", "right"),
        Note("skip", "A pure linear forecaster is\nbuilt in; the MLP adds the rest", "right")]),
    overview=("TiDE (Time-series Dense Encoder) answers the question DLinear raised: it keeps the speed of a model "
              "built only from fully connected layers, but adds covariates and non-linearity. The look-back and the "
              "projected covariates are flattened into one vector, which residual blocks encode and then decode into "
              "one small vector per future step. A temporal decoder combines each step's vector with that step's "
              "covariates, and a global linear skip from the look-back to the horizon keeps a purely linear "
              "forecaster inside the model."),
    complexity=("O((L·V + (L+H)·w) · d) for the encoder, where w is the width of the projected covariates, plus O(d ·"
                " p · H) for the decoder, where p is the size of each step's vector: linear in look-back and horizon,"
                " with no attention. Its authors report training 5 to 10 times faster than the best transformer they "
                "compared it with."),
    facts=[("feature projection, dense encoder, dense decoder, temporal decoder, global residual (linear) connection",
            "Das et al. 2023 (TMLR), Sec. 4 and Fig. 1"),
           ("residual block: one ReLU hidden layer, linear skip, dropout, layer norm on the output", "Sec. 4, Fig. 1"),
           ("Darts version: all targets in one model, past covariates projected over the look-back, layer norm off by "
            "default; widths 4 (covariates), p = 16 (decoder output), 32 (temporal decoder), 1 encoder and 1 decoder "
            "block", "darts 0.41.0, models/forecasting/tide_model.py (_TideModule)")])

_add("gbdt", Spec("gbdt", "Gradient-boosted trees on lags (LightGBM, XGBoost, CatBoost)", [Column(
    title="Lag regression", inp="target lags + covariate lags (and known-future values)", out="H steps (direct multi-step)",
    items=[
        Box("Build a table: rows = time steps,\ncolumns = lags y(t-1..t-p), covariates", name="tab"),
        Fan(["tree 1", "tree 2", "…", "tree M"], "plain", name="trees", caption="boosting"),
        Box("Sum of trees → y(t+h)"),
        Box("Direct strategy: every step h = 1..H\nis predicted from the same origin", "accent", name="direct"),
    ])], notes=[
        Note("trees", "Each tree fits the residuals of\nthe trees before it", "right"),
        Note("direct", "LightGBM: one model per step;\nXGBoost, CatBoost: one multi-output\nmodel. "
             "No error feedback", "right")]),
    overview=("Gradient-boosted trees forecast by regression on lagged values. A table of lags (and covariates) is "
              "built from the history, and an ensemble of shallow trees, each fitted to the errors left by the trees "
              "before it, predicts every step of the horizon directly."),
    complexity=("Training time grows roughly in proportion to the number of trees M, rows n and features p, because "
                "all three libraries sort each feature into a histogram of bins before they search for splits. A "
                "prediction is a few tree look-ups."),
    facts=[("gradient boosting", "Friedman 2001 (Ann. Stat.)"), ("LightGBM", "Ke et al. 2017 (NeurIPS)"),
           ("XGBoost", "Chen & Guestrin 2016 (KDD)"), ("CatBoost", "Prokhorenkova et al. 2018 (NeurIPS)")])


def fact_check_table():
    """One row per checked claim: model, claim, where it was checked."""
    import pandas as pd

    return pd.DataFrame([{"model": k, "claim": c, "checked_against": s} for k, v in ARCH.items()
                         for c, s in v["facts"]])
