"""Figures of paper 5 (vector PDF for LaTeX, 300-dpi PNG for Word) from the finished study data in this repository.

File names follow the printed figure numbers of the article."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data5 as D  # noqa: E402

OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
WIDTH = 6.5            # inches, about the text width, so that printed font sizes equal the sizes set here

# Okabe-Ito colour-blind-safe palette; hatching keeps A3 and RCHO apart in greyscale
COL = {"A3": "#E69F00", "CHO": "#56B4E9", "RA3": "#CC79A7", "MCHO": "#009E73", "RCHO": "#D55E00"}
HATCH = {"A3": "\\\\\\", "CHO": "///", "RA3": "", "MCHO": "...", "RCHO": ""}
CARD_NAME = {"D": "D (dense urban, 10 GHz)", "F1": "F1 (FR1 macro, 4 GHz)", "F2": "F2 (FR2 beams, 28 GHz)"}
CARD_COL = {"D": "#0072B2", "F1": "#E69F00", "F2": "#009E73"}
CARD_MK = {"D": "o", "F1": "s", "F2": "^"}
SPEED_NAME = {"ped": "ped.", "v30": "30 km/h", "v60": "60 km/h"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": 0.7,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "hatch.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
})
META = {"Author": "Mostafa Jalili Sanjarani", "Creator": "matplotlib"}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), metadata=META)
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300, metadata={"Author": META["Author"]})
    plt.close(fig)


def eq_number(label):
    """Number of a labelled equation of the article."""
    return {"eq:adj": 3}[label]


def strat_mean_ci(rows, rule, m, seed=500, B=10000):
    """Mean over the worlds of one card and a 95 % bootstrap CI stratified by speed."""
    x = D.arr(rows, rule, m)
    st = {}
    for i, r in enumerate(rows):
        st.setdefault(r[0][1], []).append(i)
    rng = np.random.default_rng(seed)
    tot = np.zeros(B)
    for idx in st.values():
        idx = np.array(idx)
        tot += x[idx][rng.integers(0, len(idx), size=(B, len(idx)))].sum(1)
    tot /= len(x)
    return x.mean(), np.quantile(tot, 0.025), np.quantile(tot, 0.975)


def bars(ax, x, mu, err, r, w):
    ax.bar(x, mu, w * 0.92, color=COL[r], hatch=HATCH[r], edgecolor="#222222", lw=0.4, label=r, zorder=2)
    ax.errorbar(x, mu, yerr=err, fmt="none", ecolor="#000000", elinewidth=1.0, capsize=2.6, capthick=1.0,
                zorder=3)


# ----------------------------------------------------------------------------- Fig. 1: method diagram
def fig1_method():
    fig = plt.figure(figsize=(WIDTH, 4.7))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(-4, 70)
    ax.axis("off")
    FS = 8.0

    def box(x, y, w, h, lines, fc="#FFFFFF", fs=FS):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=0.9", fc=fc, ec="#333333",
                                    lw=0.8))
        n = len(lines)
        pitch = fs * 1.5
        for i, ln in enumerate(lines):
            ax.annotate(ln, xy=(x + w / 2, y + h / 2), xytext=(0, ((n - 1) / 2 - i) * pitch),
                        textcoords="offset points", ha="center", va="center", fontsize=fs,
                        fontweight="bold" if i == 0 else "normal")

    def arrow(x0, y0, x1, y1, ls="-", col="#333333"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=0.9, color=col, linestyle=ls, shrinkA=0, shrinkB=0,
                                    mutation_scale=10))

    fb = "#D55E00"
    ax.add_patch(FancyBboxPatch((24, 3.2), 25.5, 60.3, boxstyle="round,pad=0.3,rounding_size=1.2", fc="#F4F4F4",
                                ec="#999999", lw=0.6, ls="--"))
    ax.text(36.75, 67.3, "ROAM parts (source gNB)", ha="center", va="center", fontsize=8.5, fontweight="bold")
    ax.text(85.0, 67.3, "Rel-16 CHO configuration", ha="center", va="center", fontsize=8.5, fontweight="bold")

    box(0.5, 43, 20, 19, ["UE reports", "L3 RSRP of serving", "and neighbour cells;", "position, heading,", "speed"],
        fc="#EAF4FB")
    box(0.5, 8, 20, 19, ["Observed outcomes", "useful, unnecessary,", "too early, too late;", "RLF and", "re-establishment"],
        fc="#FDF0E6")

    parts = [
        ["1  Outcome memory", r"$H_{uj}$, $L_{uj}$ per route", "and 40 m bin"],
        ["2  Path-area term", r"$\gamma_u\,(f_{uj}-f_{uc})$", "over a 5 s sector"],
        ["3  Dwell / short stay", r"$s_{uj}=\mathrm{clip}(1-\hat T_{uj}/T^{s}_{\min},0,1)$"],
        ["4  Service weight", r"$w_{s(u)}-\bar w$", "of the 5QI class"],
    ]
    ys = [49, 35, 21, 7]
    for lines, y in zip(parts, ys):
        box(25, y, 23.5, 12, lines)

    box(53, 26, 13.5, 18, ["Adjustment", r"$D_{uj}$, Eq. (%d)" % eq_number("eq:adj"), "clipped to",
                           r"$[-6,6]$ dB"], fc="#FFF7D6")

    box(70.5, 46, 29, 16, ["Preparation", r"prepare $j$ if", r"$\mathrm{gap}_{uj}\geq o_{\mathrm{prep}}+\rho D_{uj}$",
                           r"keep top $n_{\max}\leq 8$"], fc="#E8F5EE")
    box(70.5, 24.5, 29, 16, ["Execution offset", r"a3-Offset $=q(o_{\mathrm{exec}}+D_{uj})$",
                             "0.5 dB steps, frozen", r"when $j$ is prepared"], fc="#E8F5EE")
    box(70.5, 3, 29, 15.5, ["UE executes CHO", "evaluates condEventA3;", "triggered candidate", "with the best gap"],
        fc="#EAF4FB")

    arrow(20.8, 52.5, 23.6, 52.5)
    for y in ys:
        arrow(48.8, y + 6, 52.7, 35)
    arrow(66.8, 38, 70.2, 50)
    arrow(66.8, 32, 70.2, 31)
    arrow(85, 45.7, 85, 40.8)
    arrow(85, 24.2, 85, 18.8)
    ax.text(84, 21.5, "RRCReconfiguration", fontsize=7.5, va="center", ha="right", style="italic")
    # feedback: execution outcome -> labels -> memory
    ax.plot([70.2, 10.5, 10.5], [1.2, 1.2, 1.4], color=fb, lw=0.9, ls="--")
    arrow(10.5, 1.4, 10.5, 7.6, ls="--", col=fb)
    ax.text(40, -2.4, "outcome labels feed the memory (online, network-wide)", fontsize=7.5, ha="center",
            va="center", color=fb)
    ax.plot([20.8, 22.2, 22.2], [17.5, 17.5, 58.5], color=fb, lw=0.9, ls="--")
    arrow(22.2, 58.5, 24.6, 58.5, ls="--", col=fb)
    save(fig, "fig1_method")


# ----------------------------------------------------------------------------- Fig. 2: prepared cells and RRC
def fig2_cost():
    fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 2.6))
    for ax, m, meths, lab in ((axs[0], "prepared_cells_mean", ("CHO", "MCHO", "RCHO"), "Prepared cells per UE (mean)"),
                              (axs[1], "rrc_messages_pm", ("A3", "CHO", "MCHO", "RCHO"), "RRC messages per UE-min")):
        n = len(meths)
        w = 0.8 / n
        for k, r in enumerate(meths):
            st = [strat_mean_ci(D.sub(c), r, m) for c in D.CARDS]
            mu = np.array([s[0] for s in st])
            err = np.array([[s[0] - s[1] for s in st], [s[2] - s[0] for s in st]])
            bars(ax, np.arange(3) + (k - (n - 1) / 2) * w, mu, err, r, w)
        ax.set_xticks(range(3))
        ax.set_xticklabels(list(D.CARDS))
        ax.set_ylabel(lab)
        ax.grid(axis="y", lw=0.3, alpha=0.6)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, ncol=n, loc="upper center", bbox_to_anchor=(0.5, 1.17), handlelength=1.6,
                  columnspacing=1.0)
    fig.tight_layout()
    save(fig, "fig2_cost")


# ----------------------------------------------------------------------------- Fig. 3: UHO and failures per card/speed
def fig3_uho_failures():
    meths = ("A3", "CHO", "MCHO", "RCHO")
    fig, axs = plt.subplots(2, 3, figsize=(WIDTH, 4.4))
    w = 0.19
    for j, c in enumerate(D.CARDS):
        for i, (m, lab) in enumerate((("unnecessary_pm", "Unnecessary HO per UE-min"),
                                      ("failures_pm", "Failures per UE-min"))):
            ax = axs[i, j]
            for k, r in enumerate(meths):
                mu, h = zip(*[D.seed_ci(D.sub(c, s), r, m) for s in D.SPEEDS])
                bars(ax, np.arange(3) + (k - 1.5) * w, mu, h, r, w)
            ax.set_xticks(range(3))
            ax.set_xticklabels([SPEED_NAME[s] for s in D.SPEEDS])
            if j == 0:
                ax.set_ylabel(lab)
            if i == 0:
                ax.set_title(CARD_NAME[c])
            ax.grid(axis="y", lw=0.3, alpha=0.6)
            ax.set_axisbelow(True)
    fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "fig3_uho_failures")


# ----------------------------------------------------------------------------- Fig. 4: paired scatter
def fig4_paired():
    fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 3.0))
    for ax, m, lab in ((axs[0], "unnecessary_pm", "unnecessary HO per UE-min"),
                       (axs[1], "failures_pm", "failures per UE-min")):
        hi = 0
        for c in D.CARDS:
            x, y = D.arr(D.sub(c), "CHO", m), D.arr(D.sub(c), "RCHO", m)
            ax.scatter(x, y, s=11, marker=CARD_MK[c], facecolor="none", edgecolor=CARD_COL[c], lw=0.8,
                       label=CARD_NAME[c])
            hi = max(hi, x.max(), y.max())
        ax.set_xscale("symlog", linthresh=0.01)
        ax.set_yscale("symlog", linthresh=0.01)
        lim = (0, hi * 1.3)
        ax.plot(lim, lim, color="#222222", lw=0.6, ls="--")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("CHO (tuned): " + lab)
        ax.set_ylabel("RCHO: " + lab)
        ax.grid(lw=0.3, alpha=0.5)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "fig4_paired")


# ----------------------------------------------------------------------------- Fig. 5: ablations
def fig5_ablation():
    parts = (("RCHO-H", "memory (H)"), ("RCHO-A", "area (A)"), ("RCHO-D", "dwell (D)"), ("RCHO-S", "service (S)"))
    fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 2.4))
    y = np.arange(len(parts))[::-1]
    for ax, m, pct, lab in ((axs[0], "unnecessary_pm", True, "Change of unnecessary HO\nwhen removed (% of RCHO)"),
                            (axs[1], "failures_pm", False, "Change of failures when removed\n(per UE-min)")):
        for yy, (a, name) in zip(y, parts):
            r = D.paired_ci(D.ROWS, a, "RCHO", m)
            f = 100.0 / r["base"] if pct else 1.0
            mu, lo, hi = r["mu"] * f, r["lo"] * f, r["hi"] * f
            helps = D.JUDGE["ablations"][a]["helps"]
            ax.errorbar(mu, yy, xerr=[[mu - lo], [hi - mu]], fmt="o", ms=5, capsize=3, elinewidth=1.0,
                        capthick=1.0, color="#D55E00" if helps else "#555555",
                        mfc="#D55E00" if helps else "white")
        ax.axvline(0, color="#222222", lw=0.6)
        if not pct:
            dl = max(0.10 * D.JUDGE["means"]["RCHO"]["failures_pm"], 0.002)
            for s in (-dl, dl):
                ax.axvline(s, color="#0072B2", lw=0.8, ls=":")
            ax.text(dl, len(parts) - 0.45, r" $+\delta_{\mathrm{abl}}$", color="#0072B2", fontsize=8, va="center")
            ax.text(-dl, len(parts) - 0.45, r"$-\delta_{\mathrm{abl}}$ ", color="#0072B2", fontsize=8, va="center",
                    ha="right")
        ax.set_yticks(y)
        ax.set_yticklabels([p[1] for p in parts])
        ax.set_xlabel(lab)
        ax.grid(axis="x", lw=0.3, alpha=0.6)
        ax.set_ylim(-0.6, len(parts) - 0.2)
    fig.tight_layout()
    save(fig, "fig5_ablation")


ALL = (fig1_method, fig2_cost, fig3_uho_failures, fig4_paired, fig5_ablation)
FIGLIST = tuple(f.__name__ for f in ALL)

if __name__ == "__main__":
    for f in ALL:
        f()
    print("figures written to", OUT)
