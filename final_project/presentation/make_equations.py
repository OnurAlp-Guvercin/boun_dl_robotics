"""Render PPO / GRPO loss equations to PNG via matplotlib mathtext."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "assets"
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#0F2C4A"


def render(name, tex, fontsize=26, color=NAVY, pad=0.25):
    fig = plt.figure(figsize=(0.1, 0.1))
    t = fig.text(0.5, 0.5, tex, fontsize=fontsize, color=color,
                 ha="center", va="center")
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight",
                pad_inches=pad, transparent=True)
    plt.close(fig)
    print("wrote", name)


# importance-sampling ratio
render(
    "eq_ratio.png",
    r"$r_t(\theta)=\dfrac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{old}}(a_t\mid s_t)}$",
    fontsize=24,
)

# PPO objective (uses a learned value/critic for the advantage)
render(
    "eq_ppo.png",
    r"$J_{PPO}(\theta)=\mathrm{E}\left[\min\left(r_t\,\hat{A}_t,\;"
    r"\mathrm{clip}(r_t,\,1-\epsilon,\,1+\epsilon)\,\hat{A}_t\right)\right]"
    r"-\beta\,\mathrm{KL}\left[\pi_\theta\,\|\,\pi_{ref}\right]$",
    fontsize=23,
)

# PPO advantage via GAE (needs critic V)
render(
    "eq_ppo_adv.png",
    r"$\hat{A}_t=\mathrm{GAE}(\delta_t,\;V_\psi(s_t))$",
    fontsize=23, color="#9A3412",
)

# GRPO objective (group of G samples, no critic)
render(
    "eq_grpo.png",
    r"$J_{GRPO}(\theta)=\mathrm{E}\left[\dfrac{1}{G}\sum_{i=1}^{G}"
    r"\min\left(\rho_{i}\,\hat{A}_{i},\;"
    r"\mathrm{clip}(\rho_{i},\,1-\epsilon,\,1+\epsilon)\,\hat{A}_{i}\right)\right]"
    r"-\beta\,\mathrm{KL}\left[\pi_\theta\,\|\,\pi_{ref}\right]$",
    fontsize=22,
)

# GRPO group-relative advantage (no critic — the group is the baseline)
render(
    "eq_grpo_adv.png",
    r"$\hat{A}_{i}=\dfrac{r_i-\mathrm{mean}(r_1,\dots,r_G)}{\mathrm{std}(r_1,\dots,r_G)}$",
    fontsize=23, color="#166534",
)
