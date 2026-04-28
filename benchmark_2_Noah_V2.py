"""
Benchmark: Fletcher-Reeves (FR) vs Polak-Ribière+ (PR+)
Gradient conjugué non-linéaire sur fonctions non-convexes
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from dataclasses import dataclass, field
from typing import List

# ──────────────────────────────────────────────────────────────
#  Fonctions test non-convexes
# ──────────────────────────────────────────────────────────────

def rosenbrock(x):
    return sum(100*(x[i+1]-x[i]**2)**2 + (1-x[i])**2 for i in range(len(x)-1))

def rosenbrock_grad(x):
    g = np.zeros_like(x)
    for i in range(len(x)-1):
        g[i]   += -400*x[i]*(x[i+1]-x[i]**2) - 2*(1-x[i])
        g[i+1] +=  200*(x[i+1]-x[i]**2)
    return g

def rastrigin(x):
    n = len(x)
    return 10*n + sum(xi**2 - 10*np.cos(2*np.pi*xi) for xi in x)

def rastrigin_grad(x):
    return 2*x + 20*np.pi*np.sin(2*np.pi*x)

def himmelblau(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

def himmelblau_grad(x):
    return np.array([
        4*x[0]*(x[0]**2 + x[1] - 11) + 2*(x[0] + x[1]**2 - 7),
        2*(x[0]**2 + x[1] - 11) + 4*x[1]*(x[0] + x[1]**2 - 7)
    ])

def ackley(x):
    n = len(x)
    a, b, c = 20, 0.2, 2*np.pi
    return (-a*np.exp(-b*np.sqrt(np.sum(x**2)/n))
            - np.exp(np.sum(np.cos(c*x))/n) + a + np.e)

def ackley_grad(x):
    n = len(x)
    a, b, c = 20, 0.2, 2*np.pi
    norm2 = np.sum(x**2)/n + 1e-14
    s2 = np.sum(np.cos(c*x))/n
    return a*b*np.exp(-b*np.sqrt(norm2))/(np.sqrt(norm2)*n)*x + np.exp(s2)*c*np.sin(c*x)/n

def beale(x):
    return ((1.5   - x[0] + x[0]*x[1]  )**2 +
            (2.25  - x[0] + x[0]*x[1]**2)**2 +
            (2.625 - x[0] + x[0]*x[1]**3)**2)

def beale_grad(x):
    a = 1.5   - x[0] + x[0]*x[1]
    b = 2.25  - x[0] + x[0]*x[1]**2
    c = 2.625 - x[0] + x[0]*x[1]**3
    return np.array([
        2*a*(-1+x[1])   + 2*b*(-1+x[1]**2)   + 2*c*(-1+x[1]**3),
        2*a*x[0]        + 4*b*x[0]*x[1]       + 6*c*x[0]*x[1]**2
    ])

def mccormick(x):
    return np.sin(x[0]+x[1]) + (x[0]-x[1])**2 - 1.5*x[0] + 2.5*x[1] + 1

def mccormick_grad(x):
    c = np.cos(x[0]+x[1])
    return np.array([c + 2*(x[0]-x[1]) - 1.5,
                     c - 2*(x[0]-x[1]) + 2.5])

FUNCTIONS = {
    "Rosenbrock 2D":  dict(f=rosenbrock,  g=rosenbrock_grad,  x0=np.array([-1.2, 1.0]),               f_opt=0.0),
    "Rosenbrock 5D":  dict(f=rosenbrock,  g=rosenbrock_grad,  x0=np.array([-1.2,1.0,-0.5,0.8,-1.0]),  f_opt=0.0),
    "Rosenbrock 10D": dict(f=rosenbrock,  g=rosenbrock_grad,  x0=np.random.RandomState(42).uniform(-2,2,10), f_opt=0.0),
    "Rastrigin 2D":   dict(f=rastrigin,   g=rastrigin_grad,   x0=np.array([2.5, -2.5]),               f_opt=0.0),
    "Himmelblau":     dict(f=himmelblau,  g=himmelblau_grad,  x0=np.array([0.0, 0.0]),                f_opt=0.0),
    "Ackley 2D":      dict(f=ackley,      g=ackley_grad,      x0=np.array([2.0, 2.0]),                f_opt=0.0),
    "Beale":          dict(f=beale,       g=beale_grad,       x0=np.array([1.0, 1.0]),                f_opt=0.0),
    "McCormick":      dict(f=mccormick,   g=mccormick_grad,   x0=np.array([0.0, 0.0]),                f_opt=-1.9133),
}

# ──────────────────────────────────────────────────────────────
#  Recherche linéaire (conditions de Wolfe fortes)
# ──────────────────────────────────────────────────────────────

def _zoom(f, g, x, d, a_lo, a_hi, phi0, dphi0, c1, c2):
    for _ in range(30):
        alpha = 0.5*(a_lo + a_hi)
        phi   = f(x + alpha*d)
        phi_lo = f(x + a_lo*d)
        if phi > phi0 + c1*alpha*dphi0 or phi >= phi_lo:
            a_hi = alpha
        else:
            dphi = g(x + alpha*d) @ d
            if abs(dphi) <= -c2*dphi0:
                return alpha
            if dphi*(a_hi - a_lo) >= 0:
                a_hi = a_lo
            a_lo = alpha
    return alpha

def wolfe_line_search(f, g, x, d, f0, g0, c1=1e-4, c2=0.9):
    dphi0 = g0 @ d
    if dphi0 >= 0:
        d = -g0
        dphi0 = g0 @ d
    alpha, alpha_prev, f_prev = 1.0, 0.0, f0
    for i in range(40):
        f_new = f(x + alpha*d)
        if f_new > f0 + c1*alpha*dphi0 or (i > 0 and f_new >= f_prev):
            return _zoom(f, g, x, d, alpha_prev, alpha, f0, dphi0, c1, c2)
        dphi = g(x + alpha*d) @ d
        if abs(dphi) <= -c2*dphi0:
            return alpha
        if dphi >= 0:
            return _zoom(f, g, x, d, alpha, alpha_prev, f0, dphi0, c1, c2)
        alpha_prev, f_prev = alpha, f_new
        alpha = min(2*alpha, 20.0)
    return alpha

# ──────────────────────────────────────────────────────────────
#  Gradient Conjugué Non-Linéaire
# ──────────────────────────────────────────────────────────────

@dataclass
class CGResult:
    method: str
    x_opt: np.ndarray
    f_opt: float
    n_iter: int
    n_restarts: int
    converged: bool
    time_sec: float
    history_f: List[float] = field(default_factory=list)
    history_gn: List[float] = field(default_factory=list)
    history_beta: List[float] = field(default_factory=list)

def conjugate_gradient(f, g, x0, method="FR", tol=1e-7, max_iter=5000):
    """
    Gradient conjugué non-linéaire avec restart de Powell.

    Formules de beta :
      FR  : β_k = ||g_{k+1}||² / ||g_k||²
      PR+ : β_k = max( g_{k+1}·(g_{k+1}−g_k) / ||g_k||² , 0 )
    """
    x    = x0.copy().astype(float)
    n    = len(x)
    grad = g(x)
    d    = -grad.copy()
    f_val = f(x)

    hist_f  = [f_val]
    hist_gn = [np.linalg.norm(grad)]
    hist_b  = [0.0]
    n_restarts = 0
    t0 = time.perf_counter()

    for k in range(max_iter):
        if np.linalg.norm(grad) < tol:
            break

        alpha    = wolfe_line_search(f, g, x, d, f_val, grad)
        x_new    = x + alpha * d
        grad_new = g(x_new)
        f_val    = f(x_new)

        gg_old = grad @ grad + 1e-16
        if method == "FR":
            beta = (grad_new @ grad_new) / gg_old
        else:  # PR+
            y    = grad_new - grad
            beta = max((grad_new @ y) / gg_old, 0.0)

        # Restart de Powell : toutes les n itérations ou si directions trop alignées
        if (abs(grad_new @ grad) / (grad_new @ grad_new + 1e-16) >= 0.2):
            d = -grad_new
            beta = 0.0
            n_restarts += 1
        else:
            d_cand = -grad_new + beta * d
            d = d_cand if d_cand @ grad_new < 0 else -grad_new

        x, grad = x_new, grad_new
        hist_f.append(f_val)
        hist_gn.append(np.linalg.norm(grad))
        hist_b.append(beta)

    return CGResult(
        method=method, x_opt=x, f_opt=f_val,
        n_iter=k+1, n_restarts=n_restarts,
        converged=np.linalg.norm(grad) < tol,
        time_sec=time.perf_counter()-t0,
        history_f=hist_f, history_gn=hist_gn, history_beta=hist_b,
    )

# ──────────────────────────────────────────────────────────────
#  Benchmark multi-départs
# ──────────────────────────────────────────────────────────────

def run_benchmark(n_runs=20, noise_scale=0.3, seed=0):
    rng = np.random.RandomState(seed)
    all_results = {}

    for fname, fdata in FUNCTIONS.items():
        print(f"  [{fname}]")
        runs = {"FR": [], "PR+": []}
        x0_base = fdata["x0"]
        for _ in range(n_runs):
            noise = rng.randn(len(x0_base)) * noise_scale
            x0 = x0_base + noise
            for method in ["FR", "PR+"]:
                r = conjugate_gradient(fdata["f"], fdata["g"], x0, method=method)
                runs[method].append(r)
        all_results[fname] = runs

    return all_results

# ──────────────────────────────────────────────────────────────
#  Figure 1 : Courbes de convergence (run médian)
# ──────────────────────────────────────────────────────────────

def plot_convergence_curves(all_results, save_path="convergence_curves.png"):
    funcs = list(all_results.keys())
    cols, colors = 4, {"FR": "#e07b39", "PR+": "#3a86ff"}
    rows = (len(funcs) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(18, rows*4))
    axes = axes.flatten()

    for idx, fname in enumerate(funcs):
        ax = axes[idx]
        for method in ["FR", "PR+"]:
            runs_sorted = sorted(all_results[fname][method], key=lambda r: r.n_iter)
            r = runs_sorted[len(runs_sorted)//2]
            ax.semilogy(r.history_gn, label=method, color=colors[method], lw=2)
        ax.set_title(fname, fontsize=10, fontweight="bold")
        ax.set_xlabel("Itérations", fontsize=8)
        ax.set_ylabel("||∇f||", fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        ax.tick_params(labelsize=7)

    for i in range(idx+1, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Convergence du gradient conjugué : FR vs PR+  (norme du gradient, run médian)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {save_path}")

# ──────────────────────────────────────────────────────────────
#  Figure 2 : Boxplots des métriques agrégées
# ──────────────────────────────────────────────────────────────

def plot_metrics_boxplots(all_results, save_path="metrics_boxplots.png"):
    funcs = list(all_results.keys())
    metrics = [("n_iter", "Nb itérations"), ("f_opt", "f(x*) final"), ("n_restarts", "Nb restarts")]
    colors  = {"FR": "#e07b39", "PR+": "#3a86ff"}
    x_pos   = np.arange(len(funcs))
    width   = 0.3

    fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 4*len(metrics)))

    for mi, (metric, label) in enumerate(metrics):
        ax = axes[mi]
        for j, method in enumerate(["FR", "PR+"]):
            data = [[getattr(r, metric) for r in all_results[fn][method]] for fn in funcs]
            ax.boxplot(data,
                       positions=x_pos + (j-0.5)*width,
                       widths=width*0.85,
                       patch_artist=True,
                       medianprops=dict(color="white", linewidth=2),
                       boxprops=dict(facecolor=colors[method], alpha=0.75),
                       whiskerprops=dict(color=colors[method]),
                       capprops=dict(color=colors[method]),
                       flierprops=dict(marker="o", markersize=3,
                                       markerfacecolor=colors[method], alpha=0.5))
            ax.plot([], [], color=colors[method], lw=8, alpha=0.75, label=method)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(funcs, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel(label, fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        if metric == "f_opt":
            ax.set_yscale("symlog", linthresh=1e-6)

    fig.suptitle("Comparaison FR vs PR+ : métriques agrégées (20 runs par fonction)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {save_path}")

# ──────────────────────────────────────────────────────────────
#  Figure 3 : Évolution du coefficient beta
# ──────────────────────────────────────────────────────────────

def plot_beta_evolution(all_results, save_path="beta_evolution.png"):
    funcs = list(all_results.keys())
    cols, colors = 4, {"FR": "#e07b39", "PR+": "#3a86ff"}
    rows = (len(funcs) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(18, rows*4))
    axes = axes.flatten()

    for idx, fname in enumerate(funcs):
        ax = axes[idx]
        for method in ["FR", "PR+"]:
            runs_sorted = sorted(all_results[fname][method], key=lambda r: r.n_iter)
            r = runs_sorted[len(runs_sorted)//2]
            ax.plot(r.history_beta, color=colors[method], lw=1.8, alpha=0.9, label=method)
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_title(fname, fontsize=10, fontweight="bold")
        ax.set_xlabel("Itérations", fontsize=8)
        ax.set_ylabel("β", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)

    for i in range(idx+1, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Évolution du coefficient β  (run médian)\n"
                 "Note : PR+ applique max(β,0) — les valeurs négatives sont écrêtées à 0",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {save_path}")

# ──────────────────────────────────────────────────────────────
#  Figure 4 : Taux de convergence
# ──────────────────────────────────────────────────────────────

def plot_convergence_rate(all_results, save_path="convergence_rate.png"):
    funcs  = list(all_results.keys())
    colors = {"FR": "#e07b39", "PR+": "#3a86ff"}
    x      = np.arange(len(funcs))
    width  = 0.35

    rates = {m: [np.mean([r.converged for r in all_results[fn][m]])*100
                 for fn in funcs] for m in ["FR", "PR+"]}

    fig, ax = plt.subplots(figsize=(14, 5))
    for j, method in enumerate(["FR", "PR+"]):
        ax.bar(x + (j-0.5)*width, rates[method], width=width*0.9,
               color=colors[method], alpha=0.82, label=method)
        for xi, val in zip(x, rates[method]):
            ax.text(xi + (j-0.5)*width, val + 1.5, f"{val:.0f}%",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(funcs, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Taux de convergence (%)", fontsize=11)
    ax.set_ylim(0, 115)
    ax.axhline(100, color="gray", ls="--", lw=0.8)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Taux de convergence : FR vs PR+  (20 runs par fonction, tol=1e-7)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {save_path}")

# ──────────────────────────────────────────────────────────────
#  Résumé textuel
# ──────────────────────────────────────────────────────────────

def print_summary(all_results):
    print("\n" + "="*92)
    print(f"{'Fonction':<18} {'Méthode':<6} {'Iter médian':>12} {'f* médian':>14} "
          f"{'Restarts':>10} {'Convergé%':>10} {'Temps(ms)':>10}")
    print("="*92)
    for fname in all_results:
        for method in ["FR", "PR+"]:
            runs  = all_results[fname][method]
            iters = np.median([r.n_iter      for r in runs])
            fopt  = np.median([r.f_opt       for r in runs])
            rests = np.median([r.n_restarts  for r in runs])
            conv  = np.mean([r.converged    for r in runs])*100
            tms   = np.mean([r.time_sec     for r in runs])*1000
            print(f"{fname:<18} {method:<6} {iters:>12.0f} {fopt:>14.3e} "
                  f"{rests:>10.0f} {conv:>9.1f}% {tms:>9.2f}")
        print("-"*92)


if __name__ == "__main__":
    print("=" * 60)
    print("  BENCHMARK : Fletcher-Reeves vs Polak-Ribière+")
    print("  Gradient conjugué — fonctions non-convexes")
    print("=" * 60)
    print("\n▶ Lancement du benchmark (20 runs × 8 fonctions × 2 méthodes)…\n")

    results = run_benchmark(n_runs=20, noise_scale=0.3)

    print("\n▶ Génération des figures…")
    plot_convergence_curves(results, "convergence_curves.png")
    plot_metrics_boxplots  (results, "metrics_boxplots.png")
    plot_beta_evolution    (results, "beta_evolution.png")
    plot_convergence_rate  (results, "convergence_rate.png")

    print_summary(results)
    print("\n✓ Terminé — 4 figures sauvegardées.")