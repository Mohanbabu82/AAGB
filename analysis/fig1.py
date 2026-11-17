"""Fig. 1: revised AAGB pipeline (schematic). Text is auto-shrunk so that it never leaves its box."""
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, BLUE, RED, GREEN, GREY = '#1a1a1a', '#1f5fa8', '#b3261e', '#2e7d32', '#5f5f5f'


def box(ax, x, y, w, h, text, ec=INK, fc='white', fs=7.6, ls='-', lw=1.0, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.0,rounding_size=0.09', ec=ec, fc=fc, lw=lw, ls=ls))
    t = ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=INK,
                fontweight='bold' if bold else 'normal', linespacing=1.3)
    fig = ax.figure
    r = fig.canvas.get_renderer()
    (x0, y0), (x1, y1) = ax.transData.transform((x, y)), ax.transData.transform((x + w, y + h))
    while fs > 5.0:
        e = t.get_window_extent(r)
        if e.width <= 0.93 * (x1 - x0) and e.height <= 0.86 * (y1 - y0):
            break
        fs -= 0.2
        t.set_fontsize(fs)
    return t


def arrow(ax, p, q, color=INK, ls='-', rad=0.0, lw=1.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=8, color=color, lw=lw, ls=ls,
                                 connectionstyle='arc3,rad=%s' % rad, shrinkA=1.5, shrinkB=1.5))


def main(out):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans'})
    fig, ax = plt.subplots(figsize=(7.4, 5.1), dpi=300)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, 14.8)
    ax.set_ylim(0, 10.2)
    ax.axis('off')
    fig.canvas.draw()

    # ---- Stage 1
    ax.add_patch(FancyBboxPatch((0.15, 5.25), 14.5, 4.8, boxstyle='round,pad=0.0,rounding_size=0.12', ec=RED, fc='#fdf4f3', lw=1.0, ls=(0, (4, 2))))
    ax.text(0.4, 9.72, 'Stage 1: attention module (fitted on training rows only, cross-fitted)', fontsize=8.0, color=RED, fontweight='bold', va='center')
    y0, h = 6.95, 1.15
    box(ax, 0.4, y0, 1.45, h, 'input\n' + r'$\mathbf{x}_i\in\mathbb{R}^d$')
    box(ax, 2.35, y0, 2.35, h, r'shared encoder $f_\theta$' + '\n2 × (Linear, BN, ReLU)')
    box(ax, 5.25, y0 + 0.8, 2.75, 1.0, 'feature attention\n' + r'$\mathbf{a}_{f,i}=\sigma(W_f^{\top}\mathbf{h}_i+\mathbf{b}_f)$', ec=BLUE)
    box(ax, 5.25, y0 - 0.65, 2.75, 1.0, 'sample attention\n' + r'$a_{s,i}=\sigma(\mathbf{w}_s^{\top}\mathbf{h}_i+b_s)$', ec=BLUE)
    box(ax, 8.6, y0, 2.75, h, 'input-level gate\n' + r'$\tilde{\mathbf{x}}_i=\mathbf{x}_i\odot\mathbf{a}_{f,i}\,a_{s,i}$', ec=RED, lw=1.5)
    box(ax, 11.95, y0, 2.45, h, r'auxiliary predictor $\phi_\psi$' + '\n' + r'BCE on $z_i=\phi_\psi(\tilde{\mathbf{x}}_i)$', ec=RED, ls=(0, (3, 2)))
    ym = y0 + h / 2
    arrow(ax, (1.85, ym), (2.35, ym))
    arrow(ax, (4.7, ym + 0.2), (5.25, y0 + 1.3))
    arrow(ax, (4.7, ym - 0.2), (5.25, y0 - 0.15))
    arrow(ax, (8.0, y0 + 1.3), (8.6, ym + 0.25), color=BLUE)
    arrow(ax, (8.0, y0 - 0.15), (8.6, ym - 0.25), color=BLUE)
    arrow(ax, (11.35, ym), (11.95, ym), color=RED)
    arrow(ax, (1.1, y0 + h), (9.6, y0 + h), color=GREY, rad=-0.3, lw=0.9)
    ax.text(3.2, 9.18, r'$\mathbf{x}_i$ re-enters at the gate, so column $j$ of $A_f$ multiplies feature $j$', fontsize=6.6, color=GREY, ha='left', va='center')
    ax.text(9.97, y0 - 0.2, r'$\partial\mathcal{L}/\partial A_{f,ij}\ \propto\ x_{ij}\,[\nabla\phi_\psi]_j$' + '\nfeature-specific gradient', fontsize=6.6, color=RED, ha='center', va='top', linespacing=1.3)
    ax.text(13.17, y0 - 0.2, 'used for training only,\nthen discarded', fontsize=6.6, color=RED, ha='center', va='top', linespacing=1.3)
    ax.text(0.4, 5.6, 'Row-mean design (AAGB-RM): the gate ' + r'$g_i=\frac{1}{d}\sum_j A_{f,ij}$' + ' multiplies the latent vector, so all feature weights of a sample receive the same gradient.',
            fontsize=6.4, color=GREY, va='center')

    # ---- Stage 2
    ax.add_patch(FancyBboxPatch((0.15, 0.15), 14.5, 4.75, boxstyle='round,pad=0.0,rounding_size=0.12', ec=GREEN, fc='#f3f8f3', lw=1.0))
    ax.text(0.4, 4.57, 'Stage 2: attention-guided augmentation and XGBoost', fontsize=8.0, color=GREEN, fontweight='bold', va='center')
    box(ax, 0.4, 1.95, 3.3, 1.6, 'training rows: out-of-fold\nattention (K = 5)\ntest rows: module fitted on\nall training rows', ec=BLUE)
    box(ax, 4.35, 3.05, 3.2, 0.95, r'$X_w=X\odot A_f\odot(\mathbf{a}_s\mathbf{1}_d^{\top})$')
    box(ax, 4.35, 1.8, 3.2, 1.0, r'$X_{int}$: products of all pairs among' + '\n' + r'the top-$k$ features by mean attention')
    box(ax, 4.35, 0.6, 3.2, 0.95, r'original features $X$')
    box(ax, 8.25, 1.75, 2.75, 1.25, r'$X_{aug}=[\,X\ \|\ X_w\ \|\ X_{int}\,]$' + '\n' + r'$2d+\binom{k}{2}$ columns', lw=1.5)
    box(ax, 11.6, 1.75, 2.8, 1.25, 'XGBoost\nnested TPE tuning', ec=GREEN, bold=True, lw=1.5)
    arrow(ax, (3.7, 3.0), (4.35, 3.5), color=BLUE)
    arrow(ax, (3.7, 2.55), (4.35, 2.3), color=BLUE)
    arrow(ax, (7.55, 3.5), (8.25, 2.75))
    arrow(ax, (7.55, 2.3), (8.25, 2.38))
    arrow(ax, (7.55, 1.08), (8.25, 2.0))
    arrow(ax, (11.0, 2.38), (11.6, 2.38))
    ax.text(9.62, 1.5, 'TreeSHAP, aggregated\nto source variables', fontsize=6.6, color=INK, ha='center', va='top', linespacing=1.3)
    ax.text(13.0, 1.5, 'risk estimate; threshold from\ninner out-of-fold predictions', fontsize=6.6, color=INK, ha='center', va='top', linespacing=1.3)
    arrow(ax, (2.05, 5.25), (2.05, 3.55), color=BLUE, lw=1.3)
    ax.text(2.25, 4.15, r'$A_f,\ \mathbf{a}_s$', fontsize=7.0, color=BLUE, va='center')
    fig.savefig(out, dpi=300, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'fig1.png')
