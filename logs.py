import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

files = {
    'RPPO':        'runs/rppo_v14_repro/eval_logs/evaluations.npz',
    'PPO ($k=2$)': 'runs/ppo_hl2/eval_logs/evaluations.npz',
    'SAC ($k=4$)': 'runs/sac_hl4_repro/eval_logs/evaluations.npz',
}
colors = {
    'RPPO':        '#2c7fb8',
    'PPO ($k=2$)': '#d95f02',
    'SAC ($k=4$)': '#2ca25f',
}

plt.rcParams.update({
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.family': 'serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, smooth in zip(axes, [False, True]):
    for name, path in files.items():
        data = np.load(path)
        ts   = data['timesteps'] / 1e6
        res  = data['results']
        mean = res.mean(axis=1)
        std  = res.std(axis=1)
        if smooth:
            mean_plot = uniform_filter1d(mean, size=5)
            std_plot  = uniform_filter1d(std,  size=5)
        else:
            mean_plot, std_plot = mean, std
        c = colors[name]
        ax.plot(ts, mean_plot, label=name, color=c, linewidth=1.8)
        ax.fill_between(ts,
                        mean_plot - std_plot,
                        mean_plot + std_plot,
                        alpha=0.15, color=c)

    ax.set_xlabel(r'Pasos de entrenamiento [$\times 10^6$]', fontsize=11)
    ax.set_ylabel('Recompensa media de evaluación', fontsize=11)
    ax.set_title('Suavizado (ventana 5)' if smooth else 'Sin suavizar',
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1.5)
    ax.axhline(0, color='gray', linestyle=':', linewidth=0.8)

fig.suptitle('Curvas de aprendizaje — evaluación periódica (5 episodios)',
             fontsize=12)
plt.tight_layout()
plt.savefig('outputs/curvas_aprendizaje.pdf',
            bbox_inches='tight', format='pdf')
plt.savefig('outputs/curvas_aprendizaje.png',
            dpi=150, bbox_inches='tight')
print("OK")