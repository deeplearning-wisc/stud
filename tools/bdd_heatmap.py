import numpy as np
import seaborn as sns; sns.set_theme()
idx_data = np.asarray(np.load('./bdd_offset.npy', allow_pickle=True))
score_data = np.load('./bdd_score_visual.npy', allow_pickle=True)
breakpoint()
ax = sns.heatmap(score_data[6].cpu().data.numpy())
ax.savefig('./bdd_heatmap.jpg')