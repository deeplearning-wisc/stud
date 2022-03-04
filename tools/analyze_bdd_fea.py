import umap
import numpy as np
import matplotlib.pyplot as plt
import torch


pos_fea = np.load('/afs/cs.wisc.edu/u/x/f/xfdu/workspace/video/cycle-confusion/bdd_pos_single_frame_1000.npy', allow_pickle=True)
neg_fea = np.load('/afs/cs.wisc.edu/u/x/f/xfdu/workspace/video/cycle-confusion/bdd_neg_single_frame_1000.npy', allow_pickle=True)

index = 0
for fea in pos_fea:
    if index == 0:
        pos_np = fea
        index += 1
    else:
        pos_np = np.concatenate([pos_np, fea], 0)

index = 0
for fea in neg_fea:
    if index == 0:
        neg_np = fea
        index += 1
    else:
        neg_np = np.concatenate([neg_np, fea], 0)
fea_np = np.concatenate([pos_np, neg_np], 0)
print(len(fea_np))
# breakpoint()
reducer = umap.UMAP(random_state=42, n_neighbors=30, min_dist=0.6, n_components=2, metric='euclidean')
embedding = reducer.fit_transform(fea_np)

fig, ax = plt.subplots(figsize=(6, 6))
def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

classes = [str(hhh) for hhh in range(2)]
# color = targets.astype(int)#[index for index in range(20)]#
color = get_cmap(2)
# color = plt.cm.coolwarm(np.linspace(0.1,0.9,11))

index = 0
for i in range(0, 2):
    if i == 0:
        plt.scatter(embedding[:, 0][1000 * i:1000 * i + 1000],
                    embedding[:, 1][1000 * i:1000 * i + 1000],
                    c='r',
                    label=index, cmap="Spectral", s=1)
    else:
        plt.scatter(embedding[:, 0][1000 * i:1000 * i + 1000],
                    embedding[:, 1][1000 * i:1000 * i + 1000],
                    c='b',
                    label=index, cmap="Spectral", s=1)
    index += 1

plt.legend(fontsize=20)
# ax.legend(markerscale=9)
ax.legend(loc='lower left',markerscale=9)#, bbox_to_anchor=(1, 0.5)
# plt.legend(handles=scatter.legend_elements()[0], labels=classes)
# breakpoint()
plt.setp(ax, xticks=[], yticks=[])
# plt.title("With virtual outliers", fontsize=20)
# plt.savefig('./voc_coco_umap_visual_ours.jpg', dpi=250)
# plt.title("Vanilla detector", fontsize=20)
plt.savefig('./bdd_ana_single_frame_1000.jpg', dpi=250)
# plt.show()