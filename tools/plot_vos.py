import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt




# # Create a dataset with many short random walks
# rs = np.random.RandomState(4)
# # pos = rs.randint(-1, 2, (20, 5)).cumsum(axis=1)
# # breakpoint()
# # pos -= pos[:, 0, np.newaxis]
# pos = np.asarray([[80.88, 81.76,83.11, 83.29,82.76,81.84,80.43],
#                   [71.90, 73.47,74.04,74.34,73.03,71.03,70.10]])
# step = np.asarray([1,3,5,7,9,11,13,1,3,5,7,9,11,13])
# walk = np.repeat(['COCO','NuImages'], 7)
# walk1 =  np.repeat(range(2), 7)
# # breakpoint()
# df = pd.DataFrame(np.c_[pos.flat, step, walk, walk1],
#                   columns=["AUROC", "Frame range", "OOD", "dummy"])
#
# # Initialize a grid of plots with an Axes for each walk
# grid = sns.FacetGrid(df, col='OOD' , hue='dummy', palette="tab20c",
#                      col_wrap=4, height=2.2)
#
# # Draw a horizontal line to show the starting point
# # grid.refline(y=0, linestyle=":")
#
# # Draw a line plot to show the trajectory of each random walk
# grid.map(plt.plot, "Frame range", "AUROC", marker="o")
#
# # Adjust the tick positions and labels
# # grid.set(xticks=np.arange(16), yticks=[70, 90],
# #          xlim=(0,15), ylim=(70,90))
#
#
#
#
#
# # num_rows = 4
# # years = frames
# # data_preproc = pd.DataFrame({
# #     'Frame range': years,
# #     r'$T$=1, OOD=COCO': single_coco,
# #     r'$T$=1, OOD=NuImages': single_nu})
# #     # r'$T$=3, OOD=COCO': multi_coco,
# #     # r'$T$=3, OOD=NuImages': multi_nu})
# # fig = sns.lineplot(x='Frame range', y='value', hue='variable',
# #              data=pd.melt(data_preproc, ['Frame range']), marker="o")
#
# # Adjust the arrangement of the plots
# # fig.tight_layout(w_pad=1)
def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

color = get_cmap(7)
frames = [3,5,7,9,11,13]
frames1 = [1,2,3,4,5,6,8,10]
quesize = [0.5,1,2,4,6,8,10,12]
weight=[0.01,0.05,0.1,0.15,0.5]
start = [2,4,6,8,10,12,14,16]
single_coco = [83.41, 82.31,82.20,80.86,79.53,78.66,78.06,74.33]#[80.88, 81.76,83.11, 83.29,82.76,81.84,80.43]
single_coco1 = [77.04,77.68,79.96,80.20,81.99,82.26,83.41,83.07]
single_coco2 = [82.64,83.27,83.41,77.47,57.46]
single_coco3 =[78.55,79.85,79.97,80.64,83.22,83.41,82.70,81.37]



import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Setting seaborn as default style even
# if use only matplotlib
# sns.set()
sns.set(font_scale = 1.5)
sns.set_theme(style="ticks")
# figure, axes = plt.subplots()
# figure.suptitle('Geeksforgeeks - one axes with no data')
# plt.bar(data.xcol,data.ycol,4)
figure, axes = plt.subplots(1, 1, sharex=True, figsize=(3.5,3.5))
# figure.suptitle('Geeksforgeeks')
# breakpoint()


def show_values(axs, orient="v", space=.01):
    def _single(ax):
        if orient == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height() + (p.get_height()*0.001)
                value = '{:.1f}'.format(p.get_height())
                ax.text(_x, _y, value, ha="center", fontsize=9)
        elif orient == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height() - (p.get_height()*0.1)
                value = '{:.1f}'.format(p.get_width())
                ax.text(_x, _y, value, ha="left", fontsize=10)

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _single(ax)
    else:
        _single(axs)





# axes[0].set_title(r'Selected outliers')
# breakpoint()
# data_preproc = pd.DataFrame({
#     '(a) Selected outliers': frames1,
#     'AUROC': single_coco})
# sub3 = sns.barplot(data=data_preproc,x='(a) Selected outliers',y='AUROC', ax=axes, palette=sns.color_palette('Blues_r',8))
# # sub3.set(xticks=[0, 5, 10, 15], yticks= [83,84,85,86])
# sub3.set(ylim=(73,84))
#
# widthbars = [1,1,1,1,1,1,1,1]
# for bar, newwidth in zip(axes.patches, widthbars):
#     x = bar.get_x()
#     width = bar.get_width()
#     print(x)
#     centre = x #+ width/2.
#     bar.set_x(centre)
#     bar.set_width(newwidth)
# show_values(sub3)

# axes[1].set_title(r'$T$=3, OOD=NuImages')
# data_preproc = pd.DataFrame({
#     '(b) ID queue size': quesize,
#     'AUROC': single_coco1})
# sub4 = sns.barplot(data=data_preproc,x='(b) ID queue size',y='AUROC', ax=axes, palette="magma")
# # sub4.set(xticks=[0, 5, 10, 15], yticks= [74,75])
# sub4.set(ylim=(76,84))
# # axes.set_ylabel("")
# widthbars = [1,1,1,1,1,1,1,1]
# for bar, newwidth in zip(axes.patches, widthbars):
#     x = bar.get_x()
#     width = bar.get_width()
#     print(x)
#     centre = x #+ width/2.
#     bar.set_x(centre)
#     bar.set_width(newwidth)
# show_values(sub4)
#
#
# #
# data_preproc = pd.DataFrame({
#     '(c) Regularization weight': weight,
#     'AUROC': single_coco2})
# sub4 = sns.barplot(data=data_preproc,x='(c) Regularization weight',y='AUROC', ax=axes, palette="viridis")
# # sub4.set(xticks=[0, 5, 10, 15], yticks= [74,75])
# sub4.set(ylim=(56,85))
# # axes.set_ylabel("")
# widthbars = [1,1,1,1,1]
# for bar, newwidth in zip(axes.patches, widthbars):
#     x = bar.get_x()
#     width = bar.get_width()
#     print(x)
#     centre = x #+ width/2.
#     bar.set_x(centre)
#     bar.set_width(newwidth)
# show_values(sub4)

data_preproc = pd.DataFrame({
    '(d) Starting iteration': start,
    'AUROC': single_coco3})
sub4 = sns.barplot(data=data_preproc,x='(d) Starting iteration',y='AUROC', ax=axes, palette="dark:salmon")
# sub4.set(xticks=[0, 5, 10, 15], yticks= [74,75])
sub4.set(ylim=(78,84))
# axes.set_ylabel("")
widthbars = [1,1,1,1,1,1,1,1]
for bar, newwidth in zip(axes.patches, widthbars):
    x = bar.get_x()
    width = bar.get_width()
    print(x)
    centre = x #+ width/2.
    bar.set_x(centre)
    bar.set_width(newwidth)
show_values(sub4)
figure.tight_layout(w_pad=1)
figure.savefig('ablation4_vos.pdf')