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
frames1 = [1,2,3,4,5,6]
single_coco = [81.76,83.11, 83.29,82.76,81.84,80.43]#[80.88, 81.76,83.11, 83.29,82.76,81.84,80.43]
single_nu = [ 73.47,74.04,74.34,73.03,71.03,70.10]#[71.90, 73.47,74.04,74.34,73.03,71.03,70.10]

multi_coco = [83.57,84.48,85.06,84.99,83.30,82.36]
multi_nu = [72.65,73.94,74.47,73.56,72.07,72.73]
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
figure, axes = plt.subplots(1, 2, sharex=True, figsize=(7,3.5))
# figure.suptitle('Geeksforgeeks')
# breakpoint()




# axes[0][0].set_title(r'$T$=1, OOD=COCO')
# # df=pd.DataFrame(dict(x=range(5),y=[3,15,9,12,4]))
# data_preproc = pd.DataFrame({
#     'Frame range': frames,
#     'AUROC': single_coco})
# sub1 = sns.barplot(data=data_preproc,x='Frame range',y='AUROC', ax=axes[0][0], palette=sns.color_palette('Blues_r',7))
# sub1.set(ylim=(80,84))
# # axes[0][0].set_box_aspect(10/len(axes[0][0].patches))
# # sns.linplot(data=df,x='Frame_r')
# widthbars = [1,1,1,1,1,1,1]
# for bar, newwidth in zip(axes[0][0].patches, widthbars):
#     x = bar.get_x()
#     width = bar.get_width()
#     print(x)
#     centre = x #+ width/2.
#     bar.set_x(centre)
#     bar.set_width(newwidth)
#
#
# axes[0][1].set_title(r'$T$=1, OOD=NuImages')
# data_preproc = pd.DataFrame({
#     'Frame range': frames,
#     'AUROC': single_nu})
# sub2 = sns.barplot(data=data_preproc,x='Frame range',y='AUROC', ax=axes[0][1], palette="magma")
# # sub2.set(xticks=[0, 5, 10, 15])
# sub2.set(ylim=(69,75))
# axes[0][1].set_ylabel("")
# widthbars = [1,1,1,1,1,1,1]
# for bar, newwidth in zip(axes[0][1].patches, widthbars):
#     x = bar.get_x()
#     width = bar.get_width()
#     print(x)
#     centre = x #+ width/2.
#     bar.set_x(centre)
#     bar.set_width(newwidth)



axes[0].set_title(r'$T$=3, OOD=COCO')
data_preproc = pd.DataFrame({
    'Frame interval': frames1,
    'AUROC': multi_coco})
sub3 = sns.barplot(data=data_preproc,x='Frame interval',y='AUROC', ax=axes[0], palette=sns.color_palette('Blues_r',7))
# sub3.set(xticks=[0, 5, 10, 15], yticks= [83,84,85,86])
sub3.set(ylim=(82,86))

widthbars = [1,1,1,1,1,1]
for bar, newwidth in zip(axes[0].patches, widthbars):
    x = bar.get_x()
    width = bar.get_width()
    print(x)
    centre = x #+ width/2.
    bar.set_x(centre)
    bar.set_width(newwidth)


axes[1].set_title(r'$T$=3, OOD=NuImages')
data_preproc = pd.DataFrame({
    'Frame interval': frames1,
    'AUROC': multi_nu})
sub4 = sns.barplot(data=data_preproc,x='Frame interval',y='AUROC', ax=axes[1], palette="magma")
# sub4.set(xticks=[0, 5, 10, 15], yticks= [74,75])
sub4.set(ylim=(71,75))
axes[1].set_ylabel("")
widthbars = [1,1,1,1,1,1]
for bar, newwidth in zip(axes[1].patches, widthbars):
    x = bar.get_x()
    width = bar.get_width()
    print(x)
    centre = x #+ width/2.
    bar.set_x(centre)
    bar.set_width(newwidth)
figure.tight_layout(w_pad=1)
figure.savefig('ablation1.pdf')