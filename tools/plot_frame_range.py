import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# def get_cmap(n, name='hsv'):
#     '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
#     RGB color; the keyword argument name must be a standard mpl colormap name.'''
#     return plt.cm.get_cmap(name, n)
#
# color = get_cmap(7)
frames = [1,2,3,4,5]
frames1 = [3,5,7,9,11,13,'inf']
single_coco = [81.76,83.11, 83.29,82.76,81.84,80.43]#[80.88, 81.76,83.11, 83.29,82.76,81.84,80.43]
single_nu = [ 73.47,74.04,74.34,73.03,71.03,70.10]#[71.90, 73.47,74.04,74.34,73.03,71.03,70.10]

multi_coco = [83.34, 84.26,84.70,85.67,85.34,84.41, 80.35]
multi_nu = [73.89, 75.61,75.64,75.67,74.87,74.42, 71.80]


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



# axes[0].set_title(r'$T$=3, OOD=COCO')
# data_preproc = pd.DataFrame({
#     'Frame range': frames1,
#     'AUROC': multi_coco})
# sub3 = sns.barplot(data=data_preproc,x='Frame range',y='AUROC', ax=axes[0], palette="dark:salmon_r")
# # sub3.set(xticks=[0, 5, 10, 15], yticks= [83,84,85,86])
# sub3.set(ylim=(80,86))
#
# widthbars = [1,1,1,1,1,1, 1]
# for bar, newwidth in zip(axes[0].patches, widthbars):
#     x = bar.get_x()
#     width = bar.get_width()
#     print(x)
#     centre = x #+ width/2.
#     bar.set_x(centre)
#     bar.set_width(newwidth)
#
#
# axes[1].set_title(r'$T$=3, OOD=NuImages')
# data_preproc = pd.DataFrame({
#     'Frame range': frames1,
#     'AUROC': multi_nu})
# sub4 = sns.barplot(data=data_preproc,x='Frame range',y='AUROC', ax=axes[1], palette="YlOrBr")
# # sub4.set(xticks=[0, 5, 10, 15], yticks= [74,75])
# sub4.set(ylim=(71,76))
# axes[1].set_ylabel("")
# widthbars = [1,1,1,1,1,1,1]
# for bar, newwidth in zip(axes[1].patches, widthbars):
#     x = bar.get_x()
#     width = bar.get_width()
#     print(x)
#     centre = x #+ width/2.
#     bar.set_x(centre)
#     bar.set_width(newwidth)




multi_coco = [80.43,82.71,85.67,81.41,80.81]
multi_nu = [70.10,75.29,75.67,73.26,72.76]
axes[0].set_title(r'$T$=3, OOD=COCO')
data_preproc = pd.DataFrame({
    'Number of Frame': frames,
    'AUROC': multi_coco})
sub1 = sns.barplot(data=data_preproc,x='Number of Frame',y='AUROC', ax=axes[0], palette=sns.color_palette('Blues_r',7))
# sub3.set(xticks=[0, 5, 10, 15], yticks= [83,84,85,86])
sub1.set(ylim=(80,86))
axes[0].set_ylabel("")
widthbars = [1,1,1,1,1]
for bar, newwidth in zip(axes[0].patches, widthbars):
    x = bar.get_x()
    width = bar.get_width()
    print(x)
    centre = x #+ width/2.
    bar.set_x(centre)
    bar.set_width(newwidth)


axes[1].set_title(r'$T$=3, OOD=NuImages')
data_preproc = pd.DataFrame({
    'Number of Frame': frames,
    'AUROC': multi_nu})
sub2 = sns.barplot(data=data_preproc,x='Number of Frame',y='AUROC', ax=axes[1], palette="magma")
# sub4.set(xticks=[0, 5, 10, 15], yticks= [74,75])
sub2.set(ylim=(69,76))
axes[1].set_ylabel("")
widthbars = [1,1,1,1,1]
for bar, newwidth in zip(axes[1].patches, widthbars):
    x = bar.get_x()
    width = bar.get_width()
    print(x)
    centre = x #+ width/2.
    bar.set_x(centre)
    bar.set_width(newwidth)


#
# multi_coco = [83.57,84.48,85.06,84.99,83.30,82.36]
# multi_nu = [72.65,73.94,74.47,73.56,72.07,72.73]
# axes[0].set_title(r'$T$=3, OOD=COCO')
# data_preproc = pd.DataFrame({
#     'Frame interval': frames,
#     'AUROC': multi_coco})
# sub1 = sns.barplot(data=data_preproc,x='Frame interval',y='AUROC', ax=axes[0], palette=sns.color_palette('Blues_r',7))
# # sub3.set(xticks=[0, 5, 10, 15], yticks= [83,84,85,86])
# sub1.set(ylim=(82,86))
# axes[0].set_ylabel("")
# widthbars = [1,1,1,1,1,1]
# for bar, newwidth in zip(axes[0].patches, widthbars):
#     x = bar.get_x()
#     width = bar.get_width()
#     print(x)
#     centre = x #+ width/2.
#     bar.set_x(centre)
#     bar.set_width(newwidth)
#
#
# axes[1].set_title(r'$T$=3, OOD=NuImages')
# data_preproc = pd.DataFrame({
#     'Frame interval': frames,
#     'AUROC': multi_nu})
# sub2 = sns.barplot(data=data_preproc,x='Frame interval',y='AUROC', ax=axes[1], palette="magma")
# # sub4.set(xticks=[0, 5, 10, 15], yticks= [74,75])
# sub2.set(ylim=(71,75))
# axes[1].set_ylabel("")
# widthbars = [1,1,1,1,1,1]
# for bar, newwidth in zip(axes[1].patches, widthbars):
#     x = bar.get_x()
#     width = bar.get_width()
#     print(x)
#     centre = x #+ width/2.
#     bar.set_x(centre)
#     bar.set_width(newwidth)








figure.tight_layout(w_pad=1)
figure.savefig('ablation2.pdf')