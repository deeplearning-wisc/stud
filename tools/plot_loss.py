import matplotlib.pyplot as plt
import json
import numpy as np
import matplotlib
import matplotlib as mpl

mpl.rcParams['axes.linewidth'] = 2

# matplotlib.rcParams['mathtext.fontset'] = 'Arial'
matplotlib.rcParams['mathtext.rm'] = 'Arial'
matplotlib.rcParams['mathtext.it'] = 'Arial'

# matplotlib.rcParams['mathtext.fontset'] = 'stix'
# matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')
data  =open('/nobackup/dataset/my_xfdu/video/vis/checkpoints/VIS/energy_no_original_loss_direct_add_0_02_frame_9_revise_4to6_multi_random_seed1/metrics.json','r')
tweets = []
for line in data:
    tweets.append(json.loads(line))
data= tweets
epochs = []
losses = []
for epoch, loss in enumerate(data):
    epochs.append(epoch)
    losses.append(loss['ene_reg_loss']*20)

# plt.figure(figsize=(10,5))
# ax.set_title('Sine and cosine waves')


plt.figure(figsize=(10,8))
# plt.title("Training and Validation Loss")
# plt.plot(val_losses,label="val")
# plt.plot(train_losses,label="train")
x = [i*20 for i in range(len(losses))]
plt.plot(x,losses, label=r'$\mathcal{L}_{\mathrm{uncertainty}}$',color='#184E77',linewidth=3)
plt.xlabel("iterations", fontsize=25)
plt.ylabel("Uncertainty loss", fontsize=25)
plt.xticks(fontsize= 25)
plt.yticks(fontsize= 25)
plt.legend(fontsize=30, frameon=False)
plt.savefig('./loss.jpg', dpi=500)
