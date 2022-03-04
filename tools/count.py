import os
import numpy as np

root_directory = '/nobackup-slow/dataset/my_xfdu/video/vis/train/JPEGImages/'#72
# root_directory = '/nobackup-slow/dataset/my_xfdu/video/bdd/bdd100k/images/track/train/'#263


numbers = []
for video in list(os.listdir(root_directory)):
    path = os.path.join(root_directory, video)
    cur_frame = os.listdir(path)
    numbers.append(len(list(cur_frame)))

numbers = np.asarray(numbers)
print(np.min(numbers))
print(np.max(numbers))