import json
from pycocotools.coco import COCO
from os import path as osp

data = json.load(open('/nobackup-slow/dataset/my_xfdu/video/nuscene/nuimages_v1.0-val.json'))


new_annos = []
remove_image_id = []
# breakpoint()
for anno in data['annotations']:
    if anno['category_id'] not in [8, 9, 10, 11, 12, 13, 14, 15]:
        remove_image_id.append(anno['image_id'])
        continue
    else:
        # anno['category_id'] = MAPPING[anno['category_id']]
        new_annos.append(anno)
# import numpy as np
all_image_id = range(0, len(data['images']))
# breakpoint()
kept_image_id = set(all_image_id).difference(set(remove_image_id))

kept_images = []
for index in range(len(data['images'])):
    if index in kept_image_id:
        kept_images.append(data['images'][index])


new_labels = {
    'categories': data['categories'],
    'images': kept_images,#data['images'],
    'annotations': new_annos
}

save_path = '/nobackup-slow/dataset/my_xfdu/video/nuscene/'
prefix = 'nu_ood.json'

save_path = osp.join(save_path, prefix)
with open(save_path, 'w') as fp:
    json.dump(new_labels, fp)


