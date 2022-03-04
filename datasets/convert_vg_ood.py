import json
from pycocotools.coco import COCO
from os import path as osp

data = json.load(open('/nobackup-slow/dataset/my_xfdu/video/vg/anno/visual_genome_val.json'))


new_annos = []
remove_image_id = []
# breakpoint()
for anno in data['annotations']:
    if anno['category_id'] in [131, 488, 110,130,146,218,343,646,180,999,58,157,233,52,625,685,954,1181,1478,53,
                                   184,97,150,744,117,337,341,351,83,141,992,1509,444,35,37,470,42,186,1388,639,127,
                                   9,364,19,86,297,1223,138,258,135,350,59,68,70,566,814,898,1181,1447,155,810,838,
                                   85,87]:
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

save_path = '/nobackup-slow/dataset/my_xfdu/video/vg/anno'
prefix = 'vg_ood.json'
# breakpoint()
save_path = osp.join(save_path, prefix)
with open(save_path, 'w') as fp:
    json.dump(new_labels, fp)


