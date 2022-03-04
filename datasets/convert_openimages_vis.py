import json

data = json.load(open('/nobackup-slow/dataset/my_xfdu/OpenImages/coco_classes/COCO-Format/val_coco_format.json','rb'))
all_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane',
                          'bus', 'train', 'truck', 'boat', 'traffic light',
                          'fire hydrant', 'stop sign', 'parking meter', 'bench',
                          'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
                          'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
                          'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                          'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                          'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                          'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                          'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
                          'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                          'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                          'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                          'hair drier', 'toothbrush']
all_dict = {}
for i in range(len(all_classes)):
    all_dict[all_classes[i]] = i + 1

not_vis_classes = ['bicycle',
                          'bus', 'traffic light',
                          'fire hydrant', 'stop sign', 'parking meter', 'bench',
                       'sheep',
                            'backpack', 'umbrella', 'handbag',
                          'tie', 'suitcase', 'skis', 'sports ball',
                          'kite', 'baseball bat', 'baseball glove'
                          , 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                          'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                          'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
                          'bed', 'dining table', 'toilet', 'tv', 'laptop', 'remote',
                          'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                          'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                          'hair drier', 'toothbrush']
not_vis_id = []
for item in not_vis_classes:
    not_vis_id.append(all_dict[item])
remove_image_id = []
# breakpoint()
for annotation in data['annotations']:
    if annotation['category_id'] not in not_vis_id:
        remove_image_id.append(annotation['image_id'])
remove_image_id = list(set(remove_image_id))
new_annotation = []
new_image_id = []
for annotation in data['annotations']:
    if annotation['image_id'] not in remove_image_id:
        new_annotation.append(annotation)
for image in data['images']:
    if image['id'] not in remove_image_id:
        new_image_id.append(image)
# breakpoint()
new_annotation_all = data
new_annotation_all['annotations'] = new_annotation
new_annotation_all['images'] = new_image_id
breakpoint()
json.dump(new_annotation_all, open('/nobackup-slow/dataset/my_xfdu/OpenImages/coco_classes/COCO-Format/vis_open_ood.json','w'))


