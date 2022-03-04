import json
import numpy as np

video = json.load(open('/nobackup-slow/dataset/my_xfdu/video/vis/train/instances.json'))
all_video_id = list(range(1, len(video['videos']) + 1))
num_train_video = int(len(all_video_id) * 0.8)

all_categories = [{'supercategory': 'object', 'id': 1, 'name': 'airplane'}, {'supercategory': 'object', 'id': 2, 'name': 'bear'},
                  {'supercategory': 'object', 'id': 3, 'name': 'bird'}, {'supercategory': 'object', 'id': 4, 'name': 'boat'},
                  {'supercategory': 'object', 'id': 5, 'name': 'car'}, {'supercategory': 'object', 'id': 6, 'name': 'cat'},
                  {'supercategory': 'object', 'id': 7, 'name': 'cow'}, {'supercategory': 'object', 'id': 8, 'name': 'deer'},
                  {'supercategory': 'object', 'id': 9, 'name': 'dog'}, {'supercategory': 'object', 'id': 10, 'name': 'duck'},
                  {'supercategory': 'object', 'id': 11, 'name': 'earless_seal'}, {'supercategory': 'object', 'id': 12, 'name': 'elephant'},
                  {'supercategory': 'object', 'id': 13, 'name': 'fish'}, {'supercategory': 'object', 'id': 14, 'name': 'flying_disc'},
                  {'supercategory': 'object', 'id': 15, 'name': 'fox'}, {'supercategory': 'object', 'id': 16, 'name': 'frog'},
                  {'supercategory': 'object', 'id': 17, 'name': 'giant_panda'}, {'supercategory': 'object', 'id': 18, 'name': 'giraffe'},
                  {'supercategory': 'object', 'id': 19, 'name': 'horse'}, {'supercategory': 'object', 'id': 20, 'name': 'leopard'},
                  {'supercategory': 'object', 'id': 21, 'name': 'lizard'}, {'supercategory': 'object', 'id': 22, 'name': 'monkey'},
                  {'supercategory': 'object', 'id': 23, 'name': 'motorbike'}, {'supercategory': 'object', 'id': 24, 'name': 'mouse'},
                  {'supercategory': 'object', 'id': 25, 'name': 'parrot'}, {'supercategory': 'object', 'id': 26, 'name': 'person'},
                  {'supercategory': 'object', 'id': 27, 'name': 'rabbit'}, {'supercategory': 'object', 'id': 28, 'name': 'shark'},
                  {'supercategory': 'object', 'id': 29, 'name': 'skateboard'}, {'supercategory': 'object', 'id': 30, 'name': 'snake'},
                  {'supercategory': 'object', 'id': 31, 'name': 'snowboard'}, {'supercategory': 'object', 'id': 32, 'name': 'squirrel'},
                  {'supercategory': 'object', 'id': 33, 'name': 'surfboard'}, {'supercategory': 'object', 'id': 34, 'name': 'tennis_racket'},
                  {'supercategory': 'object', 'id': 35, 'name': 'tiger'}, {'supercategory': 'object', 'id': 36, 'name': 'train'},
                  {'supercategory': 'object', 'id': 37, 'name': 'truck'}, {'supercategory': 'object', 'id': 38, 'name': 'turtle'},
                  {'supercategory': 'object', 'id': 39, 'name': 'whale'}, {'supercategory': 'object', 'id': 40, 'name': 'zebra'}]
removed_class_id = [3,4,7,10,11,13,14,15,16,20,21,22,24,25,28,31,32,33,34,36,38,39,40]
tranformation_dict = {1:1,2:2,5:3,6:4,8:5,9:6,12:7,17:8,18:9,19:10,23:11,26:12,27:13,
                      29:14,30:15,35:16,37:17}
kept_categories = [{'supercategory': 'object', 'id': 1, 'name': 'airplane'}, {'supercategory': 'object', 'id': 2, 'name': 'bear'},

                  {'supercategory': 'object', 'id': 3, 'name': 'car'}, {'supercategory': 'object', 'id': 4, 'name': 'cat'},
 {'supercategory': 'object', 'id': 5, 'name': 'deer'},
                  {'supercategory': 'object', 'id': 6, 'name': 'dog'},
                  {'supercategory': 'object', 'id': 7, 'name': 'elephant'},

                  {'supercategory': 'object', 'id': 8, 'name': 'giant_panda'}, {'supercategory': 'object', 'id': 9, 'name': 'giraffe'},
                  {'supercategory': 'object', 'id': 10, 'name': 'horse'},

                  {'supercategory': 'object', 'id': 11, 'name': 'motorbike'},  {'supercategory': 'object', 'id': 12, 'name': 'person'},
                  {'supercategory': 'object', 'id': 13, 'name': 'rabbit'},
                  {'supercategory': 'object', 'id': 14, 'name': 'skateboard'}, {'supercategory': 'object', 'id': 15, 'name': 'snake'},

                  {'supercategory': 'object', 'id': 16, 'name': 'tiger'},
                  {'supercategory': 'object', 'id': 17, 'name': 'truck'}, ]

video_info_train = []
video_info_val = []
images_train = []
images_val = []
id_train = 1
id_val = 1
id_image_train = 1
id_image_val = 1
# breakpoint()
for info in video['videos']:
    if id_train <= num_train_video:
        video_info_train.append({'id': id_train, 'name': info['file_names'][0].split('/')[0]})
        for index in range(len(info['file_names'])):
            image_info = {'file_name': info['file_names'][index],
                          'height': info['height'], 'width': info['width'], 'id': id_image_train, 'video_id': id_train, 'frame_id': index}
            images_train.append(image_info)
            id_image_train += 1
        id_train += 1
    else:
        video_info_val.append({'id': id_val, 'name': info['file_names'][0].split('/')[0]})
        for index in range(len(info['file_names'])):
            image_info = {'file_name': info['file_names'][index],
                          'height': info['height'], 'width': info['width'], 'id': id_image_val, 'video_id': id_val, 'frame_id': index}
            images_val.append(image_info)
            id_image_val += 1
        id_val += 1
'''
>>> data['images'][101]
{'file_name': 'b1c66a42-6f7d68ca/b1c66a42-6f7d68ca-0000102.jpg', 
'height': 720, 'width': 1280, 'id': 102, 'video_id': 1, 'frame_id': 101}
'''

# breakpoint()
annotation_train = []
annotation_val = []
id_train = 1
id_train_image = 1
id_val = 1
id_val_image = 1
pre_video_id = -1 #video['annotations'][0]['video_id']
for index1 in range(len(video['annotations'])):
    if video['annotations'][index1]['video_id'] <= num_train_video:
        if pre_video_id != video['annotations'][index1]['video_id']:
            cur_video_ann = {}
            for index in range(len(video['annotations'][index1]['bboxes'])):
                cur_video_ann[index] = []

        for index in range(len(video['annotations'][index1]['bboxes'])):
            if video['annotations'][index1]['category_id'] not in removed_class_id:
                cur_video_ann[index].append([video['annotations'][index1]['bboxes'][index],
                                             video['annotations'][index1]['areas'][index],
                                             video['annotations'][index1]['iscrowd'],
                                             tranformation_dict[video['annotations'][index1]['category_id']]])
            else:
                cur_video_ann[index].append([None,
                                             video['annotations'][index1]['areas'][index],
                                             video['annotations'][index1]['iscrowd'],
                                             video['annotations'][index1]['category_id']])
            # annotation_train.append({'id':id_train, 'image_id'})
        if index1 == len(video['annotations']) - 1 or video['annotations'][index1]['video_id'] != video['annotations'][index1+1]['video_id']:
            for key in list(cur_video_ann.keys()):
                instance_id = 1
                # print(id_train_image)
                for item in cur_video_ann[key]:
                    if item[0] is None:
                        annotation_train.append({'id': id_train, 'image_id': id_train_image, 'category_id': item[3],
                                                 'instances_id': instance_id,
                                                 'bdd100k_id': 0, 'occluded': False, 'truncated': False,
                                                 'bbox': item[0], 'area': item[1], 'iscrowd': item[2], 'ignore': 1
                                                 })
                    else:
                        annotation_train.append({'id': id_train, 'image_id': id_train_image, 'category_id': item[3],
                                                 'instances_id': instance_id,
                                                 'bdd100k_id': 0, 'occluded': False, 'truncated': False,
                                                 'bbox': item[0], 'area': item[1], 'iscrowd': item[2], 'ignore': 0
                                                 })

                    instance_id += 1
                    id_train += 1
                id_train_image += 1
        pre_video_id = video['annotations'][index1]['video_id']
    else:
        if pre_video_id != video['annotations'][index1]['video_id']:
            cur_video_ann = {}
            for index in range(len(video['annotations'][index1]['bboxes'])):
                cur_video_ann[index] = []

        for index in range(len(video['annotations'][index1]['bboxes'])):
            if video['annotations'][index1]['category_id'] not in removed_class_id:
                cur_video_ann[index].append([video['annotations'][index1]['bboxes'][index],
                                             video['annotations'][index1]['areas'][index],
                                             video['annotations'][index1]['iscrowd'],
                                             tranformation_dict[video['annotations'][index1]['category_id']]])
            else:
                cur_video_ann[index].append([None,
                                             video['annotations'][index1]['areas'][index],
                                             video['annotations'][index1]['iscrowd'],
                                             video['annotations'][index1]['category_id']])
            # annotation_train.append({'id':id_train, 'image_id'})
        if index1 == len(video['annotations']) - 1 or video['annotations'][index1]['video_id'] != video['annotations'][index1+1]['video_id']:
            for key in list(cur_video_ann.keys()):
                instance_id = 1
                for item in cur_video_ann[key]:
                    if item[0] is None:
                        annotation_val.append({'id': id_val, 'image_id': id_val_image, 'category_id': item[3],
                                               'instances_id': instance_id,
                                                 'bdd100k_id': 0, 'occluded': False, 'truncated': False,
                                                 'bbox': item[0], 'area': item[1], 'iscrowd': item[2], 'ignore': 1
                                                 })
                    else:
                        annotation_val.append({'id': id_val, 'image_id': id_val_image, 'category_id': item[3],
                                               'instances_id': instance_id,
                                               'bdd100k_id': 0, 'occluded': False, 'truncated': False,
                                               'bbox': item[0], 'area': item[1], 'iscrowd': item[2], 'ignore': 0
                                               })

                    instance_id += 1
                    id_val += 1
                id_val_image += 1
        pre_video_id = video['annotations'][index1]['video_id']
'''
{'id': 301, 'image_id': 11, 'category_id': 3,
 'instance_id': 25, 'bdd100k_id': '00122086',
 'occluded': True, 'truncated': False,
 'bbox': [664.2417908674067, 367.9733233708366, 36.698191808020056, 28.229378313861503],
 'area': 1035.9671399832512, 'iscrowd': 0, 'ignore': 0,
 'segmentation':
     [[664.2417908674067, 367.9733233708366, 664.2417908674067, 396.2027016846981, 
     700.9399826754268, 396.2027016846981, 700.9399826754268, 367.9733233708366]]}
'''

new_annotation = dict()
new_annotation['categories'] = kept_categories#video['categories']
new_annotation['videos'] = video_info_train
new_annotation['images'] = images_train
new_annotation['annotations'] = annotation_train
json.dump(new_annotation, open('/nobackup-slow/dataset/my_xfdu/video/vis/train/instances_train.json', 'w'))


new_annotation['videos'] = video_info_val
new_annotation['images'] = images_val
new_annotation['annotations'] = annotation_val
json.dump(new_annotation, open('/nobackup-slow/dataset/my_xfdu/video/vis/train/instances_test.json', 'w'))


#
# # postprocessing.
train_data = json.load(open('/nobackup-slow/dataset/my_xfdu/video/vis/train/instances_train.json'))
not_none_image_id = []

for ann in train_data['annotations']:
    if ann['bbox'] is not None:
        not_none_image_id.append(ann['image_id'])
full_image_id = list(range(1, train_data['annotations'][-1]['image_id']+1))
none_image_id_train = list(set(full_image_id).difference(set(not_none_image_id)))
# breakpoint()

test_data = json.load(open('/nobackup-slow/dataset/my_xfdu/video/vis/train/instances_test.json'))
not_none_image_id = []
for ann in test_data['annotations']:
    if ann['bbox'] is not None:
        not_none_image_id.append(ann['image_id'])
full_image_id = list(range(1, test_data['annotations'][-1]['image_id']+1))
none_image_id_test = list(set(full_image_id).difference(set(not_none_image_id)))
# breakpoint()

video_info_train = []
video_info_val = []
images_train = []
images_val = []
id_train = 1
id_val = 1
id_image_train = 1
id_image_val = 1
id_real_image_train = 1
id_real_image_val = 1
for info in video['videos']:
    if id_train <= num_train_video:
        video_info_train.append({'id': id_train, 'name': info['file_names'][0].split('/')[0]})
        temp = 0
        for index in range(len(info['file_names'])):
            if id_image_train not in none_image_id_train:
                image_info = {'file_name': info['file_names'][index],
                              'height': info['height'], 'width': info['width'], 'id': id_real_image_train,
                              'video_id': id_train, 'frame_id': temp}
                temp += 1
                images_train.append(image_info)
                id_real_image_train += 1
            id_image_train += 1
        id_train += 1
    else:
        video_info_val.append({'id': id_val, 'name': info['file_names'][0].split('/')[0]})
        temp = 0
        for index in range(len(info['file_names'])):
            if id_image_val not in none_image_id_test:
                image_info = {'file_name': info['file_names'][index],
                              'height': info['height'], 'width': info['width'], 'id': id_real_image_val,
                              'video_id': id_val, 'frame_id': temp}
                temp += 1
                images_val.append(image_info)
                id_real_image_val += 1
            id_image_val += 1
        id_val += 1
'''
>>> data['images'][101]
{'file_name': 'b1c66a42-6f7d68ca/b1c66a42-6f7d68ca-0000102.jpg', 
'height': 720, 'width': 1280, 'id': 102, 'video_id': 1, 'frame_id': 101}
'''

# breakpoint()
annotation_train = []
annotation_val = []
id_train = 1
id_train_image = 1
id_val = 1
id_val_image = 1
id_real_val_image = 1
id_real_train_image = 1
pre_video_id = -1 #video['annotations'][0]['video_id']
for index1 in range(len(video['annotations'])):
    if video['annotations'][index1]['video_id'] <= num_train_video:
        if pre_video_id != video['annotations'][index1]['video_id']:
            cur_video_ann = {}
            for index in range(len(video['annotations'][index1]['bboxes'])):
                cur_video_ann[index] = []

        for index in range(len(video['annotations'][index1]['bboxes'])):
            if video['annotations'][index1]['category_id'] not in removed_class_id:
                cur_video_ann[index].append([video['annotations'][index1]['bboxes'][index],
                                             video['annotations'][index1]['areas'][index],
                                             video['annotations'][index1]['iscrowd'],
                                             tranformation_dict[video['annotations'][index1]['category_id']]])
            else:
                cur_video_ann[index].append([None,
                                             video['annotations'][index1]['areas'][index],
                                             video['annotations'][index1]['iscrowd'],
                                             video['annotations'][index1]['category_id']])

            # annotation_train.append({'id':id_train, 'image_id'})
        if index1 == len(video['annotations']) - 1 or video['annotations'][index1]['video_id'] != video['annotations'][index1+1]['video_id']:
            for key in list(cur_video_ann.keys()):
                instance_id = 1
                # print(id_train_image)
                for item in cur_video_ann[key]:
                    # assert item[0] is not None
                    if item[0] is not None:
                        annotation_train.append({'id': id_train, 'image_id': id_real_train_image, 'category_id': item[3],
                                                 'instances_id': instance_id,
                                                 'bdd100k_id': 0, 'occluded': False, 'truncated': False,
                                                 'bbox': item[0], 'area': item[1], 'iscrowd': item[2], 'ignore': 0
                                                 })
                        assert item[3] <= 17
                        instance_id += 1
                        id_train += 1
                if id_train_image not in none_image_id_train:
                    id_real_train_image += 1
                id_train_image += 1
        pre_video_id = video['annotations'][index1]['video_id']

    else:
        if pre_video_id != video['annotations'][index1]['video_id']:
            cur_video_ann = {}
            for index in range(len(video['annotations'][index1]['bboxes'])):
                cur_video_ann[index] = []

        for index in range(len(video['annotations'][index1]['bboxes'])):
            if video['annotations'][index1]['category_id'] not in removed_class_id:
                cur_video_ann[index].append([video['annotations'][index1]['bboxes'][index],
                                             video['annotations'][index1]['areas'][index],
                                             video['annotations'][index1]['iscrowd'],
                                             tranformation_dict[video['annotations'][index1]['category_id']]])
            else:
                cur_video_ann[index].append([None,
                                             video['annotations'][index1]['areas'][index],
                                             video['annotations'][index1]['iscrowd'],
                                             video['annotations'][index1]['category_id']])
            # annotation_train.append({'id':id_train, 'image_id'})
        if index1 == len(video['annotations']) - 1 or video['annotations'][index1]['video_id'] != video['annotations'][index1+1]['video_id']:
            for key in list(cur_video_ann.keys()):
                instance_id = 1
                for item in cur_video_ann[key]:
                    # assert item[0] is not None
                    if item[0] is not None:
                        annotation_val.append({'id': id_val, 'image_id': id_real_val_image, 'category_id': item[3],
                                               'instances_id': instance_id,
                                                 'bdd100k_id': 0, 'occluded': False, 'truncated': False,
                                                 'bbox': item[0], 'area': item[1], 'iscrowd': item[2], 'ignore': 0
                                                 })
                        assert item[3] <= 17
                        instance_id += 1
                        id_val += 1
                if id_val_image not in none_image_id_test:
                    id_real_val_image += 1
                id_val_image += 1

        pre_video_id = video['annotations'][index1]['video_id']


new_annotation = dict()
new_annotation['categories'] = kept_categories#video['categories']
new_annotation['videos'] = video_info_train
new_annotation['images'] = images_train
new_annotation['annotations'] = annotation_train
json.dump(new_annotation, open('/nobackup-slow/dataset/my_xfdu/video/vis/train/instances_train.json', 'w'))


new_annotation['videos'] = video_info_val
new_annotation['images'] = images_val
new_annotation['annotations'] = annotation_val
json.dump(new_annotation, open('/nobackup-slow/dataset/my_xfdu/video/vis/train/instances_test.json', 'w'))
breakpoint()



