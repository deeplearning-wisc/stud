import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools
import json
from PIL import Image

tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

# categories = [{'id': 1, 'name': 'vehicle'},
#               {'id': 2, 'name': 'pedestrian'},
#               {'id': 3, 'name': 'sign'},
#               {'id': 4, 'name': 'cyclist'},
#               {'id': 255, 'name': 'unknown'}]

# only 3 categories needed
categories = [{
    'id': 1,
    'name': 'vehicle'
}, {
    'id': 2,
    'name': 'pedestrian'
}, {
    'id': 3,
    'name': 'cyclist'
}]

base_dir = '/nobackup-slow/dataset/my_xfdu/video/waymo/' # path to raw tfrecords
# raw_dir = os.path.join(base_dir, 'raw')
raw_dir = base_dir
out_dir = '/nobackup-slow/dataset/my_xfdu/video/waymo/' # path to save directory
image_dir = os.path.join(out_dir, 'images')

# different cameras
id_camera_dict = {
    1: 'front',
    2: 'front_left',
    3: 'front_right',
    4: 'side_left',
    5: 'side_right'
}
cameras = [{'id': k, 'name': id_camera_dict[k]} for k in id_camera_dict.keys()]

phases = ['validation', 'training']
instance_id_dict = dict()

from tqdm import tqdm

for phase in phases:
    coco_json = {
        k:
        dict(cameras=cameras[k-1], videos=[], images=[], categories=categories,
             annotations=[])
        for k in id_camera_dict.keys()
    }
    coco_json['all'] = dict(
        cameras=cameras, videos=[], images=[], categories=categories, annotations=[])
    print('Converting phase {}...'.format(phase))
    tf_records_filenames = [
        os.path.join(raw_dir, phase, p)
        for p in os.listdir(os.path.join(raw_dir, phase))
        if p.endswith('.tfrecord')
    ]
    dataset = tf.data.TFRecordDataset(tf_records_filenames, compression_type='')
    counter = 0
    cur_video = ''
    video_base_dir = ''
    for data in tqdm(dataset, total=39987 if phase == 'validation' else 158081):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        if not cur_video == frame.context.name:
            cur_video = frame.context.name
            counter = 0
            for k in id_camera_dict.keys():
                for label_name in [k, 'all']:
                    coco_json[label_name]['videos'].append(
                        dict(
                            id=len(coco_json[label_name]['videos']),
                            file_name='{}/{}/{}'.format(phase, id_camera_dict[k],
                                                        cur_video),
                            camera_id=k,
                    ))
        for i, label in zip(
                frame.images, frame.camera_labels
                if phase != 'test' else [None for _ in frame.images]):
            video_base_dir = os.path.join(image_dir, phase,
                                          id_camera_dict[i.name], cur_video)
            img_name = os.path.join(video_base_dir,
                                    '{}_{:07d}.jpg'.format(cur_video, counter))
            os.makedirs(video_base_dir, exist_ok=True)
            img_array = tf.image.decode_jpeg(i.image).numpy()
            img = Image.fromarray(img_array)
            # save image: $base_dir/images/$camera/$phase/$video_name
            img.save(img_name)

            for label_name in [i.name, 'all']:
                coco_json[label_name]['images'].append(
                    dict(
                        id=len(coco_json[label_name]['images']),
                        file_name='{}/{}/{}/{}_{:07d}.jpg'.format(
                            phase, id_camera_dict[i.name], cur_video, cur_video,
                            counter),
                        height=img.size[1],
                        width=img.size[0],
                        index=counter,
                        video_id=coco_json[label_name]['videos'][-1]['id'],
                        timestamp_micros=frame.timestamp_micros))

            # save label
            if phase == 'test':
                continue
            lbl = label.labels
            for l in lbl:
                # type = 0 UNKNOWN, type = 3 SIGN
                if l.type == 0 or l.type == 3:
                    continue
                x, y, w, h = [
                    l.box.center_x - l.box.length / 2.,
                    l.box.center_y - l.box.width / 2., l.box.length,
                    l.box.width
                ]
                if l.id not in instance_id_dict:
                    instance_id_dict[l.id] = len(instance_id_dict)
                for label_name in [i.name, 'all']:
                    coco_json[label_name]['annotations'].append(
                        dict(
                            id=len(coco_json[label_name]['annotations']),
                            image_id=coco_json[label_name]['images'][-1]['id'],
                            category_id=l.type if l.type != 4 else 3,
                            instance_id=instance_id_dict[l.id],
                            # N/A
                            is_occluded=False,
                            is_truncated=False,
                            # xywh
                            bbox=[x, y, w, h],
                            area=l.box.width * l.box.length,
                            # N/A
                            iscrowd=False,
                            ignore=False,
                            segmentation=[[x, y, x + w, y, x + w, y + h, x,
                                           y + h]]))
        counter += 1

    os.makedirs(os.path.join(out_dir, 'labels'), exist_ok=True)
    with open(
            os.path.join(out_dir, 'labels',
                         'waymo12_all_{}_3cls.json'.format(phase)), 'w') as f:
        json.dump(coco_json['all'], f)
    for k in coco_json.keys():
        if k != 'all':
            with open(
                    os.path.join(
                        out_dir, 'labels',
                        'waymo12_{}_{}_3cls.json'.format(id_camera_dict[k], phase)),
                    'w') as f:
                json.dump(coco_json[k], f)
