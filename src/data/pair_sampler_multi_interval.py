import torch
from torch.utils.data.sampler import Sampler

from detectron2.utils import comm

import copy
import itertools
import math
import numpy as np
import random
from collections import defaultdict
from typing import Optional

__all__ = ["PairTrainingMultiIntervalSampler", "PairDataIntervalLoader"]


class PairTrainingMultiIntervalSampler(Sampler):
    """
    Similar to TrainingSampler, but produces a pair of training images from the
    same video sequence.
    """

    def __init__(
        self, cfg, dataset_dicts, batch_size, shuffle=True, seed=None
    ):
        """
        Args:
            cfg: config parameters
            dataset_dicts (list[dict]): annotations in Detectron2 dataset format.
            batch_size (int): Size of mini-batch.
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        self._offset = cfg.DATALOADER.PAIR_OFFSET_RANGE
        self._select = cfg.DATALOADER.SELCTED_NUMBER
        self._interval = cfg.DATALOADER.INTERVAL

        self._shuffle = shuffle
        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)
        # only sample the previous frame during eval

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

        self._total_size = len(dataset_dicts)
        total_batch_size = batch_size * self._world_size
        self._size = (
            len(dataset_dicts) // total_batch_size
        ) * total_batch_size
        self._batch_size = batch_size
        self.num_per_worker = self._size // self._world_size

        self._dataset_dicts = dataset_dicts
        self._data_by_video = {}
        for i, data in enumerate(dataset_dicts):
            data["total_idx"] = i
            if data["video_id"] in self._data_by_video:
                self._data_by_video[data["video_id"]][data["index"]] = data
            else:
                self._data_by_video[data["video_id"]] = {data["index"]: data}

    def __iter__(self):
        while True:
            indices = self._infinite_indices()[: self._size]
            split = indices[
                self._rank
                * self.num_per_worker : (self._rank + 1)
                * self.num_per_worker
            ]
            for i in range(0, len(split), self._batch_size):
                chunk = split[i : i + self._batch_size]
                pairs = []
                for c in chunk:
                    pairs.append(c)
                    vid_id = self._dataset_dicts[c]["video_id"]
                    index = self._dataset_dicts[c]["index"]
                    offset = []
                    for cur_index in range(self._select):
                        offsets = [
                            o
                            for o in [-self._interval*(cur_index+1), self._interval*(cur_index+1)]
                            if o != 0
                            and index + o in self._data_by_video[vid_id].keys()
                        ]
                        if not offsets:
                            offsets = [0]
                        offset += [random.choice(offsets)]

                    # offset = random.sample(offsets, self._select)
                    pair_idx = index + np.asarray(offset)
                    for temp in pair_idx:
                        pair = self._data_by_video[vid_id][temp]
                        pairs.append(pair["total_idx"])
                yield pairs

    def _infinite_indices(self):
        # pylint: disable=no-member
        g = torch.Generator()
        g.manual_seed(self._seed)
        if self._shuffle:
            return torch.randperm(self._total_size, generator=g)
        else:
            return torch.arange(self._total_size)


class PairDataIntervalLoader:
    """
    Wrapping DataLoader to add random flipping for pairs of images.
    """

    def __init__(self, cfg, dataloader):
        self.cfg = cfg
        self.dataloader = dataloader

    def __iter__(self):
        # pylint: disable=no-member
        for data in iter(self.dataloader):
            # print(len(data))
            num_pairs = len(data) // (self.cfg.DATALOADER.SELCTED_NUMBER + 1)
            for i in range(num_pairs):
                # breakpoint()
                datum = data[i * (self.cfg.DATALOADER.SELCTED_NUMBER + 1) :
                             (i + 1) * (self.cfg.DATALOADER.SELCTED_NUMBER + 1)]
                rand = random.randint(0, 1)
                if self.cfg.DATALOADER.NO_FLIP or rand == 0:
                    continue
                # flip both images in pair
                for d in datum:
                    w = d["instances"]._image_size[1]
                    d["image"] = torch.flip(d["image"], [2])
                    boxes = d["instances"].get("gt_boxes")
                    boxes.tensor[:, 0] = w - boxes.tensor[:, 0]
                    boxes.tensor[:, 2] = w - boxes.tensor[:, 2]
                    temp = copy.deepcopy(boxes.tensor[:, 2])
                    boxes.tensor[:, 2] = boxes.tensor[:, 0]
                    boxes.tensor[:, 0] = temp
            yield data
