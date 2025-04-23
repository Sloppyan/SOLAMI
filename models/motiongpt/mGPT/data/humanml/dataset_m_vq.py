import random
import codecs as cs
import numpy as np
from torch.utils import data
from rich.progress import track
from os.path import join as pjoin
from .dataset_m import MotionDataset
from .dataset_t2m import Text2MotionDataset
from .dataset_vq_new import Text2MotionVQDataset


class MotionDatasetVQ(Text2MotionVQDataset):
    def __init__(
        self,
        data_root,
        split,
        mean,
        std,
        transform_mean,
        transform_std,
        max_motion_length,
        min_motion_length,
        win_size,
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        **kwargs,
    ):
        super().__init__(data_root, split, mean, std, transform_mean, transform_std,
                         max_motion_length, min_motion_length, unit_length, fps, tmpFile, tiny,
                         debug, **kwargs)

        # Filter out the motions that are too short
        self.window_size = win_size
        name_list = list(self.name_list)
        for name in self.name_list:
            motion = self.data_dict[name]["motion"]
            if motion.shape[0] < self.window_size:
                name_list.remove(name)
                self.data_dict.pop(name)
        self.name_list = name_list
        print('Training data len: ', len(self.name_list))

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, item):
        
        idx = self.pointer + item
        data_item = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list, transform, partner_name = \
            data_item['motion'], data_item['length'], data_item['text'], data_item['transform'], data_item['partner_motion']
        
        
        ### transform
        if partner_name is None:
            # single person transform
            transform = np.random.normal(loc=self.transform_mean, scale=self.transform_std, size=self.transform_mean.shape)
            transform = (transform - self.transform_mean) / self.transform_std
        else:
            # two person transform
            transform = (transform - self.transform_mean) / self.transform_std
        
        
        ### next partner motion
        partner_motion = None
        if partner_name is not None:
            if partner_name in self.data_dict:
                partner_motion = self.data_dict[partner_name]['motion']
        
        
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption = text_data["caption"]

        # Text
        all_captions = [i['caption'] for i in text_list]
        if len(all_captions) >= 3:
            all_captions = all_captions[:3]
        else:
            for _ in range(3-len(all_captions)):
                all_captions.append(all_captions[0])

        # Crop the motions in to times of 4, and introduce small variations
        idx = random.randint(0, motion.shape[0] - self.window_size)
        motion = motion[idx:idx + self.window_size]

        motion = (motion - self.mean) / self.std
        m_length = min(m_length - idx, self.window_size)
        
        return caption, motion, m_length, all_captions, transform, partner_motion
