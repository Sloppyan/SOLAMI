import random
import numpy as np
from .dataset_t2m import Text2MotionDataset


class Text2MotionDatasetEval(Text2MotionDataset):

    def __init__(
        self,
        data_root,
        split,
        mean,
        std,
        transform_mean,
        transform_std,
        w_vectorizer,
        max_motion_length=196,
        min_motion_length=40,
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        **kwargs,
    ):
        super().__init__(data_root, split, mean, std, transform_mean, transform_std, max_motion_length,
                         min_motion_length, unit_length, fps, tmpFile, tiny,
                         debug, **kwargs)

        self.w_vectorizer = w_vectorizer


    def __getitem__(self, item):
        # Get text data
        idx = self.pointer + item
        data_item = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list, transform, partner_name = \
            data_item['motion'], data_item['length'], data_item['text'], data_item['transform'], data_item['partner_motion']

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
        
        all_captions = [i['caption'] for i in text_list]
        if len(all_captions) >= 3:
            all_captions = all_captions[:3]
        else:
            for _ in range(3-len(all_captions)):
                all_captions.append(all_captions[0])
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]

        m_length = (m_length // self.unit_length) * self.unit_length

        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]
    
        # Z Normalization
        motion = (motion - self.mean) / self.std

        return caption, motion, m_length, all_captions, transform, partner_motion
