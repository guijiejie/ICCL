import torchvision.transforms as T
import torch
from PIL import Image, ImageFilter
import cv2
import numpy as np
import random
# from . import tarfile
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma_min=0.1, sigma_max=2.0, kernel_size=0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, x):
        sigma = random.uniform(self.sigma_min, self.sigma_max)
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

    def __repr__(self):
        return self.__class__.__name__

class ColorPermutation(object):
    def __init__(self):
        pass

    def __call__(self, x):
        assert x.size(0) == 3
        assert len(x.size()) == 3
        order = [0, 1, 2]
        random.shuffle(order)
        x = x[order,:,:]
        return x

    def __repr__(self):
        return self.__class__.__name__

class UnSqueeze(object):
    def __init__(self):
        pass

    def __call__(self, x):
        x = torch.unsqueeze(x, 0)
        return x

    def __repr__(self):
        return self.__class__.__name__

class SampleAllFrames(object):
    def __init__(self, start_index):
        self.start_index = start_index

    def __call__(self, total_frames):
        return np.array([i for i in range(self.start_index, total_frames)], dtype=np.int32).tolist()

    def __repr__(self):
        return self.__class__.__name__

class SampleMiddleFrames(object):
    def __init__(self, start_index, total_length, interval=1):
        self.start_index = start_index
        self.total_length = total_length
        self.interval = interval

    def __call__(self, total_frames):
        total_frames = total_frames if total_frames <= 2 * self.start_index else total_frames - 2 * self.start_index
        
        if total_frames < self.total_length:
            g_cpu = torch.Generator()
            g_cpu = g_cpu.manual_seed(2147483647)
            
            random_sample = torch.randint(total_frames, size=(self.total_length-total_frames,), generator=g_cpu) + self.start_index
            # random_sample = np.random.randint(total_frames, size=self.total_length-total_frames) + self.start_index
            
            selected_sample = np.arange(total_frames) + self.start_index
            return random_sample.cpu().numpy().tolist() + selected_sample.tolist()
        elif total_frames < self.total_length * self.interval:
            g_cpu = torch.Generator()
            g_cpu = g_cpu.manual_seed(2147483647)
            random_sample = torch.randint(total_frames, size=(self.total_length,), generator=g_cpu)

            # random_sample = np.random.randint(total_frames, size=self.total_length) +self.start_index
            return random_sample.cpu().numpy().tolist()
        else:
            start_index = (total_frames - self.total_length * self.interval) // 2
            return np.arange(start_index, start_index + self.total_length * self.interval, self.interval).tolist()

    def __repr__(self):
        return self.__class__.__name__

class SampleFrames(object):
    def __init__(self,
                 clip_len,
                 frame_interval=1,
                 clip_interval=1,
                 num_clips=1,
                 twice_sample=False,
                 out_of_bound_opt='repeat_last',
                 test_mode=False,
                 start_index=0):

        self.clip_len = clip_len
        self.clip_interval = clip_interval
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.twice_sample = twice_sample
        self.out_of_bound_opt = out_of_bound_opt
        self.test_mode = test_mode
        self.start_index = start_index
        assert self.out_of_bound_opt in ['loop', 'repeat_last']

    def _get_train_clips(self, num_frames):
        ori_clip_len = self.clip_len * self.clip_interval
        avg_interval = (num_frames - ori_clip_len + 1) // self.num_clips

        if avg_interval > 0:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = base_offsets + np.random.randint(
                avg_interval, size=self.num_clips)
        elif num_frames > max(self.num_clips, ori_clip_len):
            clip_offsets = np.sort(
                np.random.randint(
                    num_frames - ori_clip_len + 1, size=self.num_clips))
        elif avg_interval == 0:
            ratio = (num_frames - ori_clip_len + 1.0) / self.num_clips
            clip_offsets = np.around(np.arange(self.num_clips) * ratio)
        else:
            clip_offsets = np.zeros((self.num_clips, ), dtype=np.int)

        return clip_offsets

    def _get_test_clips(self, num_frames):
        ori_clip_len = self.clip_len * self.clip_interval
        avg_interval = (num_frames - ori_clip_len + 1) / float(self.num_clips)
        if num_frames > ori_clip_len - 1:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = (base_offsets + avg_interval / 2.0).astype(np.int)
            if self.twice_sample:
                clip_offsets = np.concatenate([clip_offsets, base_offsets])
        else:
            clip_offsets = np.zeros((self.num_clips, ), dtype=np.int)
        return clip_offsets

    def _sample_clips(self, num_frames):
        if self.test_mode:
            clip_offsets = self._get_test_clips(num_frames)
        else:
            clip_offsets = self._get_train_clips(num_frames)

        return clip_offsets

    def __call__(self, total_frames):
        tmp_frames = total_frames - 2 * self.start_index - self.frame_interval[1]
        total_frames = total_frames if tmp_frames < 0 else tmp_frames

        clip_offsets = self._sample_clips(total_frames)
        frame_inds = clip_offsets[:, None] + np.arange(
            self.clip_len)[None, :] * self.clip_interval
        frame_inds = np.concatenate(frame_inds)

        frame_inds = frame_inds.reshape((-1, self.clip_len))
        if self.out_of_bound_opt == 'loop':
            frame_inds = np.mod(frame_inds, total_frames)
        elif self.out_of_bound_opt == 'repeat_last':
            safe_inds = frame_inds < total_frames
            unsafe_inds = 1 - safe_inds
            last_ind = np.max(safe_inds * frame_inds, axis=1)
            new_inds = (safe_inds * frame_inds + (unsafe_inds.T * last_ind).T)
            frame_inds = new_inds
        else:
            raise ValueError('Illegal out_of_bound option.')

        frame_inds = np.concatenate(frame_inds) + self.start_index
        result = [frame_inds, []]
        for inds in frame_inds:
            result[1].append(inds + random.randint(*self.frame_interval))

        result = np.array(result, dtype=np.int32).tolist()
        return result

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'clip_interval={self.clip_interval}, '
                    f'frame_interval={self.frame_interval}, '
                    f'num_clips={self.num_clips}, '
                    f'twice_sample={self.twice_sample}, '
                    f'out_of_bound_opt={self.out_of_bound_opt}, '
                    f'start_index={self.start_index}, '
                    f'test_mode={self.test_mode})')
        return repr_str

class Solarization(object):
    def __init__(self, threshold=128):
        self.threshold = threshold

    def __call__(self, img):
        img = np.array(img)
        img = np.where(img < self.threshold, img, 255 -img)
        return Image.fromarray(img.astype(np.uint8))

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str

__EXCLUDE_DICT = {
    "GaussianBlur":GaussianBlur,
    "SampleFrames":SampleFrames,
    "SampleAllFrames":SampleAllFrames,
    "SampleMiddleFrames":SampleMiddleFrames,
    "UnSqueeze":UnSqueeze,
    "Solarization":Solarization,
    "ColorPermutation":ColorPermutation
}

def build_multiview_transforms(transform_list):
    trans_funcs = []
    for cfg in transform_list:
        t = cfg.pop("type")
        p = cfg.pop("rand_apply") if "rand_apply" in cfg else None
        if t in __EXCLUDE_DICT:
            func = __EXCLUDE_DICT[t](**cfg)
        else:
            func = getattr(T, t)(**cfg)
        
        if p is not None:
            func = T.RandomApply([func], p=p)
            cfg["rand_apply"] = p
        
        trans_funcs.append(func)

        cfg["type"] = t
    
    return trans_funcs

class MultiViewCrop(object):
    def __init__(self, transform_list, nmb_crops=[2, 6], size_crops=[224, 96], min_scale_crops=[0.2, 0.05], max_scale_crops=[1.0, 0.2], type=None):
        super().__init__()
        self.trans = {}
        basetrans = build_multiview_transforms(transform_list)
        for i in range(len(size_crops)):
            randomresizedcrop = T.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )
            tmptrans = [randomresizedcrop] + basetrans
            self.trans[size_crops[i]] = T.Compose(tmptrans)
        
        self.nmb_crops = nmb_crops
        self.size_crops = size_crops
    
    def __call__(self, x):
        rets = []
        for i in range(len(self.size_crops)):
            for j in range(self.nmb_crops[i]):
                rets.append(self.trans[self.size_crops[i]](x))
        
        return rets
    
    def __repr__(self):
        transstr = '\n'.join(list(map(str, list(self.trans.values()))))
        repr_str = f"{self.__class__.__name__} ({transstr})"
        return repr_str

# def MultiViewCrop(transform_list, nmb_crops=[2, 6], size_crops=[224, 96], min_scale_crops=[0.2, 0.05], max_scale_crops=[1.0, 0.2], type=None):
#     trans = []
#     # if sum(nmb_crops) > 2:
#     #     transform_list.append(dict(type="UnSqueeze"))
        
#     for i in range(len(size_crops)):
#         randomresizedcrop = T.RandomResizedCrop(
#             size_crops[i],
#             scale=(min_scale_crops[i], max_scale_crops[i]),
#         )

#         trans.extend([T.Compose([randomresizedcrop] + build_multiview_transforms(transform_list))] * nmb_crops[i])
    
#     return trans

def build_transforms(transform_list):
    assert len(transform_list) > 0
    if isinstance(transform_list[0], list):
        assert len(transform_list) == 2
        return [build_transforms(transform_list[0]), build_transforms(transform_list[1])]
    
    if transform_list[0]['type'] == "MultiViewCrop":
        return MultiViewCrop(transform_list[1:], **transform_list[0])

    trans_funcs = []
    for cfg in transform_list:
        t = cfg.pop("type")
        p = cfg.pop("rand_apply") if "rand_apply" in cfg else None

        if t == 'GaussianBlur':
            func = GaussianBlur(**cfg)
        elif t in __EXCLUDE_DICT:
            func = __EXCLUDE_DICT[t](**cfg)
        else:
            func = getattr(T, t)(**cfg)
        
        if p is not None:
            func = T.RandomApply([func], p=p)
        
        trans_funcs.append(func)
    return T.Compose(trans_funcs)
