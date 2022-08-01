import numpy as np
from PIL import Image, ImageDraw
import math
import random
import torch

# def RandomMask(s, hole_range=[0,1]):
#     coef = min(hole_range[0] + hole_range[1], 1.0)
#     while True:
#         mask = np.ones((s, s), np.uint8)
#         def Fill(max_size):
#             w, h = rng_seed_train.randint(max_size), rng_seed_train.randint(max_size)
#             ww, hh = w // 2, h // 2
#             x, y = rng_seed_train.randint(-ww, s - w + ww), rng_seed_train.randint(-hh, s - h + hh)
#             mask[max(y, 0): min(y + h, s), max(x, 0): min(x + w, s)] = 0
#         def MultiFill(max_tries, max_size):
#             for _ in range(rng_seed_train.randint(max_tries)):
#                 Fill(max_size)
#         MultiFill(int(10 * coef), s // 2)
#         MultiFill(int(5 * coef), s)
#         mask = np.logical_and(mask, 1 - RandomBrush(int(20 * coef), s))
#         hole_ratio = 1 - np.mean(mask)
#         if hole_range is not None and (hole_ratio <= hole_range[0] or hole_ratio >= hole_range[1]):
#             continue
#         return mask[np.newaxis, ...].astype(np.float32)

# def BatchRandomMask(batch_size, s, hole_range=[0, 1]):
#     return np.stack([RandomMask(s, hole_range=hole_range) for _ in range(batch_size)], axis = 0)

class RandomMask:
    def __init__(self, s, hole_range=[0,1]):
        self.s = s
        self.hole_range = hole_range
        self.rng_seed_train = np.random.RandomState()


    def RandomBrush(
        self,
        max_tries,
        s,
        min_num_vertex = 4,
        max_num_vertex = 18,
        mean_angle = 2*math.pi / 5,
        angle_range = 2*math.pi / 15,
        min_width = 12,
        max_width = 48):
        H, W = s, s
        average_radius = math.sqrt(H*H+W*W) / 8
        mask = Image.new('L', (W, H), 0)
        for _ in range(self.rng_seed_train.randint(max_tries)):
            num_vertex = self.rng_seed_train.randint(min_num_vertex, max_num_vertex)
            angle_min = mean_angle - self.rng_seed_train.uniform(0, angle_range)
            angle_max = mean_angle + self.rng_seed_train.uniform(0, angle_range)
            angles = []
            vertex = []
            for i in range(num_vertex):
                if i % 2 == 0:
                    angles.append(2*math.pi - self.rng_seed_train.uniform(angle_min, angle_max))
                else:
                    angles.append(self.rng_seed_train.uniform(angle_min, angle_max))

            h, w = mask.size
            vertex.append((int(self.rng_seed_train.randint(0, w)), int(self.rng_seed_train.randint(0, h))))
            for i in range(num_vertex):
                r = np.clip(
                    self.rng_seed_train.normal(loc=average_radius, scale=average_radius//2),
                    0, 2*average_radius)
                new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
                new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
                vertex.append((int(new_x), int(new_y)))

            draw = ImageDraw.Draw(mask)
            width = int(self.rng_seed_train.uniform(min_width, max_width))
            draw.line(vertex, fill=1, width=width)
            for v in vertex:
                draw.ellipse((v[0] - width//2,
                            v[1] - width//2,
                            v[0] + width//2,
                            v[1] + width//2),
                            fill=1)
            if self.rng_seed_train.random() > 0.5:
                mask.transpose(Image.FLIP_LEFT_RIGHT)
            if self.rng_seed_train.random() > 0.5:
                mask.transpose(Image.FLIP_TOP_BOTTOM)
        mask = np.asarray(mask, np.uint8)
        if self.rng_seed_train.random() > 0.5:
            mask = np.flip(mask, 0)
        if self.rng_seed_train.random() > 0.5:
            mask = np.flip(mask, 1)
        return mask

    def Fill(self, mask, max_size):
        w, h = self.rng_seed_train.randint(max_size), self.rng_seed_train.randint(max_size)
        ww, hh = w // 2, h // 2
        x, y = self.rng_seed_train.randint(-ww, self.s - w + ww), self.rng_seed_train.randint(-hh, self.s - h + hh)
        mask[max(y, 0): min(y + h, self.s), max(x, 0): min(x + w, self.s)] = 0

    def MultiFill(self, mask, max_tries, max_size):
        for _ in range(self.rng_seed_train.randint(max_tries)):
            self.Fill(mask, max_size)

    def __call__(self, seed):
        if seed is not None:
            # print(f'fixed seed {seed}')
            self.rng_seed_train = np.random.RandomState(seed)

        mask = np.ones((self.s, self.s), np.uint8)
        while True:
            coef = min(self.hole_range[0] + self.hole_range[1], 1.0)    
            self.MultiFill(mask, int(10 * coef), self.s // 2)
            self.MultiFill(mask, int(5 * coef), self.s)
            mask = np.logical_and(mask, 1 - self.RandomBrush(int(20 * coef), self.s))
            hole_ratio = 1 - np.mean(mask)
            if self.hole_range is not None and (hole_ratio <= self.hole_range[0] or hole_ratio >= self.hole_range[1]):
                mask.fill(1)
                continue
            else:
                break
        mask = mask.astype(np.float32)
        return mask



    def call_rectangle(self, seed):
        if seed is not None:
            # print(f'fixed seed {seed}')
            self.rng_seed_train = np.random.RandomState(seed)

        mask = np.ones((self.s, self.s), np.uint8)
        while True:
            coef = min(self.hole_range[0] + self.hole_range[1], 1.0)    
            self.MultiFill(mask, int(10 * coef), self.s // 2)
            self.MultiFill(mask, int(5 * coef), self.s)
            # mask = np.logical_and(mask, 1 - self.RandomBrush(int(20 * coef), self.s))
            hole_ratio = 1 - np.mean(mask)
            if self.hole_range is not None and (hole_ratio <= self.hole_range[0] or hole_ratio >= self.hole_range[1]):
                mask.fill(1)
                continue
            else:
                break
        mask = mask.astype(np.float32)
        return mask