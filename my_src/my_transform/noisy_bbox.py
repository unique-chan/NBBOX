from mmdet.datasets import PIPELINES

import numpy as np


@PIPELINES.register_module()
class NoisyBBOX:
    def __init__(self, scale_range=(0.7, 1.0), isotropically_rescaled=False,
                 angle_range=(-2, 2), translate_range=(-2, 2), isotropically_translated=False,
                 threshold=16):
        # hint for threshold
        # AP_{vt}:  AP for very tiny objects    -> Pixel areas between 2 & 8
        # AP_{t}:   AP for tiny objects         -> Pixel areas between 8 & 16
        # AP_{S}:   AP for small                -> Pixel areas less than 32^2
        # AP_{M}:   AP for medium               -> Pixel areas between 32^2 & 96^2
        # AP_{L}:   AP for large                -> Pixel areas greater than 96^2
        self.scale_range = scale_range
        self.isotropically_rescaled = isotropically_rescaled
        self.angle_range = angle_range
        self.translate_range = translate_range
        self.isotropically_translated = isotropically_translated
        self.threshold = threshold

    def __call__(self, results):
        # Extract
        bboxes = results['ann_info']['bboxes']

        # Rescale BBOX
        if self.isotropically_rescaled:
            scales = np.random.uniform(self.scale_range[0], self.scale_range[1], size=(bboxes.shape[0]))
            scales[(bboxes[:, 2] < self.threshold) | (bboxes[:, 3] < self.threshold)] = 1
            # if width < threshold or height < threshold -> no scaled
            bboxes[:, 2] *= scales          # width
            bboxes[:, 3] *= scales          # height
        else:
            scales = np.random.uniform(self.scale_range[0], self.scale_range[1], size=(bboxes.shape[0], 2))
            scales[(bboxes[:, 2] < self.threshold) | (bboxes[:, 3] < self.threshold)] = 1
            # if width < threshold or height < threshold -> no scaled
            bboxes[:, 2] *= scales[:, 0]    # width
            bboxes[:, 3] *= scales[:, 1]    # height

        # Re-rotate BBOX
        angles = np.random.uniform(self.angle_range[0], self.angle_range[1], size=(bboxes.shape[0]))
        angles[(bboxes[:, 2] < self.threshold) | (bboxes[:, 3] < self.threshold)] = 0
        # if width < threshold or height < threshold -> no rotated
        bboxes[:, 4] += angles

        # Re-translate BBOX
        if self.translate_range[0] < self.translate_range[1]:
            if self.isotropically_translated:
                translations = np.random.randint(self.translate_range[0], self.translate_range[1],
                                                 size=(bboxes.shape[0]))
                translations[(bboxes[:, 2] < self.threshold) | (bboxes[:, 3] < self.threshold)] = 0
                # if width < threshold or height < threshold -> no translated
                bboxes[:, 0] += translations
                bboxes[:, 1] += translations
            else:
                translations = np.random.randint(self.translate_range[0], self.translate_range[1],
                                                 size=(bboxes.shape[0], 2))
                translations[(bboxes[:, 2] < self.threshold) | (bboxes[:, 3] < self.threshold), :] = 0
                # if width < threshold or height < threshold -> no translated
                bboxes[:, 0] += translations[:, 0]
                bboxes[:, 1] += translations[:, 1]

        # Update
        results['ann_info']['bboxes'] = bboxes
        if 'img_info' in results and 'ann' in results['img_info']:
            results['img_info']['ann']['bboxes'] = bboxes

        # Update the polygons if they exist
        if 'polygons' in results['ann_info']:
            polygons = results['ann_info']['polygons']
            for i, poly in enumerate(polygons):
                cx, cy, w, h, angle = bboxes[i]
                cos_angle = np.cos(np.radians(angle))
                sin_angle = np.sin(np.radians(angle))
                half_w, half_h = w / 2, h / 2
                points = np.array([
                    [-half_w, -half_h],
                    [half_w, -half_h],
                    [half_w, half_h],
                    [-half_w, half_h]
                ])
                scaled_points = points * scales[i]
                rotated_points = np.dot(scaled_points, np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]]))
                translated_points = rotated_points + [cx, cy]
                results['ann_info']['polygons'][i] = translated_points.flatten()
                if 'polygons' in results['img_info']['ann']:
                    results['img_info']['ann']['polygons'][i] = translated_points.flatten()

        results['gt_bboxes'] = results['ann_info']['bboxes'].copy()

        return results
