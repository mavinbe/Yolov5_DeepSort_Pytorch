import numpy as np
import torch
import sys

from utils.general import LOGGER
from utils.torch_utils import time_sync
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.detection import Detection
from .sort.tracker import Tracker

sys.path.append('./Yolov5_DeepSort_Pytorch/deep_sort/deep/reid')
from torchreid.utils import FeatureExtractor

__all__ = ['DeepSort']


class DeepSort(object):
    def __init__(self, model_type, device, max_dist=0.2, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100, _lambda=0):

        self.extractor = FeatureExtractor(
            model_name=model_type,
            device=str(device)
        )

        max_cosine_distance = max_dist
        metric = NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(
            metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init, _lambda=_lambda)

    def update(self, bbox_xywh, confidences, classes, ori_img, use_yolo_preds=False):
        self.height, self.width = ori_img.shape[:2]
        # generate detections
        t1 = time_sync()
        features = self._get_features(bbox_xywh, ori_img)
        t2 = time_sync()
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(
            confidences)]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])

        t3 = time_sync()
        # update tracker
        self.tracker.predict()
        t4 = time_sync()
        self.tracker.update(detections, classes)
        t5 = time_sync()
        # output bbox identities
        outputs = []
        confirmed_track_list = []
        for track in self.tracker.tracks:
            if track.is_confirmed():
                confirmed_track_list.append(track)
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            if use_yolo_preds:
                det = track.get_yolo_pred()
                x1, y1, x2, y2 = self._tlwh_to_xyxy(det.tlwh)
            else:
                box = track.to_tlwh()
                x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            class_id = track.class_id
            time_since_update = track.time_since_update
            outputs.append(np.array([x1, y1, x2, y2, track_id, class_id], dtype=np.int))
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        t6 = time_sync()
        #LOGGER.info(f'update_tracker:({(t6 - t1) * 1000:.3f}ms) _get_features::({(t2 - t1) * 1000:.3f}ms), prepare::({(t3 - t2) * 1000:.3f}ms), predict::({(t4 - t3) * 1000:.3f}ms), tracker.update::({(t5 - t4) * 1000:.3f}ms), appendix::({(t6 - t5) * 1000:.3f}ms)')

        return outputs, confirmed_track_list

    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x+w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y+h), self.height - 1)
        return x1, y1, x2, y2

    def increment_ages(self):
        self.tracker.increment_ages()

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features
