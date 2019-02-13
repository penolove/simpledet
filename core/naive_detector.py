import argparse

import arrow
import numpy as np
from six.moves.queue import Queue
from threading import Thread
from eyewitness.detection_utils import DetectionResult
from eyewitness.config import BoundedBoxObject
from eyewitness.image_id import ImageId
from eyewitness.object_detector import ObjectDetector
from eyewitness.image_utils import ImageHandler, Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"

def parse_args():
    parser = argparse.ArgumentParser(description='Test Detection')
    # general
    parser.add_argument('--config', help='config file path', type=str)
    args = parser.parse_args()

    config = importlib.import_module(args.config.replace('.py', '').replace('/', '.'))
    return config


class SimpleDetWrapper(ObjectDetector):
    def __init__(self, config):
        pGen, pKv, pRpn, pRoi, pBbox, pDataset, pModel, pOpt, pTest, \
        transform, data_name, label_name, metric_list = config.get_config(is_train=False)
        
        # skip the first read image transformer
        self.transform = transform[1:]
        self.data_name = data_name
        sym = pModel.test_symbol

        # TODO: design how to support multiple gpu (data paraellism)
        # expect only one gpu will be used
        assert len(pKv.gpus) == 1    
        ctx = mx.gpu(pKv.gpus[0])
        arg_params, aux_params = load_checkpoint(pTest.model.prefix, pTest.model.epoch)
        mod = DetModule(sym, data_names=data_names, context=ctx)
        mod.bind(data_shapes=loader.provide_data, for_training=False)
        self.mod.set_params(arg_params, aux_params, allow_extra=False)

    def do_nms(self, output_dict, nms):
        bbox_xyxy = output_dict["bbox_xyxy"]
        cls_score = output_dict["cls_score"]
        final_dets = {}

        for cid in range(cls_score.shape[1]):
            score = cls_score[:, cid]
            if bbox_xyxy.shape[1] != 4:
                cls_box = bbox_xyxy[:, cid * 4:(cid + 1) * 4]
            else:
                cls_box = bbox_xyxy
            valid_inds = np.where(score > pTest.min_det_score)[0]
            box = cls_box[valid_inds]
            score = score[valid_inds]
            det = np.concatenate((box, score.reshape(-1, 1)), axis=1).astype(np.float32)
            det = nms(det)
            dataset_cid = coco.getCatIds()[cid]
            final_dets[dataset_cid] = det
        output_dict["det_xyxys"] = final_dets
        del output_dict["bbox_xyxy"]
        del output_dict["cls_score"]
    
    def detect(self, image_obj) -> DetectionResult:

        # create a input_record with fake gt_bbox
        input_record = {
            'image': np.array(image_obj.pil_image_obj), 
            'gt_bbox': np.array()  # empty gt bbox
            'rec_id': 1  # fake id
            'im_id': 1  # fake id
        }   
        
        for trans in self.transform:
            trans.apply(input_record)
        
        data_batch = {}
        
        for name in self.data_name + self.label_name:
            data_batch[name] = np.stack([input_record[name]])

        data = [mx.nd.array(data_batch[name]) for name in self.data_name]
        label = [mx.nd.array(data_batch[name]) for name in self.label_name]
        provide_data = [(k, v.shape) for k, v in zip(self.data_name, data)]
        provide_label = [(k, v.shape) for k, v in zip(self.label_name, label)]
        # generate mx.io.DataBatch
        data_batch = mx.io.DataBatch(data=data,
                                     label=label,
                                     provide_data=provide_data,
                                     provide_label=provide_label)
        
        self.mod.forward(batch, is_train=False)
        out = [x.asnumpy() for x in self.mod.get_outputs()]
        
        _, _, info, label, box = out
        info, label, box = info.squeeze(), label.squeeze(), box.squeeze()

        scale = info[2]  # h_raw, w_raw, scale
        box = box / scale  # scale to original image scale
        label = label[:, 1:]   # remove background
        box = box[:, 4:] if box.shape[1] != 4 else box

        output_record = dict(
            im_info=info,
            bbox_xyxy=box,  # ndarray (n, class * 4) or (n, 4)
            cls_score=label   # ndarray (n, class)
        )
        all_outputs = pTest.process_output([output_record], None)

        # aggregate different scale outputs
        output_dict = defaultdict(list)
        for rec in all_outputs:
            output_dict['bbox_xyxy'].append(rec["bbox_xyxy"])
            output_dict['cls_score'].append(rec["cls_score"])

        # transform to numpy format
        if len(output_dict["bbox_xyxy"]) > 1:
            output_dict["bbox_xyxy"] = np.concatenate(output_dict["bbox_xyxy"])
        else:
            output_dict["bbox_xyxy"] = output_dict["bbox_xyxy"][0]

        if len(output_dict["cls_score"]) > 1:
            output_dict["cls_score"] = np.concatenate(output_dict["cls_score"])
        else:
            output_dict["cls_score"] = output_dict["cls_score"][0]


        if callable(pTest.nms.type):
            nms = pTest.nms.type(pTest.nms.thr)
        else:
            from operator_py.nms import py_nms_wrapper
            nms = py_nms_wrapper(pTest.nms.thr)
        
        self.do_nms(output_dict, nms)
        
        result = []
        for cid, det in output_dict["det_xyxys"].items():
            if det.shape[0] == 0:
                continue
                scores = det[:, -1]
                xs = det[:, 0]
                ys = det[:, 1]
                ws = det[:, 2] - xs + 1
                hs = det[:, 3] - ys + 1
                result += [
                    {'image_id': int(iid),
                    'category_id': int(cid),
                    'bbox': [float(xs[k]), float(ys[k]), float(ws[k]), float(hs[k])],
                    'score': float(scores[k])}
                    for k in range(det.shape[0])
                ]
        result = sorted(result, key=lambda x: x['score'])[-pTest.max_det_per_image:]
        
        # TODO: wrapper with eyewitness
        # detected_objects = []
        # for bbox, score, label_class in zip(out_boxes, out_scores, out_classes):
        #     label = self.core_model.class_names[label_class]
        #     y1, x1, y2, x2 = bbox
        #     if score > self.threshold:
        #         detected_objects.append(BoundedBoxObject(x1, y1, x2, y2, label, score, ''))

        # image_dict = {
        #     'image_id': image_obj.image_id,
        #     'detected_objects': detected_objects,
        # }
        # detection_result = DetectionResult(image_dict)
        # return detection_result

    @property
    def valid_labels(self):
        pass

if __name__ == "__main__":
    config = parse_args()

    object_detector = SimpleDetWrapper(config)
    raw_image_path = 'demo/test_image.jpg'
    image_id = ImageId(channel='demo', timestamp=arrow.now().timestamp, file_format='jpg')
    image_obj = Image(image_id, raw_image_path=raw_image_path)
    detection_result = object_detector.detect(image_obj)

    



