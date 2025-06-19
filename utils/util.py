import copy
import math
import random
import time

import numpy
import torch
import torchvision
from torch.nn.functional import cross_entropy, one_hot

import torch
from torchsummary import summary
from contextlib import redirect_stdout

def flatten_model(model):
    """
    This function flattens a model into a Sequential-like list of layers
    while handling custom layers and standard ones.
    """
    modules = []
    
    # Iterate through all child modules in the model
    for name, layer in model.named_children():
        if isinstance(layer, torch.nn.ModuleList) or isinstance(layer, torch.nn.Sequential):
            # If the layer is a ModuleList or Sequential, recursively flatten it
            modules.extend(flatten_model(layer))
        elif isinstance(layer, torch.nn.Conv2d):
            # If it's a Conv2d, add Conv2d and BatchNorm if exists
            modules.append(layer)
            # Check if BatchNorm is attached
            if hasattr(layer, 'norm'):
                modules.append(layer.norm)
            # Check for activation function (e.g., SiLU or ReLU)
            if hasattr(layer, 'relu'):
                modules.append(layer.relu)
        elif isinstance(layer, torch.nn.BatchNorm2d):
            modules.append(layer)
        elif isinstance(layer, torch.nn.ReLU) or isinstance(layer, torch.nn.SiLU):
            modules.append(layer)
        elif isinstance(layer, torch.nn.MaxPool2d):
            modules.append(layer)
        else:
            # For any other layer, just append it directly
            modules.append(layer)
    
    return modules

class GeneralizedBackboneWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        # Flatten the model
        modules = flatten_model(model)
        
        # Now the backbone is clearly a flat Sequential model of primitive layers
        self.backbone = torch.nn.Sequential(*modules)

    def forward(self, x):
        return self.backbone(x)
    
    


import cv2, os
import matplotlib.pyplot as plt
import numpy as np
def generate_colors(num_classes):
    """
    Helper function to generate random colors for each class.
    Ensures colors are created for all possible class IDs up to num_classes.
    """
    np.random.seed(42)  # For consistent colors
    colors = {}
    for i in range(num_classes):
        colors[i] = tuple(np.random.randint(0, 256, 3).tolist())
    return colors


def visualize_predictions(samples, outputs, gt_boxes, shapes, params, class_colors, results_dir, index):
    """
    Visualize first 5 images from the val set: draws ground-truth boxes on the
    left, predicted boxes on the right, and saves side-by-side images.
    """
    
    
    # Convert single image to CPU numpy
    img = samples[0].cpu().float().numpy()
    img = img.transpose((1, 2, 0))  # CHW -> HWC
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = (img * 255).astype(np.uint8)

    # Remove padding, resize back
    original_shape = shapes[0][0]
    pad_w, pad_h = shapes[0][1][1]
    h, w = img.shape[:2]
    img = img[int(pad_h):int(h - pad_h), int(pad_w):int(w - pad_w)]
    img = cv2.resize(img, (int(original_shape[1]), int(original_shape[0])))

    img_gt = img.copy()
    img_pred = img.copy()

    # Draw GT
    for gt in gt_boxes:
        cls_gt = int(gt[1])
        coords = gt[2:].cpu().numpy()
        x_c, y_c = coords[0], coords[1]
        w_b, h_b = coords[2], coords[3]

        # Convert to corner coordinates
        x1 = int(x_c - w_b / 2)
        y1 = int(y_c - h_b / 2)
        x2 = int(x_c + w_b / 2)
        y2 = int(y_c + h_b / 2)
        color = tuple(map(int, class_colors[cls_gt]))
        cv2.rectangle(img_gt, (x1, y1), (x2, y2), color, 2)
        if cls_gt in params['names']:
            cls_name = params['names'][cls_gt]
        else:
            cls_name = str(cls_gt)
        cv2.putText(img_gt, cls_name, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Draw predictions
    if outputs[0] is not None:
        
        # Scale predictions using util.scale like in test function
        det_clone = outputs[0].clone()
        scale(det_clone[:, :4], samples[0].shape[1:], shapes[0][0], shapes[0][1])
        
        for det in det_clone.cpu().numpy():
            x1, y1, x2, y2, conf, cls_id = det
            cls_id = int(cls_id)
            color = tuple(map(int, class_colors[cls_id]))
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            if cls_id in params['names']:
                cls_name = params['names'][cls_id]
            else:
                cls_name = str(cls_id)
            cv2.rectangle(img_pred, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_pred, f"{cls_name} {conf:.2f}",
                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Save side-by-side
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB))
    plt.title('Ground Truth')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB))
    plt.title('Predictions')
    plt.axis('off')

    save_path = os.path.join(results_dir, f'result_{index}.png')
    plt.savefig(save_path)
    plt.close()


def setup_seed():
    """
    Setup random seed.
    """
    random.seed(0)
    numpy.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_multi_processes():
    """
    Setup multi-processing environment variables.
    """
    import cv2
    from os import environ
    from platform import system

    # set multiprocess start method as `fork` to speed up the training
    if system() != 'Windows':
        torch.multiprocessing.set_start_method('fork', force=True)

    # disable opencv multithreading to avoid system being overloaded
    cv2.setNumThreads(0)

    # setup OMP threads
    if 'OMP_NUM_THREADS' not in environ:
        environ['OMP_NUM_THREADS'] = '1'

    # setup MKL threads
    if 'MKL_NUM_THREADS' not in environ:
        environ['MKL_NUM_THREADS'] = '1'


def scale(coords, shape1, shape2, ratio_pad=None):
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(shape1[0] / shape2[0], shape1[1] / shape2[1])  # gain  = old / new
        pad = (shape1[1] - shape2[1] * gain) / 2, (shape1[0] - shape2[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain

    coords[:, 0].clamp_(0, shape2[1])  # x1
    coords[:, 1].clamp_(0, shape2[0])  # y1
    coords[:, 2].clamp_(0, shape2[1])  # x2
    coords[:, 3].clamp_(0, shape2[0])  # y2
    return coords


def make_anchors(x, strides, offset=0.5):
    """
    Generate anchors from features
    """
    assert x is not None
    anchor_points, stride_tensor = [], []
    for i, stride in enumerate(strides):
        _, _, h, w = x[i].shape
        sx = torch.arange(end=w, dtype=x[i].dtype, device=x[i].device) + offset  # shift x
        sy = torch.arange(end=h, dtype=x[i].dtype, device=x[i].device) + offset  # shift y
        sy, sx = torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=x[i].dtype, device=x[i].device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # intersection(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
    intersection = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = intersection / (area1 + area2 - intersection)
    box1 = box1.T
    box2 = box2.T

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return intersection / (area1[:, None] + area2 - intersection)


def wh2xy(x):
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def non_max_suppression(prediction, conf_threshold=0.25, iou_threshold=0.45):
    print(f"\n--- [utils/util.py::non_max_suppression] PYTORCH NMS DEBUG ---")
    print(f"[utils/util.py::non_max_suppression] Prediction shape: {prediction.shape}")
    print(f"[utils/util.py::non_max_suppression] Conf threshold: {conf_threshold}, IoU threshold: {iou_threshold}")
    
    nc = prediction.shape[1] - 4  # number of classes
    print(f"[utils/util.py::non_max_suppression] Number of classes: {nc}")
    xc = prediction[:, 4:4 + nc].amax(1) > conf_threshold  # candidates
    print(f"[utils/util.py::non_max_suppression] Candidates per image: {[xc[i].sum().item() for i in range(prediction.shape[0])]}")

    # Settings
    max_wh = 7680  # (pixels) maximum box width and height
    max_det = 300  # the maximum number of boxes to keep after NMS
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()

    start = time.time()
    outputs = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for index, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x = x.transpose(0, -1)[xc[index]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (box, conf, cls)
        box, cls = x.split((4, nc), 1)
        # center_x, center_y, width, height) to (x1, y1, x2, y2)
        box = wh2xy(box)
        if nc > 1:
            i, j = (cls > conf_threshold).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_threshold]
        # Check shape
        if not x.shape[0]:  # no boxes
            continue
        # sort by confidence and remove excess boxes
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_threshold)  # NMS
        i = i[:max_det]  # limit detections
        outputs[index] = x[i]
        if (time.time() - start) > 0.5 + 0.05 * prediction.shape[0]:
            print(f'WARNING ⚠️ NMS time limit {0.5 + 0.05 * prediction.shape[0]:.3f}s exceeded')
            break  # time limit exceeded

    return outputs


def smooth(y, f=0.05):
    # Box filter of fraction f
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = numpy.ones(nf // 2)  # ones padding
    yp = numpy.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return numpy.convolve(yp, numpy.ones(nf) / nf, mode='valid')  # y-smoothed


def compute_ap(tp, conf, pred_cls, target_cls, eps=1e-16):
    """
    Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Object-ness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
    # Returns
        The average precision
    """
    print(f"\n--- [utils/util.py::compute_ap] PYTORCH COMPUTE_AP DEBUG ---")
    print(f"[utils/util.py::compute_ap] TP shape: {tp.shape}, dtype: {tp.dtype}")
    print(f"[utils/util.py::compute_ap] Conf shape: {conf.shape}, dtype: {conf.dtype}")
    print(f"[utils/util.py::compute_ap] Pred_cls shape: {pred_cls.shape}, dtype: {pred_cls.dtype}")
    print(f"[utils/util.py::compute_ap] Target_cls shape: {target_cls.shape}, dtype: {target_cls.dtype}")
    if conf.size > 0:
        print(f"[utils/util.py::compute_ap] Conf min/max: {conf.min():.6f}/{conf.max():.6f}")
    else:
        print(f"[utils/util.py::compute_ap] Conf array is empty - no predictions to evaluate")
    
    # Sort by object-ness
    i = numpy.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    print(f"After sorting - first 5 conf: {conf[:5]}")

    # Find unique classes
    unique_classes, nt = numpy.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections
    print(f"Unique classes: {unique_classes}")
    print(f"Class counts: {nt}")
    print(f"Number of classes: {nc}")

    # Create Precision-Recall curve and compute AP for each class
    p = numpy.zeros((nc, 1000))
    r = numpy.zeros((nc, 1000))
    ap = numpy.zeros((nc, tp.shape[1]))
    px, py = numpy.linspace(0, 1, 1000), []  # for plotting
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        nl = nt[ci]  # number of labels
        no = i.sum()  # number of outputs
        if no == 0 or nl == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (nl + eps)  # recall curve
        # negative x, xp because xp decreases
        r[ci] = numpy.interp(-px, -conf[i], recall[:, 0], left=0)

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = numpy.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            m_rec = numpy.concatenate(([0.0], recall[:, j], [1.0]))
            m_pre = numpy.concatenate(([1.0], precision[:, j], [0.0]))

            # Compute the precision envelope
            m_pre = numpy.flip(numpy.maximum.accumulate(numpy.flip(m_pre)))

            # Integrate area under curve
            x = numpy.linspace(0, 1, 101)  # 101-point interp (COCO)
            ap[ci, j] = numpy.trapz(numpy.interp(x, m_rec, m_pre), x)  # integrate

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)

    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
    m_pre, m_rec = p.mean(), r.mean()
    map50, mean_ap = ap50.mean(), ap.mean()
    return tp, fp, m_pre, m_rec, map50, mean_ap


def strip_optimizer(filename):
    x = torch.load(filename, map_location=torch.device('cpu'))
    x['model'].half()  # to FP16
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, filename)


def clip_gradients(model, max_norm=10.0):
    parameters = model.parameters()
    torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)


class EMA:
    """
    Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = copy.deepcopy(model).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        # decay exponential ramp (to help early epochs)
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        if hasattr(model, 'module'):
            model = model.module
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()


class AverageMeter:
    def __init__(self):
        self.num = 0
        self.sum = 0
        self.avg = 0

    def update(self, v, n):
        if not math.isnan(float(v)):
            self.num = self.num + n
            self.sum = self.sum + v * n
            self.avg = self.sum / self.num


class ComputeLoss:
    def __init__(self, model, params):
        super().__init__()
        if hasattr(model, 'module'):
            model = model.module

        device = next(model.parameters()).device  # get model device

        m = model.head  # Head() module
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.device = device
        self.params = params

        # task aligned assigner
        self.top_k = 10
        self.alpha = 0.5
        self.beta = 6.0
        self.eps = 1e-9

        self.bs = 1
        self.num_max_boxes = 0
        # DFL Loss params
        self.dfl_ch = m.dfl.ch
        self.project = torch.arange(self.dfl_ch, dtype=torch.float, device=device)
        
        print(f"\n--- [utils/util.py::ComputeLoss.__init__] PYTORCH INITIALIZATION DEBUG ---")
        print(f"[utils/util.py::ComputeLoss.__init__] Model head stride: {self.stride}")
        print(f"[utils/util.py::ComputeLoss.__init__] Number of classes: {self.nc}")
        print(f"[utils/util.py::ComputeLoss.__init__] Output channels (no): {self.no}")
        print(f"[utils/util.py::ComputeLoss.__init__] DFL channels: {self.dfl_ch}")
        print(f"[utils/util.py::ComputeLoss.__init__] Task aligned assigner params - top_k: {self.top_k}, alpha: {self.alpha}, beta: {self.beta}")
        print(f"[utils/util.py::ComputeLoss.__init__] Loss weights - cls: {self.params['cls']}, box: {self.params['box']}, dfl: {self.params['dfl']}")

    def __call__(self, outputs, targets):
        print(f"\n--- [utils/util.py::ComputeLoss.__call__] PYTORCH LOSS COMPUTATION DEBUG ---")
        print(f"[utils/util.py::ComputeLoss.__call__] Outputs type: {type(outputs)}")
        
        x = outputs[1] if isinstance(outputs, tuple) else outputs
        print(f"[utils/util.py::ComputeLoss.__call__] x type: {type(x)}, len: {len(x) if isinstance(x, (list, tuple)) else 'not list/tuple'}")
        
        if isinstance(x, (list, tuple)):
            for i, xi in enumerate(x):
                print(f"[utils/util.py::ComputeLoss.__call__] x[{i}] shape: {xi.shape}")
        
        output = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], 2)
        print(f"[utils/util.py::ComputeLoss.__call__] Concatenated output shape: {output.shape}")
        print(f"[utils/util.py::ComputeLoss.__call__] self.no: {self.no}, self.dfl_ch: {self.dfl_ch}, self.nc: {self.nc}")
        
        pred_output, pred_scores = output.split((4 * self.dfl_ch, self.nc), 1)
        print(f"[utils/util.py::ComputeLoss.__call__] pred_output shape: {pred_output.shape}")
        print(f"[utils/util.py::ComputeLoss.__call__] pred_scores shape: {pred_scores.shape}")

        pred_output = pred_output.permute(0, 2, 1).contiguous()
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        print(f"[utils/util.py::ComputeLoss.__call__] After permute - pred_output shape: {pred_output.shape}")
        print(f"[utils/util.py::ComputeLoss.__call__] After permute - pred_scores shape: {pred_scores.shape}")

        size = torch.tensor(x[0].shape[2:], dtype=pred_scores.dtype, device=self.device)
        size = size * self.stride[0]
        print(f"[utils/util.py::ComputeLoss.__call__] Size tensor: {size}")
        print(f"[utils/util.py::ComputeLoss.__call__] Stride: {self.stride}")

        anchor_points, stride_tensor = make_anchors(x, self.stride, 0.5)
        print(f"[utils/util.py::ComputeLoss.__call__] Anchor points shape: {anchor_points.shape}")
        print(f"[utils/util.py::ComputeLoss.__call__] Stride tensor shape: {stride_tensor.shape}")
        
        print(f"\n--- [utils/util.py::ComputeLoss.__call__] TARGET PROCESSING ---")
        print(f"[utils/util.py::ComputeLoss.__call__] Targets shape: {targets.shape}")
        print(f"[utils/util.py::ComputeLoss.__call__] Targets dtype: {targets.dtype}")
        if targets.shape[0] > 0:
            print(f"[utils/util.py::ComputeLoss.__call__] First few targets: {targets[:min(5, targets.shape[0])]}")

        # targets
        if targets.shape[0] == 0:
            gt = torch.zeros(pred_scores.shape[0], 0, 5, device=self.device)
            print(f"[utils/util.py::ComputeLoss.__call__] No targets - gt shape: {gt.shape}")
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            print(f"[utils/util.py::ComputeLoss.__call__] Image indices: {i}")
            print(f"[utils/util.py::ComputeLoss.__call__] Counts per image: {counts}")
            gt = torch.zeros(pred_scores.shape[0], counts.max(), 5, device=self.device)
            print(f"[utils/util.py::ComputeLoss.__call__] GT tensor shape: {gt.shape}")
            for j in range(pred_scores.shape[0]):
                matches = i == j
                n = matches.sum()
                if n:
                    gt[j, :n] = targets[matches, 1:]
                    print(f"[utils/util.py::ComputeLoss.__call__] Image {j}: {n} targets assigned")
            print(f"[utils/util.py::ComputeLoss.__call__] GT before coordinate conversion: {gt}")
            gt[..., 1:5] = wh2xy(gt[..., 1:5].mul_(size[[1, 0, 1, 0]]))
            print(f"[utils/util.py::ComputeLoss.__call__] GT after coordinate conversion: {gt}")

        gt_labels, gt_bboxes = gt.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        print(f"[utils/util.py::ComputeLoss.__call__] GT labels shape: {gt_labels.shape}")
        print(f"[utils/util.py::ComputeLoss.__call__] GT bboxes shape: {gt_bboxes.shape}")
        print(f"[utils/util.py::ComputeLoss.__call__] GT mask shape: {mask_gt.shape}")

        # boxes
        b, a, c = pred_output.shape
        print(f"[utils/util.py::ComputeLoss.__call__] Pred output dimensions - b: {b}, a: {a}, c: {c}")
        pred_bboxes = pred_output.view(b, a, 4, c // 4).softmax(3)
        print(f"[utils/util.py::ComputeLoss.__call__] Pred bboxes after softmax shape: {pred_bboxes.shape}")
        pred_bboxes = pred_bboxes.matmul(self.project.type(pred_bboxes.dtype))
        print(f"[utils/util.py::ComputeLoss.__call__] Pred bboxes after matmul shape: {pred_bboxes.shape}")

        a, b = torch.split(pred_bboxes, 2, -1)
        pred_bboxes = torch.cat((anchor_points - a, anchor_points + b), -1)
        print(f"[utils/util.py::ComputeLoss.__call__] Final pred bboxes shape: {pred_bboxes.shape}")

        scores = pred_scores.detach().sigmoid()
        bboxes = (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype)
        print(f"[utils/util.py::ComputeLoss.__call__] Scores shape: {scores.shape}")
        print(f"[utils/util.py::ComputeLoss.__call__] Bboxes shape: {bboxes.shape}")

        print(f"\n--- [utils/util.py::ComputeLoss.__call__] ASSIGNMENT PHASE ---")
        target_bboxes, target_scores, fg_mask = self.assign(scores, bboxes,
                                                            gt_labels, gt_bboxes, mask_gt,
                                                            anchor_points * stride_tensor)
        target_bboxes /= stride_tensor
        target_scores_sum = target_scores.sum()
        print(f"[utils/util.py::ComputeLoss.__call__] Target bboxes shape: {target_bboxes.shape}")
        print(f"[utils/util.py::ComputeLoss.__call__] Target scores shape: {target_scores.shape}")
        print(f"[utils/util.py::ComputeLoss.__call__] FG mask shape: {fg_mask.shape}")
        print(f"[utils/util.py::ComputeLoss.__call__] Target scores sum: {target_scores_sum.item():.6f}")

        # cls loss
        loss_cls = self.bce(pred_scores, target_scores.to(pred_scores.dtype))
        loss_cls = loss_cls.sum() / target_scores_sum
        print(f"[utils/util.py::ComputeLoss.__call__] Classification loss computed: {loss_cls.item():.6f}")

        # box loss
        loss_box = torch.zeros(1, device=self.device)
        loss_dfl = torch.zeros(1, device=self.device)
        fg_count = fg_mask.sum()
        print(f"[utils/util.py::ComputeLoss.__call__] Foreground count: {fg_count.item()}")
        
        if fg_mask.sum():
            # IoU loss
            weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
            print(f"[utils/util.py::ComputeLoss.__call__] Weight shape: {weight.shape}")
            loss_box = self.iou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
            print(f"[utils/util.py::ComputeLoss.__call__] IoU values shape: {loss_box.shape}")
            loss_box = ((1.0 - loss_box) * weight).sum() / target_scores_sum
            print(f"[utils/util.py::ComputeLoss.__call__] Box loss computed: {loss_box.item():.6f}")
            
            # DFL loss
            a, b = torch.split(target_bboxes, 2, -1)
            target_lt_rb = torch.cat((anchor_points - a, b - anchor_points), -1)
            target_lt_rb = target_lt_rb.clamp(0, self.dfl_ch - 1.01)  # distance (left_top, right_bottom)
            print(f"[utils/util.py::ComputeLoss.__call__] Target LT-RB shape: {target_lt_rb.shape}")
            loss_dfl = self.df_loss(pred_output[fg_mask].view(-1, self.dfl_ch), target_lt_rb[fg_mask])
            print(f"[utils/util.py::ComputeLoss.__call__] DFL loss before weighting: {loss_dfl.mean().item():.6f}")
            loss_dfl = (loss_dfl * weight).sum() / target_scores_sum
            print(f"[utils/util.py::ComputeLoss.__call__] DFL loss computed: {loss_dfl.item():.6f}")

        print(f"\n--- [utils/util.py::ComputeLoss.__call__] LOSS COMPONENTS DEBUG ---")
        print(f"[utils/util.py::ComputeLoss.__call__] Raw loss_cls: {loss_cls.item():.6f}")
        print(f"[utils/util.py::ComputeLoss.__call__] Raw loss_box: {loss_box.item():.6f}")
        print(f"[utils/util.py::ComputeLoss.__call__] Raw loss_dfl: {loss_dfl.item():.6f}")
        print(f"[utils/util.py::ComputeLoss.__call__] Loss weights - cls: {self.params['cls']}, box: {self.params['box']}, dfl: {self.params['dfl']}")
        print(f"[utils/util.py::ComputeLoss.__call__] target_scores_sum: {target_scores_sum.item():.6f}")
        print(f"[utils/util.py::ComputeLoss.__call__] fg_mask sum: {fg_mask.sum().item()}")
        
        loss_cls *= self.params['cls']
        loss_box *= self.params['box']
        loss_dfl *= self.params['dfl']
        
        total_loss = loss_cls + loss_box + loss_dfl
        print(f"[utils/util.py::ComputeLoss.__call__] Weighted loss_cls: {loss_cls.item():.6f}")
        print(f"[utils/util.py::ComputeLoss.__call__] Weighted loss_box: {loss_box.item():.6f}")
        print(f"[utils/util.py::ComputeLoss.__call__] Weighted loss_dfl: {loss_dfl.item():.6f}")
        print(f"[utils/util.py::ComputeLoss.__call__] Total loss: {total_loss.item():.6f}")
        print(f"--- [utils/util.py::ComputeLoss.__call__] END PYTORCH LOSS COMPUTATION DEBUG ---\n")
        
        return total_loss  # loss(cls, box, dfl)

    @torch.no_grad()
    def assign(self, pred_scores, pred_bboxes, true_labels, true_bboxes, true_mask, anchors):
        """
        Task-aligned One-stage Object Detection assigner
        """
        print(f"\n--- [utils/util.py::ComputeLoss.assign] PYTORCH ASSIGNMENT DEBUG ---")
        print(f"[utils/util.py::ComputeLoss.assign] pred_scores shape: {pred_scores.shape}")
        print(f"[utils/util.py::ComputeLoss.assign] pred_bboxes shape: {pred_bboxes.shape}")
        print(f"[utils/util.py::ComputeLoss.assign] true_labels shape: {true_labels.shape}")
        print(f"[utils/util.py::ComputeLoss.assign] true_bboxes shape: {true_bboxes.shape}")
        print(f"[utils/util.py::ComputeLoss.assign] true_mask shape: {true_mask.shape}")
        print(f"[utils/util.py::ComputeLoss.assign] anchors shape: {anchors.shape}")
        
        self.bs = pred_scores.size(0)
        self.num_max_boxes = true_bboxes.size(1)
        print(f"[utils/util.py::ComputeLoss.assign] Batch size: {self.bs}, Max boxes: {self.num_max_boxes}")

        if self.num_max_boxes == 0:
            device = true_bboxes.device
            print(f"[utils/util.py::ComputeLoss.assign] No ground truth boxes, returning zeros")
            return (torch.zeros_like(pred_bboxes).to(device),
                    torch.zeros_like(pred_scores).to(device),
                    torch.zeros_like(pred_scores[..., 0]).to(device).bool())

        i = torch.zeros([2, self.bs, self.num_max_boxes], dtype=torch.long)
        i[0] = torch.arange(end=self.bs).view(-1, 1).repeat(1, self.num_max_boxes)
        i[1] = true_labels.long().squeeze(-1)
        print(f"[utils/util.py::ComputeLoss.assign] Created indices tensor with shape: {i.shape}")
        
        # Calculate overlaps
        print(f"\n--- [utils/util.py::ComputeLoss.assign] IOUs DEBUG ---")
        print(f"[utils/util.py::ComputeLoss.assign] true_bboxes shape: {true_bboxes.shape}")
        print(f"[utils/util.py::ComputeLoss.assign] pred_bboxes shape: {pred_bboxes.shape}")
        

        overlaps = self.iou(true_bboxes.unsqueeze(2), pred_bboxes.unsqueeze(1))
        overlaps = overlaps.squeeze(3).clamp(0)
        print(f"[utils/util.py::ComputeLoss.assign] overlaps shape: {overlaps.shape}")
        print(f"[utils/util.py::ComputeLoss.assign] overlaps min: {overlaps.min()}, max: {overlaps.max()}")
        align_metric = pred_scores[i[0], :, i[1]].pow(self.alpha) * overlaps.pow(self.beta)
        print(f"[utils/util.py::ComputeLoss.assign] align_metric shape: {align_metric.shape}")
        print(f"[utils/util.py::ComputeLoss.assign] align_metric min: {align_metric.min()}, max: {align_metric.max()}")
        
        bs, n_boxes, _ = true_bboxes.shape
        lt, rb = true_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
        bbox_deltas = torch.cat((anchors[None] - lt, rb - anchors[None]), dim=2)
        mask_in_gts = bbox_deltas.view(bs, n_boxes, anchors.shape[0], -1).amin(3).gt_(1e-9)
        print(f"[utils/util.py::ComputeLoss.assign] mask_in_gts shape: {mask_in_gts.shape}")
        print(f"[utils/util.py::ComputeLoss.assign] mask_in_gts sum: {mask_in_gts.sum()}")
        
        metrics = align_metric * mask_in_gts
        print(f"[utils/util.py::ComputeLoss.assign] metrics shape: {metrics.shape}")
        top_k_mask = true_mask.repeat([1, 1, self.top_k]).bool()
        print(f"[utils/util.py::ComputeLoss.assign] top_k_mask shape: {top_k_mask.shape}")
        
        num_anchors = metrics.shape[-1]
        top_k_metrics, top_k_indices = torch.topk(metrics, self.top_k, dim=-1, largest=True)
        print(f"[utils/util.py::ComputeLoss.assign] top_k_metrics shape: {top_k_metrics.shape}")
        print(f"[utils/util.py::ComputeLoss.assign] top_k_indices shape: {top_k_indices.shape}")
        
        if top_k_mask is None:
            top_k_mask = (top_k_metrics.max(-1, keepdim=True) > self.eps).tile([1, 1, self.top_k])
        top_k_indices = torch.where(top_k_mask, top_k_indices, 0)
        is_in_top_k = one_hot(top_k_indices, num_anchors).sum(-2)
        print(f"[utils/util.py::ComputeLoss.assign] is_in_top_k shape: {is_in_top_k.shape}")
        
        # filter invalid boxes
        is_in_top_k = torch.where(is_in_top_k > 1, 0, is_in_top_k)
        mask_top_k = is_in_top_k.to(metrics.dtype)
        print(f"[utils/util.py::ComputeLoss.assign] mask_top_k shape: {mask_top_k.shape}")
        
        # merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_top_k * mask_in_gts * true_mask
        print(f"[utils/util.py::ComputeLoss.assign] mask_pos shape: {mask_pos.shape}")

        fg_mask = mask_pos.sum(-2)
        print(f"[utils/util.py::ComputeLoss.assign] fg_mask shape: {fg_mask.shape}")
        print(f"[utils/util.py::ComputeLoss.assign] fg_mask sum: {fg_mask.sum()}")
        if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
            print(f"[utils/util.py::ComputeLoss.assign] Detected anchors assigned to multiple GT boxes")
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).repeat([1, self.num_max_boxes, 1])
            max_overlaps_idx = overlaps.argmax(1)
            is_max_overlaps = one_hot(max_overlaps_idx, self.num_max_boxes)
            is_max_overlaps = is_max_overlaps.permute(0, 2, 1).to(overlaps.dtype)
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)
            fg_mask = mask_pos.sum(-2)
            print(f"[utils/util.py::ComputeLoss.assign] After resolving multiple assignments - fg_mask sum: {fg_mask.sum()}")
        
        # find each grid serve which gt(index)
        target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
        print(f"[utils/util.py::ComputeLoss.assign] target_gt_idx shape: {target_gt_idx.shape}")

        # assigned target labels, (b, 1)
        batch_index = torch.arange(end=self.bs,
                                   dtype=torch.int64,
                                   device=true_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_index * self.num_max_boxes
        target_labels = true_labels.long().flatten()[target_gt_idx]
        print(f"[utils/util.py::ComputeLoss.assign] target_labels shape: {target_labels.shape}")

        # assigned target boxes
        target_bboxes = true_bboxes.view(-1, 4)[target_gt_idx]
        print(f"[utils/util.py::ComputeLoss.assign] target_bboxes shape: {target_bboxes.shape}")

        # assigned target scores
        target_labels.clamp(0)
        target_scores = one_hot(target_labels, self.nc)
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.nc)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)
        print(f"[utils/util.py::ComputeLoss.assign] target_scores shape: {target_scores.shape}")

        # normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(axis=-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(axis=-1, keepdim=True)
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2)
        norm_align_metric = norm_align_metric.unsqueeze(-1)
        target_scores = target_scores * norm_align_metric
        print(f"[utils/util.py::ComputeLoss.assign] Normalized target_scores shape: {target_scores.shape}")
        print(f"[utils/util.py::ComputeLoss.assign] target_scores sum: {target_scores.sum()}")
        
        print(f"--- [utils/util.py::ComputeLoss.assign] END PYTORCH ASSIGNMENT DEBUG ---\n")
        return target_bboxes, target_scores, fg_mask.bool()

    @staticmethod
    def df_loss(pred_dist, target):
        # Return sum of left and right DFL losses
        # Distribution Focal Loss https://ieeexplore.ieee.org/document/9792391
        print(f"\n--- [utils/util.py::ComputeLoss.df_loss] PYTORCH DFL LOSS DEBUG ---")
        print(f"[utils/util.py::ComputeLoss.df_loss] pred_dist shape: {pred_dist.shape}")
        print(f"[utils/util.py::ComputeLoss.df_loss] target shape: {target.shape}")
        
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        
        l_loss = cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape)
        r_loss = cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape)
        
        loss = (l_loss * wl + r_loss * wr).mean(-1, keepdim=True)
        print(f"[utils/util.py::ComputeLoss.df_loss] DFL loss shape: {loss.shape}")
        print(f"[utils/util.py::ComputeLoss.df_loss] DFL loss mean: {loss.mean().item():.6f}")
        print(f"--- [utils/util.py::ComputeLoss.df_loss] END PYTORCH DFL LOSS DEBUG ---\n")
        
        return loss

    @staticmethod
    def iou(box1, box2, eps=1e-7):
        # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)
        print(f"\n--- [utils/util.py::ComputeLoss.iou] PYTORCH IOU DEBUG ---")
        print(f"[utils/util.py::ComputeLoss.iou] box1 shape: {box1.shape}")
        print(f"[utils/util.py::ComputeLoss.iou] box2 shape: {box2.shape}")

        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
        
        print(f"[utils/util.py::ComputeLoss.iou] Box1 dimensions - w: {w1.shape}, h: {h1.shape}")
        print(f"[utils/util.py::ComputeLoss.iou] Box2 dimensions - w: {w2.shape}, h: {h2.shape}")

        # Intersection area
        area1 = b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)
        area2 = b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
        intersection = area1.clamp(0) * area2.clamp(0)
        print(f"[utils/util.py::ComputeLoss.iou] Intersection shape: {intersection.shape}")

        # Union Area
        union = w1 * h1 + w2 * h2 - intersection + eps
        print(f"[utils/util.py::ComputeLoss.iou] Union shape: {union.shape}")

        # IoU
        iou = intersection / union
        print(f"[utils/util.py::ComputeLoss.iou] IoU shape: {iou.shape}")
        if iou.numel() > 0:
            print(f"[utils/util.py::ComputeLoss.iou] IoU min: {iou.min().item():.6f}, max: {iou.max().item():.6f}")
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        # Complete IoU https://arxiv.org/abs/1911.08287v1
        c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
        print(f"[utils/util.py::ComputeLoss.iou] Convex diagonal squared shape: {c2.shape}")
        
        # center dist ** 2
        rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
        print(f"[utils/util.py::ComputeLoss.iou] Center distance squared shape: {rho2.shape}")
        
        # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
        v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
        print(f"[utils/util.py::ComputeLoss.iou] Aspect ratio term shape: {v.shape}")
        
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        
        ciou = iou - (rho2 / c2 + v * alpha)  # CIoU
        print(f"[utils/util.py::ComputeLoss.iou] CIoU shape: {ciou.shape}")
        if ciou.numel() > 0:
            print(f"[utils/util.py::ComputeLoss.iou] CIoU min: {ciou.min().item():.6f}, max: {ciou.max().item():.6f}")
        print(f"--- [utils/util.py::ComputeLoss.iou] END PYTORCH IOU DEBUG ---\n")
        
        return ciou  # CIoU
