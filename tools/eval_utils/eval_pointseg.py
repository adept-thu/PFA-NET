import pickle
import time
import torch
import tqdm
import numpy as np

from pcdet.models import load_data_to_gpu
from pcdet.utils import box_utils
from pcdet.datasets.kitti.kitti_object_eval_python.eval import print_str



def eval_one_epoch_seg(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            pred_dict = model(batch_dict)
            # for key, value in pred_dict.items():
            #     if key != 'batch_size':
            #         print(key, type(value), value.shape)
            #     else:
            #         print(key, value)
            print(batch_dict['frame_id'])
        annos = []
        pts_num = int(len(pred_dict['points'])/pred_dict['batch_size'])
        points = pred_dict['point_coords'].cpu().numpy()
        gt_boxes = pred_dict['gt_boxes'].cpu().numpy()
        point_cls_scores = pred_dict['point_cls_scores'].cpu().numpy()
        for j in range(pred_dict['batch_size']):
            boxes = gt_boxes[j]
            boxes = boxes[~np.all(boxes == 0, axis=1)]
            annos.append({
                'point_coords': points[j*pts_num:(j+1)*pts_num, :],
                'gt_boxes': boxes,
                'point_cls_scores': point_cls_scores[j*pts_num:(j+1)*pts_num]
            })
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix()
            progress_bar.update()

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = point_seg_evaluation(det_annos, class_names, output_path=final_output_dir)

    logger.info(result_str)
    #ret_dict.update(result_dict)

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return result_dict


def compute_confusion_matrix(label, pred, normalize=False):
  '''Computes a confusion matrix
   Args:
         label: true labels(numpy array: [N])
         pred: predicted labels(numpy array: [N])
  return:
         conf: confusion matrix(numpy array: [N*N])
  '''
  N = len(np.unique(label))
  conf = np.zeros((N, N))
  for i in range(len(label)):
      conf[label[i]][pred[i]] += 1
  if normalize:
      conf = conf / conf.sum(axis=1, keepdims=True)
  return conf


def point_seg_evaluation(det_dicts, classnames, output_path):
    result_str = ''
    result_dict = {}
    total_correct = 0
    total_seen = 0
    total_correct_class = [0 for _ in classnames]
    total_seen_class = [0 for _ in classnames]
    total_iou_class = [0 for _ in classnames]
    labels = []
    preds = []
    for det in det_dicts:
        # print('****************************Eval detection point scores************************ ')
        # print(det['point_cls_scores'][:100])############
        point_cls_labels = np.zeros((len(det['point_coords'])))
        for box in det['gt_boxes']:
            box_dim = box[np.newaxis, :]
            box_dim = box_dim[:, 0:7]
            corners = box_utils.boxes_to_corners_3d(box_dim)
            corners = np.squeeze(corners, axis=0)
            flag = box_utils.in_hull(det['point_coords'][:, 1:], corners)
            point_cls_labels[flag] = box[-1]
        # print(point_cls_labels[:100])###############
        # print(det['point_cls_scores'].shape, point_cls_labels.shape)#################
        total_correct += np.sum(det['point_cls_scores'] == point_cls_labels)
        total_seen += det['point_cls_scores'].size
        for i in range(len(classnames)):
            total_seen_class[i] += np.sum((point_cls_labels == i+1))
            total_correct_class[i] += np.sum((det['point_cls_scores'] == i+1) & (point_cls_labels == i+1))
            total_iou_class += np.sum((det['point_cls_scores'] == i+1) | (point_cls_labels == i+1))
        labels += [round(x) for x in point_cls_labels]
        preds += det['point_cls_scores'].tolist()

    total_correct /= total_seen
    acc = np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6)
    # mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_class, dtype=np.float) + 1e-6))
    mIoU = np.array(total_correct_class) / (np.array(total_iou_class, dtype=np.float) + 1e-6)
    #result_str += print_str((f"point avg IoU: {mIoU:.4f}"))
    result_str += print_str((f"Avg point segmentation accuracy: {total_correct:.4f}"))
    result_str += print_str(f"Car acc: {acc[0]:.4f}")
    result_str += print_str(f"Pedestrian acc: {acc[1]:.4f}")
    result_str += print_str(f"Cyclist acc: {acc[2]:.4f}")
    result_str += print_str(f"Car IoU: {mIoU[0]:.4f}")
    result_str += print_str(f"Pedestrian IoU: {mIoU[1]:.4f}")
    result_str += print_str(f"Cyclist IoU: {mIoU[2]:.4f}")

    confusion_matrix = compute_confusion_matrix(labels, preds)
    recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
    precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
    result_str += print_str(f"Car precision score: {precision[0]:.4f}")
    result_str += print_str(f"Pedestrian precision score: {precision[1]:.4f}")
    result_str += print_str(f"Cyclist precision score: {precision[2]:.4f}")
    result_str += print_str(f"Car recall score: {recall[0]:.4f}")
    result_str += print_str(f"Pedestrian recall score: {recall[1]:.4f}")
    result_str += print_str(f"Cyclist recall score: {recall[2]:.4f}")


    result_dict['avg_acc'] = total_correct
    result_dict['avg_car_acc'] = acc[0]
    result_dict['avg_ped_acc'] = acc[1]
    result_dict['avg_cyc_acc'] = acc[2]
    # result_dict['mIoU'] = mIoU
    result_dict['avg_car_iou'] = mIoU[0]
    result_dict['avg_ped_iou'] = mIoU[1]
    result_dict['avg_cyc_iou'] = mIoU[2]
    result_dict['avg_car_precision'] = precision[0]
    result_dict['avg_ped_precision'] = precision[1]
    result_dict['avg_cyc_precision'] = precision[2]
    result_dict['avg_car_recall'] = recall[0]
    result_dict['avg_ped_recall'] = recall[1]
    result_dict['avg_cyc_recall'] = recall[2]

    return result_str, result_dict


if __name__ == '__main__':
    pass
