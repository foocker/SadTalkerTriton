
import numpy as np
from tqdm import tqdm 

def softmax(x):
    if x.ndim == 1:
        S = np.sum(np.exp(x))
        return np.exp(x)/S
    elif x.ndim == 2:
        result = np.zeros_like(x)
        M, N = x.shape
        for m in range(M):
            S = np.sum(np.exp(x[m, :]))
            result[m, :] = np.exp(x[m, :])/S
        return result
    else:
        print("The input array is not 1- or 2-dimensional.")


def headpose_pred_to_degree(pred):
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = np.array(idx_tensor).astype('float32')
    pred = softmax(pred)
    degree = np.sum(pred*idx_tensor, 1) * 3 - 99
    return degree


def get_rotation_matrix(yaw, pitch, roll):
    yaw = yaw / 180 * 3.14
    pitch = pitch / 180 * 3.14
    roll = roll / 180 * 3.14

    roll = np.expand_dims(roll, 1)
    pitch = np.expand_dims(pitch, 1)
    yaw = np.expand_dims(yaw, 1)

    pitch_mat = np.concatenate([np.ones_like(pitch), np.zeros_like(pitch), np.zeros_like(pitch), 
                                np.zeros_like(pitch), np.cos(pitch), -np.sin(pitch),
                                np.zeros_like(pitch), np.sin(pitch), np.cos(pitch)], axis=1)
    pitch_mat = pitch_mat.reshape(pitch_mat.shape[0], 3, 3)

    yaw_mat = np.concatenate([np.cos(yaw), np.zeros_like(yaw), np.sin(yaw), 
                              np.zeros_like(yaw), np.ones_like(yaw), np.zeros_like(yaw),
                              -np.sin(yaw), np.zeros_like(yaw), np.cos(yaw)], axis=1)
    yaw_mat = yaw_mat.reshape(yaw_mat.shape[0], 3, 3)

    roll_mat = np.concatenate([np.cos(roll), -np.sin(roll), np.zeros_like(roll),  
                              np.sin(roll), np.cos(roll), np.zeros_like(roll),
                              np.zeros_like(roll), np.zeros_like(roll), np.ones_like(roll)], axis=1)
    roll_mat = roll_mat.reshape(roll_mat.shape[0], 3, 3)

    rot_mat = np.einsum('bij,bjk,bkm->bim', pitch_mat, yaw_mat, roll_mat)

    return rot_mat.astype('float32')


def keypoint_transformation(kp, yaw, pitch, roll, t, exp):   
    yaw = headpose_pred_to_degree(yaw)
    pitch = headpose_pred_to_degree(pitch)
    roll = headpose_pred_to_degree(roll)

    rot_mat = get_rotation_matrix(yaw, pitch, roll)    # (bs, 3, 3)
    
    # keypoint rotation
    kp_rotated = np.einsum('bmp,bkp->bkm', rot_mat, kp)

    # keypoint translation
    t[:, 0] = t[:, 0]*0
    t[:, 2] = t[:, 2]*0
    # t = t.unsqueeze(1).repeat(1, kp.shape[1], 1)
    t = np.tile(np.expand_dims(t, 1), (1, kp.shape[1], 1))
    kp_t = kp_rotated + t

    # add expression deviation 
    exp = exp.reshape(exp.shape[0], -1, 3)
    kp_transformed = kp_t + exp

    return kp_transformed

def make_animation(source_image, source_semantics, target_semantics,
                   generator, kp_detector, mapping):
    predictions = []
    ort_inputs = {kp_detector.get_inputs()[0].name: source_image}
    kp_canonical = kp_detector.run(None, ort_inputs)[0]
    ort_inputs = {mapping.get_inputs()[0].name: source_semantics}
    yaw, pitch, roll, t, exp = mapping.run(None, ort_inputs)
    kp_source = keypoint_transformation(kp_canonical, yaw, pitch, roll, t, exp)
    for frame_idx in tqdm(range(target_semantics.shape[1]), 'Face Renderer:'):
        # still check the dimension
        if frame_idx % 2 == 0:
            target_semantics_frame = target_semantics[:, frame_idx]
            ort_inputs = {mapping.get_inputs()[0].name: target_semantics_frame}
            yaw, pitch, roll, t, exp = mapping.run(None, ort_inputs)
            
            kp_driving = keypoint_transformation(kp_canonical, yaw, pitch, roll, t, exp)
            ort_inputs = {'source': source_image,
                        'driving_value': kp_driving,
                        'source_value': kp_source
                        }
            out = generator.run(None, ort_inputs)[0]  # TODO 多batch 或结果插值， 插值可根据音频结果分析或者视频out数据分析。
            predictions.append(out)
            # print(out.shape)  # (1, 3, 256, 256)
    predictions_ts = np.stack(predictions, axis=1)  # (1, 42, 3, 256, 256)
    p_shape = list(predictions_ts.shape )
    p_shape[1] = p_shape[1] * 2 - 1
    predictions_tt = np.empty(p_shape)
    
    predictions_tt[:, ::2, ...] = predictions_ts
    predictions_tt[:, 1::2, ...] = (predictions_ts[:, :-1, ...] + predictions_ts[:, 1:, ...] )/ 2

    return predictions_tt