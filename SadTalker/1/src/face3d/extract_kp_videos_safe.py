import cv2
import numpy as np
import onnxruntime
from src.face3d.scrfd import SCRFD


def calculate_points(heatmaps):
    # change heatmaps to landmarks
    B, N, H, W = heatmaps.shape
    HW = H * W
    BN_range = np.arange(B * N)

    heatline = heatmaps.reshape(B, N, HW)
    indexes = np.argmax(heatline, axis=2)

    preds = np.stack((indexes % W, indexes // W), axis=2)
    preds = preds.astype(np.float, copy=False)

    inr = indexes.ravel()

    heatline = heatline.reshape(B * N, HW)
    x_up = heatline[BN_range, inr + 1]
    x_down = heatline[BN_range, inr - 1]
    # y_up = heatline[BN_range, inr + W]

    if any((inr + W) >= 4096):
        y_up = heatline[BN_range, 4095]
    else:
        y_up = heatline[BN_range, inr + W]
    if any((inr - W) <= 0):
        y_down = heatline[BN_range, 0]
    else:
        y_down = heatline[BN_range, inr - W]

    think_diff = np.sign(np.stack((x_up - x_down, y_up - y_down), axis=1))
    think_diff *= .25

    preds += think_diff.reshape(B, N, 2)
    preds += .5
    return preds


def landmark_98_to_68(landmark_98):
    """Transfer 98 landmark positions to 68 landmark positions.
    Args:
        landmark_98(numpy array): Polar coordinates of 98 landmarks, (98, 2)
    Returns:
        landmark_68(numpy array): Polar coordinates of 98 landmarks, (68, 2)
    """

    landmark_68 = np.zeros((68, 2), dtype='float32')
    # cheek
    for i in range(0, 33):
        if i % 2 == 0:
            landmark_68[int(i / 2), :] = landmark_98[i, :]
    # nose
    for i in range(51, 60):
        landmark_68[i - 24, :] = landmark_98[i, :]
    # mouth
    for i in range(76, 96):
        landmark_68[i - 28, :] = landmark_98[i, :]
    # left eyebrow
    landmark_68[17, :] = landmark_98[33, :]
    landmark_68[18, :] = (landmark_98[34, :] + landmark_98[41, :]) / 2
    landmark_68[19, :] = (landmark_98[35, :] + landmark_98[40, :]) / 2
    landmark_68[20, :] = (landmark_98[36, :] + landmark_98[39, :]) / 2
    landmark_68[21, :] = (landmark_98[37, :] + landmark_98[38, :]) / 2
    # right eyebrow
    landmark_68[22, :] = (landmark_98[42, :] + landmark_98[50, :]) / 2
    landmark_68[23, :] = (landmark_98[43, :] + landmark_98[49, :]) / 2
    landmark_68[24, :] = (landmark_98[44, :] + landmark_98[48, :]) / 2
    landmark_68[25, :] = (landmark_98[45, :] + landmark_98[47, :]) / 2
    landmark_68[26, :] = landmark_98[46, :]
    # left eye
    LUT_landmark_68_left_eye = [36, 37, 38, 39, 40, 41]
    LUT_landmark_98_left_eye = [60, 61, 63, 64, 65, 67]
    for idx, landmark_98_index in enumerate(LUT_landmark_98_left_eye):
        landmark_68[LUT_landmark_68_left_eye[idx], :] = landmark_98[landmark_98_index, :]
    # right eye
    LUT_landmark_68_right_eye = [42, 43, 44, 45, 46, 47]
    LUT_landmark_98_right_eye = [68, 69, 71, 72, 73, 75]
    for idx, landmark_98_index in enumerate(LUT_landmark_98_right_eye):
        landmark_68[LUT_landmark_68_right_eye[idx], :] = landmark_98[landmark_98_index, :]

    return landmark_68


class KeypointExtractor():
    def __init__(self):
        self.face_det_model = SCRFD(model_file='/models/SadTalker/1/onnx_weights/scrfd_500m_bnkps.onnx')
        
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        # TODO resnet->18 or repvgg
        facelandmarks = onnxruntime.InferenceSession('/models/SadTalker/1/onnx_weights/face_landmark_awing_fan_simple.onnx', sess_options, 
                                                     providers=[('TensorrtExecutionProvider', {'trt_fp16_enable': True, 'trt_engine_cache_enable': True, 
                                                                                               'trt_engine_cache_path': '/models/SadTalker/1/cache'}), 'CUDAExecutionProvider'])
        self.facelandmarks = facelandmarks
        
    def facelandmark_infer_onnx(self, img):
        H, W, _ = img.shape
        offset = W / 64, H / 64, 0, 0
        img = cv2.resize(img, (256, 256)) # crop的人脸
        inp = img[..., ::-1]
        inp = np.ascontiguousarray(inp.transpose((2, 0, 1))) / 255
        inp = inp[np.newaxis, ...].astype(np.float32)
        ort_inputs = {self.facelandmarks.get_inputs()[0].name: inp}
        face_landmakrs98_outputs = self.facelandmarks.run(None, ort_inputs)
        outputs = face_landmakrs98_outputs[len(face_landmakrs98_outputs)//2-1][-1]
        
        heatmaps = outputs[:-1, :, :][np.newaxis, ...]  # 98, 64, 64 -> (1, 98, 64, 64)
        pred = calculate_points(heatmaps).reshape(-1, 2)

        pred *= offset[:2]
        pred += offset[-2:]
        # (98, 2)
        return pred
    
    def extract_keypoint_onnx(self, image):
        img = np.array(image)
        bboxes, _ = self.face_det_model.detect_faces(img, 0.55, input_size = (256, 256))  # srcfd 阈值比较低
        bboxes = bboxes[0]  # x1, y1, x2, y2, scores  # 无脸 TODO 
        # bboxes 
        img_face_crop = img[int(bboxes[1]):int(bboxes[3]), int(bboxes[0]):int(bboxes[2]), :]
        # cv2.imwrite("face_crop_256.png", img_face_crop)
        keypoints = landmark_98_to_68(self.facelandmark_infer_onnx(img_face_crop))  # FAN 网络 检测人脸关键点 98

        # keypoints to the original location
        keypoints[:,0] += int(bboxes[0])
        keypoints[:,1] += int(bboxes[1])
        
        return keypoints
