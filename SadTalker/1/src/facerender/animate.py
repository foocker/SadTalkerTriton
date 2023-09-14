import os
import cv2
import torch
import numpy as np
import onnxruntime
# import src.utils.audio as audio
from src.utils.paste_pic import paste_pic_func
from src.facerender.modules.make_animation import make_animation 


class AnimateFromCoeff():

    def __init__(self):

        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

        generator = onnxruntime.InferenceSession('/models/SadTalker/1/onnx_weights/generator.onnx', sess_options, providers=[('TensorrtExecutionProvider', \
            {'trt_fp16_enable': True, 'trt_engine_cache_enable': True, 'trt_engine_cache_path': '/models/SadTalker/1/cache'}), 'CUDAExecutionProvider'])
        kp_extractor = onnxruntime.InferenceSession('/models/SadTalker/1/onnx_weights/kp_detector.onnx', sess_options, providers=[('TensorrtExecutionProvider', \
            {'trt_fp16_enable': True, 'trt_engine_cache_enable': True, 'trt_engine_cache_path': '/models/SadTalker/1/cache'}), 'CUDAExecutionProvider'])
        mapping = onnxruntime.InferenceSession('/models/SadTalker/1/onnx_weights/mapping.onnx', sess_options, providers=[('TensorrtExecutionProvider', \
            {'trt_fp16_enable': True, 'trt_engine_cache_enable': True, 'trt_engine_cache_path': '/models/SadTalker/1/cache'}), 'CUDAExecutionProvider'])

        self.kp_extractor = kp_extractor
        self.generator = generator
        self.mapping = mapping
        

    def generate(self, data_pred_forfacerender, pic_original, crop_info, original_wav, img_size=256, fps=25, **kwargs):
        """
            data_pred_forfacerender:  dict from get_facerender_data
                keys : source_image, source_semantics, frame_num, target_semantics_list
            video_save_dir : 视频保存目录
            pic_original : 用户传入的原始图片
            crop_info : 原始图片的人脸检测，关键点检测，对齐后的crop信息
        """

        source_image = data_pred_forfacerender['source_image'].type(torch.FloatTensor).data.numpy()
        source_semantics = data_pred_forfacerender['source_semantics'].type(torch.FloatTensor).data.numpy()
        target_semantics = data_pred_forfacerender['target_semantics_list'].type(torch.FloatTensor).data.numpy()

        frame_num = data_pred_forfacerender['frame_num']

        predictions_video = make_animation(source_image, source_semantics, target_semantics,
                                           self.generator, self.kp_extractor, self.mapping)

        predictions_video = predictions_video.reshape((-1,)+predictions_video.shape[2:])
        predictions_video = predictions_video[:frame_num]  # 生成所有帧,tensor

        video = []
        for idx in range(predictions_video.shape[0]):
            image = predictions_video[idx]  # 3, 256, 256
            image = np.transpose(image, [1, 2, 0]).astype(np.float32)
            image = (image*255).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            video.append(image)

        # the generated video is 256x256, so we keep the aspect ratio, 
        original_size = crop_info[0]
        if original_size:
            results = [cv2.resize(result_i, (img_size, int(img_size * original_size[1]/original_size[0]))) for result_i in video]
        else:
            results = video
            
        test_loacl = kwargs.get("test_local", False)
        # 泊松融合全身
        outputs = paste_pic_func(results, pic_original, crop_info, original_wav, fps=fps, test_local=test_loacl)

        return outputs
