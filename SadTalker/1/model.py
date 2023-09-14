import cv2
import copy
import torch
from src.audio2coeff_infer import Audio2Coeff  
from src.config.args_setting import arg_parser
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_first_coeff_audio_mel
from src.generate_facerender_batch import get_facerender_data
from src.utils.preprocess import CropAndExtract, resize_by_level

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.device = 'cuda' if torch.cuda.is_available() else "cpu"
        args_ = arg_parser()
        self.base_dir = "/models/SadTalker/1"
        self.batch_size = args_.batch_size
        self.expression_scale = args_.expression_scale
        self.size = args_.size
        self.fps = args_.fps
        self.result_dir = args_.result_dir
        self.levels = args_.levels
        self.pose_style = args_.pose_style
        self.debug = args_.debug
        self.save_flag = args_.save_flag
        self.test_local = args_.test_local
        
        self.preprocess_model = CropAndExtract(self.device)
        self.audio_to_coeff = Audio2Coeff(self.device)
        self.animate_from_coeff = AnimateFromCoeff()
        self.output_frame_sizes_type = pb_utils.triton_string_to_numpy("TYPE_INT32")
        self.output_encoded_frames_type = pb_utils.triton_string_to_numpy("TYPE_STRING")  # triton_string_to_numpy

    def execute(self, requests):
        responses = []
        for request in requests:
            source_image = pb_utils.get_input_tensor_by_name(request, 'source_image').as_numpy()
            pic_input = resize_by_level(source_image, self.levels)
            pic_input_rgb = cv2.cvtColor(pic_input, cv2.COLOR_BGR2RGB)
            pic_input_cp = copy.deepcopy(pic_input)

            # crop image and extract 3dmm from image
            print('3DMM Extraction for source image')
            first_coeff_dict, crop_pic, crop_info =  self.preprocess_model.generate(pic_input_rgb, pic_size=256, save_flag=self.save_flag, 
                                                                               save_dir=self.result_dir, debug=self.debug)
            
            driven_audio = pb_utils.get_input_tensor_by_name(request, 'driven_audio').as_numpy()
            
            driver_inputs_dict = get_first_coeff_audio_mel(first_coeff_dict, driven_audio, self.device, fps=self.fps)
            coeffs_pred_dict = self.audio_to_coeff.generate(driver_inputs_dict, self.pose_style, coeff_save_dir=self.result_dir, save_flag=self.save_flag)
            data_pred_forfacerender = get_facerender_data(coeffs_pred_dict, crop_pic, first_coeff_dict, 
                                        self.batch_size, expression_scale=self.expression_scale, save_dir=self.result_dir, save_flag=self.save_flag)
            
            original_wav = driver_inputs_dict['original_wav']

            outputs = self.animate_from_coeff.generate(data_pred_forfacerender, pic_input_cp, crop_info, original_wav,
                                            img_size=self.size,fps=self.fps, test_local=self.test_local)
            if outputs is not None:
                print("outputs is successed!")
                tensor_output = [pb_utils.Tensor("frame_sizes", outputs[0].astype(self.output_frame_sizes_type))]
                responses.append(pb_utils.InferenceResponse(output_tensors=tensor_output))
            else:
                print("outputs failed ")
                responses.append(pb_utils.InferenceResponse(output_tensor=[], \
                    error=pb_utils.TritonError("you didn't crop the image please see in past_pic_func")))
        
        return responses

    def finalize(self):
        print("Cleaning up")
        """ model is being unloaded"""