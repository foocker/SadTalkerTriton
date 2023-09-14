import cv2
import librosa
import subprocess
import numpy as np
import soundfile as sf
import tritonclient.grpc, tritonclient.http
import tritonclient.grpc.model_config_pb2 as model_config
from tritonclient.utils import  np_to_triton_dtype


triton_type_to_np_dtype = {
    'TYPE_BOOL': np.bool_,
    'TYPE_INT8': np.int8,
    'TYPE_INT16': np.int16,
    'TYPE_INT32': np.int32,
    'TYPE_INT64': np.int64,
    'TYPE_UINT8': np.uint8,
    'TYPE_FP16': np.float16,
    'TYPE_FP32': np.float32,
    'TYPE_FP64': np.float64,
    'TYPE_STRING': np.object_
}


class AudioSegment(object):
    """Monaural audio segment abstraction.
    :param samples: Audio samples [num_samples x num_channels].
    :type samples: ndarray.float32
    :param sample_rate: Audio sample rate.
    :type sample_rate: int
    :raises TypeError: If the sample data type is not float or int.
    """

    def __init__(self, samples, sample_rate, target_sr=16000, trim=False,
                 trim_db=60):
        """Create audio segment from samples.
        Samples are convert float32 internally, with int scaled to [-1, 1].
        """
        samples = self._convert_samples_to_float32(samples)
        self._samples = samples
        self._sample_rate = sample_rate
        if self._samples.ndim >= 2:
            self._samples = np.mean(self._samples, 1)

    @staticmethod
    def _convert_samples_to_float32(samples):
        """Convert sample type to float32.
        Audio sample type is usually integer or float-point.
        Integers will be scaled to [-1, 1] in float32.
        """
        float32_samples = samples.astype('float32')
        if samples.dtype in np.sctypes['int']:
            bits = np.iinfo(samples.dtype).bits
            float32_samples *= (1. / 2 ** (bits - 1))
        elif samples.dtype in np.sctypes['float']:
            pass
        else:
            raise TypeError("Unsupported sample type: %s." % samples.dtype)
        return float32_samples

    @classmethod
    def from_file(cls, filename, target_sr=16000, int_values=False, offset=0,
                  duration=0, trim=False):
        """
        Load a file supported by librosa and return as an AudioSegment.
        :param filename: path of file to load
        :param target_sr: the desired sample rate
        :param int_values: if true, load samples as 32-bit integers
        :param offset: offset in seconds when loading audio
        :param duration: duration in seconds when loading audio
        :return: numpy array of samples
        """
        with sf.SoundFile(filename, 'r') as f:
            dtype = 'int32' if int_values else 'float32'
            sample_rate = f.samplerate
            if offset > 0:
                f.seek(int(offset * sample_rate))
            if duration > 0:
                samples = f.read(int(duration * sample_rate), dtype=dtype)
            else:
                samples = f.read(dtype=dtype)

        samples = samples.transpose()
        return cls(samples, sample_rate, target_sr=target_sr, trim=trim)

    @property
    def samples(self):
        return self._samples.copy()

    @property
    def sample_rate(self):
        return self._sample_rate


class SpeechImageClient(object):

    def __init__(self, url, protocol, model_name, model_version, batch_size,
                 model_platform=None, verbose=False, mode="batch"):
        
        self.model_name = model_name
        self.model_version = model_version
        self.verbose = verbose
        self.batch_size = batch_size
        self.transpose_audio_features = False
        self.grpc_stub = None
        self.ctx = None
        self.correlation_id = 0
        self.first_run = True
        if mode == "streaming" or mode == "asynchronous":
            self.correlation_id = 1

        self.buffer = []

        if protocol == "grpc":
            # Create gRPC client for communicating with the server
            self.prtcl_client = tritonclient.grpc
        else:
            # Create HTTP client for communicating with the server
            self.prtcl_client = tritonclient.http

        self.triton_client = self.prtcl_client.InferenceServerClient(
            url=url, verbose=self.verbose)

        self.source_image_name, self.driven_audio_name,  \
            self.frame_sizes_name, self.source_image_type, \
            self.driven_audio_type, self.frame_sizes_type = \
        self.parse_model(model_name,batch_size, model_platform, verbose)


    def check_num_samples(self, num_samples, model_name):
        if num_samples['data_type'] != 'TYPE_UINT32' and num_samples['data_type'] != 'TYPE_INT32':
             raise Exception(
                    "expecting num_samples datatype to be TYPE_UINT32/TYPE_INT32, "
                    "model '" + model_name + "' output type is " +
                    model_config.DataType.Name(num_samples['data_type']))
        if len(num_samples['dims']) != 1:
            raise Exception("Expecting num_samples to have 1 dimension, "
                            "model '{}' num_samples has {}".format(
                                model_name,len(num_samples['dims'])))

    def parse_model(self,
                    model_name, batch_size,
                    model_platform=None, verbose=False):
        """
        Check the configuration of the BLS model
        Inputs are:
          1) source_image: -1,-1,3
          2) driven_audio: bytes, -1

        Outputs are:
          1) encoded_frames:        frames from video 
          2) frame_sizes:        every frames size for bytes to read (chunk)
        """

        if self.prtcl_client is tritonclient.grpc:
            config = self.triton_client.get_model_config(model_name, as_json=True)
            config = config['config']
        else:
            config = self.triton_client.get_model_config(model_name)

        self.model_platform = model_platform

        source_image = config['input'][0]
        driven_audio = config['input'][1]
        frame_sizes = config['output'][0]

        return (source_image['name'],
                driven_audio['name'],
                frame_sizes['name'],
                triton_type_to_np_dtype[source_image['data_type']],
                triton_type_to_np_dtype[driven_audio['data_type']],
                triton_type_to_np_dtype[frame_sizes['data_type']],
        )
        
    def encode_image(self, img):
        pass
    def encode_video(self, frames):
        '''
        frames = []
        cap = cv2.VideoCapture(video_file)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        '''
        encoded_frames  = []
        frame_sizes = []
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        for frame in frames:
            result, enc_frame = cv2.imencode('.jpg', frame, encode_param)
            if result:
                size = len(enc_frame)
                encoded_frames.append(enc_frame)
                frame_sizes.append(size)
        frame_sizes_arr = np.array(frame_sizes, dtype=np.int32)
        encoded_frames_arr = b''.join(encoded_frames)
                
        return frame_sizes_arr, encoded_frames_arr
    
    def decode_video_with_save(self, outputs, fps=25, save_name="./results/final_video.mp4"):
        frame_sizes = outputs[0]
        if frame_sizes.dtype != np.int32:
            frame_sizes = np.frombuffer(outputs[0], dtype=np.int32)
        encoded_frames = outputs[1]
        frame_cursor = 0
        
        for size in frame_sizes:
            frame_data = encoded_frames[frame_cursor:frame_cursor+size]
            frame = cv2.imdecode(np.frombuffer(frame_data, dtype='uint8'), cv2.IMREAD_COLOR) 
            h, w = frame.shape[:2]
            break
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 写->音频和视频的合并，ffmpeg-python 没查。
        out = cv2.VideoWriter(save_name, fourcc, fps, (w, h))
        
        for size in frame_sizes:
            frame_data = encoded_frames[frame_cursor:frame_cursor+size]
            frame = cv2.imdecode(np.frombuffer(frame_data, dtype='uint8'), cv2.IMREAD_COLOR)
            frame_cursor += size
            out.write(frame)
        out.release()
        
    def merge_video_with_audio(self, video, audio, save_path):
        cmd = r'ffmpeg -y -hide_banner -loglevel error -i "%s" -i "%s" -vcodec copy "%s"' % (video, audio, save_path)
        subprocess.call(cmd, shell=True)
        
    def encode_audio(self, audio_file, sr=16000):
        '''
        仅读取
        '''
        return librosa.core.load(audio_file, sr=sr, dtype=np.float32)[0]
    
    def decode_audio_with_save(self, audio_input_array, path='./results/pad_audio.wav', sr=16000, fps=25):
        '''
        '''
        def parse_audio_length(audio_length, sr, fps):
            bit_per_frames = sr / fps

            num_frames = int(audio_length / bit_per_frames)
            audio_length = int(num_frames * bit_per_frames)

            return audio_length, num_frames
        
        def crop_pad_audio(wav, audio_length):
            if len(wav) > audio_length:
                wav = wav[:audio_length]
            elif len(wav) < audio_length:
                wav = np.pad(wav, [0, audio_length - len(wav)], mode='constant', constant_values=0)
            return wav

        wav_length, num_frames = parse_audio_length(len(audio_input_array), sr, fps)
        wav = crop_pad_audio(wav, wav_length)
        
        sf.write(path, wav, sr)

    def prepare_inputs(self, img_file, audio_file, sr=16000):
        '''
        最原始的输入,其他处理交给服务端
        '''
        img = cv2.imread(img_file)
        wav = librosa.core.load(audio_file, sr=sr, dtype=np.float32)[0]
        return img, wav
    
    
    def render(self, image, audio_input_array, fps=25, sr=16000):
        inputs_image_audio = []
        inputs_image_audio.append(self.prtcl_client.InferInput(self.source_image_name,
                                                   image.shape,
                                                   np_to_triton_dtype(image.dtype)))
        inputs_image_audio.append(self.prtcl_client.InferInput(self.driven_audio_name,
                                                   audio_input_array.shape,
                                                   np_to_triton_dtype(self.driven_audio_type)))

        if self.prtcl_client is tritonclient.grpc:
            inputs_image_audio[0].set_data_from_numpy(image)
            inputs_image_audio[1].set_data_from_numpy(audio_input_array)
        else: # http
            inputs_image_audio[0].set_data_from_numpy(image, binary_data=True)
            inputs_image_audio[1].set_data_from_numpy(audio_input_array, binary_data=True)

        outputs = []
        if self.prtcl_client is tritonclient.grpc:
            outputs.append(self.prtcl_client.InferRequestedOutput(self.frame_sizes_name))
        else:
            outputs.append(self.prtcl_client.InferRequestedOutput(self.frame_sizes_name, binary_data=True))

        triton_result = self.triton_client.infer(self.model_name, inputs=inputs_image_audio, outputs=outputs)
        frame_sizes = triton_result.as_numpy(self.frame_sizes_name)
        print(frame_sizes)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()  
    parser.add_argument("--driven_audio", default='./examples/driven_audio/bus_chinese.wav', help="path to driven audio")
    parser.add_argument("--source_image", default='./examples/source_image/2.jpg', help="path to source image")
    parser.add_argument("--result_dir", default='./results', help="path to output")
    parser.add_argument("--pose_style", type=int, default=0,  help="input pose style from [0, 46)")
    parser.add_argument("--batch_size", type=int, default=1,  help="the batch size of facerender")
    parser.add_argument("--size", type=int, default=256,  help="the image size of the facerender")
    parser.add_argument("--expression_scale", type=float, default=1.,  help="the batch size of facerender")
    parser.add_argument("--cpu", dest="cpu", action="store_true") 
    parser.add_argument('--levels', type=list, default=[480,600], help='resize the input image for sepeedup')
    parser.add_argument('--fps', type=int, default=25, help='fps of video and audio for sample')
    args = parser.parse_args()
    
    SIC = SpeechImageClient("localhost:8001", "grpc", model_name='SadTalker', model_version=1, batch_size=2, verbose=False)
    img, audio_wav = SIC.prepare_inputs(args.source_image, args.driven_audio)
    SIC.render(img, audio_wav, fps=args.fps)
    