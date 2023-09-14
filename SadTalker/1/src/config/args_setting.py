from argparse import ArgumentParser

def arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--result_dir", default='/models/SadTalker/1/results', help="path to output")
    parser.add_argument("--pose_style", type=int, default=0,  help="input pose style from [0, 46)")
    parser.add_argument("--batch_size", type=int, default=1,  help="the batch size of facerender")
    parser.add_argument("--size", type=int, default=256,  help="the image size of the facerender")
    parser.add_argument("--expression_scale", type=float, default=1.,  help="the batch size of facerender")
    parser.add_argument('--levels', type=list, default=[320, 480], help='resize the input image for sepeedup')
    parser.add_argument('--fps', type=int, default=25, help='fps of video and audio for sample')
    parser.add_argument('--save_flag', type=bool, default=False, help='save some outputs during pipeline')
    parser.add_argument('--debug', type=bool, default=False, help='debug the pipeline, will be repeat with save_flag')
    parser.add_argument('--test_local', type=bool, default=True, help='test the pipeline in local')
    
    args = parser.parse_args()
    return args