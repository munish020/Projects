from configobj import ConfigObj
import numpy as np

def config_reader():
    config = ConfigObj('config')

    Munish = config['Munish']
    model_id = Munish['modelID']
    model = config['models'][model_id]
    model['boxsize'] = int(model['boxsize'])
    model['stride'] = int(model['stride'])
    model['padValue'] = int(model['padValue'])
    Munish['octave'] = int(Munish['octave'])
    Munish['use_gpu'] = int(Munish['use_gpu'])
    Munish['starting_range'] = float(Munish['starting_range'])
    Munish['ending_range'] = float(Munish['ending_range'])
    Munish['scale_search'] = map(float, Munish['scale_search'])
    Munish['thre1'] = float(Munish['thre1'])
    Munish['thre2'] = float(Munish['thre2'])
    Munish['thre3'] = float(Munish['thre3'])
    Munish['mid_num'] = int(Munish['mid_num'])
    Munish['min_num'] = int(Munish['min_num'])
    Munish['crop_ratio'] = float(Munish['crop_ratio'])
    Munish['bbox_ratio'] = float(Munish['bbox_ratio'])
    Munish['GPUdeviceNumber'] = int(Munish['GPUdeviceNumber'])
    return Munish, model

if __name__ == "__main__":
    config_reader()
