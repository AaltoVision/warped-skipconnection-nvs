from options.test_options import TestOptions
from data.custom_dataset_data_loader import CustomDatasetDataLoader
from tqdm import tqdm
import numpy as np
import torch
from models.base_model import BaseModel


opt = TestOptions().parse()
opt.nThreads = 1
opt.batchSize = 1
opt.serial_batches = True
opt.no_flip = True
opt.isTrain = False
opt.max_dataset_size = float("inf")

data_loader = CustomDatasetDataLoader(opt)
dataset = data_loader.load_data()

model = BaseModel(opt)

L1s = []
SSIMs = []
with torch.no_grad():
    for idx, data in enumerate(tqdm(dataset)):
        ida = data['id_a'][0].split('_')
        idb = data['id_b'][0].split('_')

        assert (ida[0] == idb[0])
        model_id = ida[0]
        ida = '_'.join(ida[1:])
        idb = '_'.join(idb[1:])

        model.set_input(data)

        model.switch_mode('eval')

        model.anim_dict = {'vis': []}
        model.real_A = model.real_A[:1]
        model.real_B = model.real_B[:1]


        eval_res = model.evaluate()
        L1s.append(eval_res['L1'])
        SSIMs.append(eval_res['SSIM'])

print('L1:{l1}, SSIM:{ssim}'.format(l1=np.mean(L1s), ssim=np.mean(SSIMs)))

