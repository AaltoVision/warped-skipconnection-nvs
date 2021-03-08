import random
import torch
import numpy as np
import torch.utils.data as data
import os
from PIL import Image
from scipy.spatial.transform import Rotation as ROT


class ShapeNetDataLoader(data.Dataset):
    def __init__(self):
        super(ShapeNetDataLoader, self).__init__()

    def initialize(self, opt):
        self.opt = opt
        self.view_per_model = 54
        self.category = opt.category
        self.dataroot = './datasets/dataset_%s'% (opt.category)


        with open(os.path.join(self.dataroot, 'id_train.txt'), 'r') as fp:
            self.ids_train = [s.strip() for s in fp.readlines() if s]
            
        with open(os.path.join(self.dataroot, 'eval_%s_40.txt' % (opt.category)), 'r') as fp:
            self.tuples_test = [s.strip().split(' ') for s in fp.readlines() if s]


        if opt.isTrain:
            self.ids = self.ids_train
        else:
            self.ids = self.tuples_test
            
        self.dataset_size = len(self.ids)
        #self.opt.bound = 2
        self.opt.bound = 4

    def __getitem__(self, index):
        if self.opt.isTrain:
            idx_model = self.ids[index]
            elev_a = 20 if not self.opt.random_elevation else random.randint(0, 2)*10
            azim_a = random.randint(0, 17)*20
            id_a = '%s_%d_%d' %(idx_model, azim_a, elev_a)

            elev_b = 20 if not self.opt.random_elevation else random.randint(0, 2)*10
            #delta = random.choice([20 * x for x in range(-self.opt.bound, self.opt.bound+1) if x != 0] )
            delta = random.choice([10 * x for x in range(-self.opt.bound, self.opt.bound + 1) if x != 0])

            azim_b = (azim_a + delta)%360
            id_b = '%s_%d_%d' %(idx_model, azim_b, elev_b)

            A = self.load_image(id_a) / 255. * 2 - 1
            B = self.load_image(id_b) / 255. * 2 - 1

            A = torch.from_numpy(A.astype(np.float32)).permute((2, 0, 1))
            B = torch.from_numpy(B.astype(np.float32)).permute((2, 0, 1))

            T = np.array([0, 0, 2]).reshape((3, 1))

            RA = ROT.from_euler('xyz', [-elev_a, azim_a, 0],degrees=True).as_dcm()
            RB = ROT.from_euler('xyz', [-elev_b, azim_b, 0],degrees=True).as_dcm()
            R = RA.T @ RB

            T = -R.dot(T) + T
            mat = np.block([[R               , T],
                            [np.zeros((1, 3)), 1]])


            mat_A = np.block([[RA.T, T],
                              [np.zeros((1, 3)), 1]])

            mat_B = np.block([[RB.T, T],
                              [np.zeros((1, 3)), 1]])



            dict = {'A': A, 'B': B, 'RT': mat.astype(np.float32),
                    'RT_A': mat_A.astype(np.float32), 'RT_B': mat_B.astype(np.float32),
                    }

            return dict

        else:
            id_a, id_b = self.ids[index]

            _, azim_a, elev_a = id_a.split('_')
            _, azim_b, elev_b = id_b.split('_')

            azim_a, elev_a = int(azim_a), int(elev_a)
            azim_b, elev_b = int(azim_b), int(elev_b)

            A = self.load_image(id_a) / 255. * 2 - 1
            B = self.load_image(id_b) / 255. * 2 - 1

            A = torch.from_numpy(A.astype(np.float32)).permute((2, 0, 1))
            B = torch.from_numpy(B.astype(np.float32)).permute((2, 0, 1))

            T = np.array([0, 0, 2]).reshape((3, 1))

            RA = ROT.from_euler('xyz', [-elev_a, azim_a, 0],degrees=True).as_dcm()
            RB = ROT.from_euler('xyz', [-elev_b, azim_b, 0],degrees=True).as_dcm()

            
            R = RA.T @ RB
            T = -R.dot(T) + T

            mat = np.block([[R, T],
                            [np.zeros((1, 3)), 1]])

            mat_A = np.block([[RA.T, T],
                            [np.zeros((1, 3)), 1]])

            mat_B = np.block([[RB.T, T],
                            [np.zeros((1, 3)), 1]])


            dict = {'A': A, 'B': B, 'RT': mat.astype(np.float32),
                    'RT_A': mat_A.astype(np.float32), 'RT_B':mat_B.astype(np.float32),
                    'id_a':id_a, 'id_b':id_b,
                    }

            return dict
        
    def __len__(self):
        return self.dataset_size

    def name(self):
        return 'ShapeNetDataLoader'

    def load_image(self, id):
        image_path = os.path.join(self.dataroot, 'images', id + '.png')
        pil_image = Image.open(image_path)
        image = np.asarray(pil_image.convert('RGB'))
        pil_image.close()
        return image
