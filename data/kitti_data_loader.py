import torch
import numpy as np
from scipy.spatial.transform import Rotation as ROT
import torch.utils.data as data
import os
import csv
import random
from PIL import Image
class KITTIDataLoader(data.Dataset):
    def __init__(self):
        super(KITTIDataLoader, self).__init__()

    def initialize(self, opt):
        self.opt = opt
        self.dataroot = './datasets/dataset_kitti'

        self.opt.bound = 5


        with open(os.path.join(self.dataroot, 'filtered_id_trains.txt' ), 'r') as fp:
            self.ids_train = [s.strip().split(' ') for s in fp.readlines() if s]

        with open(os.path.join(self.dataroot, 'eval_%s.txt' % (opt.category)), 'r') as fp:
            self.tuples_test = [s.strip().split(' ') for s in fp.readlines() if s]


        if opt.isTrain:
            self.ids = self.ids_train
            self.dataset_size = int(len(self.ids)) // (opt.bound * 2)
        else:
            self.ids = self.tuples_test
            self.dataset_size = int(len(self.ids))



        self.pose_dict = {}
        pose_path = os.path.join(self.dataroot, 'poses.txt')
        with open(pose_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=' ')
            for row in csv_reader:
                id = row[0]
                self.pose_dict[id] = []
                for col in row[1:-1]:
                    self.pose_dict[id].append(float(col))
                self.pose_dict[id] = np.array(self.pose_dict[id])

    def __getitem__(self, index):
        if self.opt.isTrain:
            id_target, id = self.ids[index]

            B = self.load_image(id) / 255. * 2 - 1
            B = torch.from_numpy(B.astype(np.float32)).permute((2,0,1))
            A = self.load_image(id_target) / 255. * 2 - 1
            A = torch.from_numpy(A.astype(np.float32)).permute((2,0,1))

            poseB = self.pose_dict[id]
            poseA = self.pose_dict[id_target]
            TB = poseB[3:].reshape(3, 1)
            RB = ROT.from_euler('xyz',poseB[0:3]).as_dcm()
            TA = poseA[3:].reshape(3, 1)
            RA = ROT.from_euler('xyz',poseA[0:3]).as_dcm()
            T = RA.T.dot(TB-TA)/50.

            mat = np.block(
                [ [RA.T@RB, T],
                  [np.zeros((1,3)), 1] ] )

            return {'A': A, 'B': B, 'RT': mat.astype(np.float32)}

        else:
            id_a, id_b = self.ids[index]
            B = self.load_image(id_b) / 255. * 2 - 1
            B = torch.from_numpy(B.astype(np.float32)).permute((2, 0, 1))
            A = self.load_image(id_a) / 255. * 2 - 1
            A = torch.from_numpy(A.astype(np.float32)).permute((2, 0, 1))

            poseB = self.pose_dict[id_b]
            poseA = self.pose_dict[id_a]
            TB = poseB[3:].reshape(3, 1)
            RB = ROT.from_euler('xyz', poseB[0:3]).as_dcm()
            TA = poseA[3:].reshape(3, 1)
            RA = ROT.from_euler('xyz', poseA[0:3]).as_dcm()
            T = RA.T.dot(TB - TA) / 50.

            mat = np.block(
                [[RA.T @ RB, T],
                 [np.zeros((1, 3)), 1]])

            return {'A': A, 'B': B, 'RT': mat.astype(np.float32),
                    'id_a':id_a, 'id_b':id_b}



    def load_image(self, id):
        image_path = os.path.join(self.dataroot, 'images', id + '.png')
        image = np.asarray(Image.open(image_path).convert('RGB'))
        return image

    def __len__(self):
        return self.dataset_size

    def name(self):
        return 'KITTIDataLoader'
