import time
from options.train_options import TrainOptions
from data.custom_dataset_data_loader import CustomDatasetDataLoader
from util.visualizer import Visualizer
import copy
from tqdm import tqdm
import numpy as np
import torch
from models.base_model import BaseModel


torch.manual_seed(0)

opt = TrainOptions().parse()

data_loader = CustomDatasetDataLoader(opt)
dataset = data_loader.load_data()


opt_for_eval = copy.deepcopy(opt)
opt_for_eval.isTrain = False
opt_for_eval.max_dataset_size = 1000
val_loader = CustomDatasetDataLoader(opt_for_eval)
valset = val_loader.load_data()

dataset_size = len(data_loader)
print('#training samples = %d' % dataset_size)

model = BaseModel(opt)

visualizer = Visualizer(opt)
total_steps = 0

best_l1 = np.inf




for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0

    iter_start_time = 0

    for i, data in enumerate(dataset):

        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        model.set_input(data)
        model.forward()
        

        if total_steps % opt.display_freq == 0:
            save_result = total_steps % opt.update_html_freq == 0
            visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

        model.optimize_parameters()

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        iter_start_time = time.time()



    if epoch % 100 == 0:
        print('saving the model (epoch %d, total_steps %d)' % (epoch, total_steps))
        model.save(epoch)

        model.switch_mode('eval')

        l1s = []

        for i, data in enumerate(tqdm(valset)):
            model.set_input(data)
            model.forward()
            res = model.evaluate()
            l1s.append(res['L1'])
        print('eval metric: L1 %f' %(np.mean(l1s)))
        if np.mean(l1s) < best_l1:
            model.save('best')
            message = 'Update best checkpoint at end of epoch %d' % epoch
            with open(visualizer.log_name, "a") as log_file:
                log_file.write('%s l1: %f\n' % (message, np.mean(l1s)))
            print(message)
            best_l1 = np.mean(l1s)

        model.switch_mode('train')


    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    model.update_learning_rate()


