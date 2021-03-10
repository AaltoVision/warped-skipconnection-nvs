# Novel View Synthesis via Depth-guided Skip Connections
Code for paper [Novel View Synthesis via Depth-guided Skip Connections](https://openaccess.thecvf.com/content/WACV2021/html/Hou_Novel_View_Synthesis_via_Depth-Guided_Skip_Connections_WACV_2021_paper.html)
```
@InProceedings{Hou_2021_WACV,
    author    = {Hou, Yuxin and Solin, Arno and Kannala, Juho},
    title     = {Novel View Synthesis via Depth-Guided Skip Connections},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2021},
    pages     = {3119-3128}
}
```

# Data
Download the dataset from the [Google Drive](https://drive.google.com/drive/folders/1YbgU-JOXYsGi7yTrYb1F3niXj6nZp4Li) provided by [1] and 
unzip the dataset under `./datasets/` folder.

Download the evaluation list files from the [Google Drive](https://drive.google.com/drive/folders/1JmyCvT7pvtZ3k7aOoVnLcTetwgi-vWM-?usp=sharing). Put the corresponding file under corresponding dataset folder. E.g. `./datasets/dataset_kitti/eval_kitti.txt`.

# Training
```
python train.py\
    --name chair\
    --category chair\
    --niter 2000\
    --niter_decay 2000\
    --save_epoch_freq 100\
    --random_elevation\
    --lr 1e-4
```
If you don't want to view the real-time results, you can add command `--display_id 0`

If you want to view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097. 

# Testing
Download our pre-trained model from our [Google Drive](https://drive.google.com/file/d/1xRTCSfNb40oRT6QMfchOIlh9QFIpkUmc/view?usp=sharing)
To evaluate the performance, run 
```
python eval.py\
    --name chair\
    --category chair\
    --checkpoints_dir checkpoints\ 
    --which_epoch best
```

# Acknowledgments
The code is based on the source code of the paper:

[1] Chen, Xu and Song, Jie and Hilliges, Otmar (2019). **Monocular Neural Image-based Rendering with Continuous View Control**. In: *International Conference on Computer Vision (ICCV)*. (https://github.com/xuchen-ethz/continuous_view_synthesis), 
