# LearningToCompare
PyTorch code for CVPR 2018 paper: [Learning to Compare: Relation Network for Few-Shot Learning](https://arxiv.org/abs/1711.06025) (Few-Shot)

Hyper-parameters and input-size(84 by 84) remain the same as the original MINIImageNet settings. You may further tune the hyper-params to get better performance.

# Requirements
Tested on
- Python 3.7
- PyTorch 1.5.1

Other versions **should** work, but not tested.

# Data
```
./data
  └─skin-lesions-84
    ├─test
    │  ├─melanoma
    │  ├─nevus
    │  └─seborrheic_keratosis
    ├─train
    │  ├─melanoma
    │  ├─nevus
    │  └─seborrheic_keratosis
    └─valid
        ├─melanoma
        ├─nevus
        └─seborrheic_keratosis
```

# Train and Test

3 way 1 shot:

- Train: `python skin_train_one_shot.py -w 3 -s 1 -b 19`
- Test: `python skin_test_one_shot.py -w 3 -s 1`
- Test Acc: 0.620(Avg), 0.629(Best)

3 way 3 shot:
- Train: `python skin_train_few_shot.py -w 3 -s 3 -b 19`
- Test: `python skin_test_one_shot.py -w 3 -s 3`
- Test Acc: 0.651(Avg), 0.657(Best)

3 way 5 shot:
- Train: `python skin_train_few_shot.py -w 3 -s 5 -b 19`
- Test: `python skin_test_few_shot.py -w 3 -s 5`
- Test Acc: 0.650(Avg), 0.661(Best)

Test results may vary (+/- 0.003) on each run, depending on the random seed, CUDA environment, etc.

You can change -b parameter based on your GPU memory. Currently It will load my trained model, if you want to train from scratch, you can delete models by yourself.

Note that training will always **OVERRIDE** the pretrained models! Please backup the models on your need.

# Parameters
```
"-f", "--feature_dim", type=int, default=64
"-r", "--relation_dim", type=int, default=8
"-w", "--class_num", type=int, default=5
"-s", "--sample_num_per_class", type=int, default=1
"-b", "--batch_num_per_class", type=int, default=10
"-e", "--episode", type=int, default=10
"-t", "--test_episode", type=int, default=600
"-l", "--learning_rate", type=float, default=0.001
"-g", "--gpu", type=int, default=0
"-u", "--hidden_unit", type=int, default=10
```

## Citing

If you use this code in your research, please use the following BibTeX entry.

```
@inproceedings{sung2018learning,
  title={Learning to Compare: Relation Network for Few-Shot Learning},
  author={Sung, Flood and Yang, Yongxin and Zhang, Li and Xiang, Tao and Torr, Philip HS and Hospedales, Timothy M},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2018}
}
```

## Reference

[MAML](https://github.com/cbfinn/maml)

[MAML-pytorch](https://github.com/katerakelly/pytorch-maml)


