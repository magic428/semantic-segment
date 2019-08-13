import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'

import sys
sys.path.insert(0, "/home/tdmc/work/gitwork/dl_ai/dl_framework/segment/pytorch-semantic-segmentation")

import datetime
from math import sqrt
import numpy as np
import torchvision.utils as vutils
import torchvision.transforms as standard_transforms
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import utils.joint_transforms as joint_transforms
import utils.transforms as extended_transforms
from datasets import cityscapes
from models import *
from utils import check_mkdir, evaluate, AverageMeter, CrossEntropyLoss2d

from PIL import Image

'''
可能更改的参数是:

test-batch-size
train-batch-size
snapshot
break
val_freq
'''

# args = {
#     'train_batch_size': 4,
#     'lr': 1e-2 / sqrt(16 / 2),
#     'lr_decay': 0.9,
#     'max_iter': 2e5,
#     'longer_size': 2048,
#     'weight_decay': 1e-4,
#     'momentum': 0.9,
#     # 'snapshot': '',
#     'snapshot': 'epoch_2_iter_456_loss_0.52281_acc_0.86247_acc-cls_0.48824_mean-iu_0.40518_fwavacc_0.76335_lr_0.0034931145.pth',
#     'print_freq': 10,
    
#     'val_img_sample_rate': 0.01,  # randomly sample some validation results to display,
#     'val_img_display_size': 384,
#     'val_freq': 600
# }


ckpt_path = '../../ckpt'

args = {
    'exp_name': 'cityscapes_fine-psp_net',
    'val_save_to_img_file': True,
    'crop_size': 300,
    'stride_rate': 2 / 3.,
    'longer_size': 2048,
    # 'snapshot': 'epoch_114_iter_41_loss_0.24338_acc_0.94690_acc-cls_0.69728_mean-iu_0.63821_fwavacc_0.90151_lr_0.0003090566.pth'
    'snapshot': 'epoch_100_iter_243_loss_0.23982_acc_0.94762_acc-cls_0.69750_mean-iu_0.63494_fwavacc_0.90304_lr_0.0007554825.pth'
}


def test_(img_path):
    net = PSPNet(num_classes=cityscapes.num_classes)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0:  [30, xxx] -> [15, ...], [15, ...] on 2 GPUs
        net = nn.DataParallel(net)

    print('load model ' + args['snapshot'])
    net.load_state_dict(torch.load(os.path.join(ckpt_path, args['exp_name'], args['snapshot'])))
    net.eval()

    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    val_input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    sliding_crop = joint_transforms.SlidingCrop(args['crop_size'], args['stride_rate'], cityscapes.ignore_label)

    img = Image.open(img_path).convert('RGB') 
    img_slices, _, slices_info = sliding_crop(img, img.copy())
    img_slices = [val_input_transform(e) for e in img_slices]
    img = torch.stack(img_slices, 0) 

    img = Variable(img, volatile=True).cuda()
    torch.no_grad()
    output = net(img)

    # prediction = output.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()
    # prediction = voc.colorize_mask(prediction)
    # prediction.save(os.path.join(ckpt_path, args['exp_name'], 'test', img_name + '.png'))

    img.transpose_(0, 1)
    slices_info.squeeze_(0)

    count = torch.zeros(args['longer_size'] // 2, args['longer_size']).cuda()
    output = torch.zeros(cityscapes.num_classes, args['longer_size'] // 2, args['longer_size']).cuda()

    slice_batch_pixel_size = img.size(1) * img.size(3) * img.size(4)
    prediction = np.zeros((args['longer_size'] // 2, args['longer_size']), dtype=int)

    for input_slice, info in zip(img, slices_info):
        input_slice = Variable(input_slice).cuda()
        output_slice = net(input_slice)
        assert output_slice.size()[1] == cityscapes.num_classes
        output[:, info[0]: info[1], info[2]: info[3]] += output_slice[0, :, :info[4], :info[5]].data
        count[info[0]: info[1], info[2]: info[3]] += 1

    output /= count
    prediction[:, :] = output.max(0)[1].squeeze_(0).cpu().numpy()

    test_dir = os.path.join(ckpt_path, args['exp_name'], 'test')
    img_name = os.path.basename(img_path)
    img_name = os.path.splitext(img_name)[0]
    print(img_name)
    if train_args['val_save_to_img_file']:
        check_mkdir(test_dir)

    val_visual = []
    prediction_pil = cityscapes.colorize_mask(prediction)
    if train_args['val_save_to_img_file']:
        prediction_pil.save(os.path.join(test_dir, '%s_prediction.png' % img_name))
    
def test(img_path):
    net = PSPNet(num_classes=cityscapes.num_classes) 
    if torch.cuda.is_available():
        net = nn.DataParallel(net)   # 添加了该句后, 就能正常导入在多GPU上训练的模型参数了
    print('loading model ' + args['snapshot'] + '...')
    model_state_path = os.path.join(ckpt_path, args['exp_name'], args['snapshot'])
    net.load_state_dict(torch.load(model_state_path))
    net.cuda()
    net.eval()


    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    val_input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])

    img = Image.open(img_path).convert('RGB') 
    img = val_input_transform(img)

    img.unsqueeze_(0)

    with torch.no_grad():
        img = Variable(img).cuda()
        output = net(img)
        # prediction = output.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()
        # prediction = voc.colorize_mask(prediction)
        # prediction.save(os.path.join(ckpt_path, args['exp_name'], 'test', img_name + '.png'))
        prediction = output.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()
    
    test_dir = os.path.join(ckpt_path, args['exp_name'], 'test')
    img_name = os.path.basename(img_path)
    img_name = os.path.splitext(img_name)[0]
    print(img_name)
    if args['val_save_to_img_file']:
        check_mkdir(test_dir)

    predictions_pil = cityscapes.colorize_mask(prediction)
    if args['val_save_to_img_file']:
        predictions_pil.save(os.path.join(test_dir, '%s_prediction.png' % img_name))

if __name__ == '__main__':
    img_path = '/home/tdmc/data/segmentation/yjl-coal-stone/YJL_201906031132069.jpg'
    test(img_path)
