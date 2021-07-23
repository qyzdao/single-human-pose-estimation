import os, gzip, csv, torch, cv2, torchvision, random
import torch.nn as nn
import numpy as np
import scipy.ndimage as ndi
import scipy.misc
import imageio
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from collections import defaultdict

IMG_HEIGHT, IMG_WIDTH = 400, 400
patch_size = 128
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

cuda = True if torch.cuda.is_available() else False
print('cuda:', cuda)

resnet = torchvision.models.resnet34(pretrained=True)
for param in resnet.parameters():
    param.requires_grad = False
if cuda:
    resnet = resnet.cuda()
layer1 = resnet._modules.get('layer1')
layer2 = resnet._modules.get('layer2')
layer3 = resnet._modules.get('layer3')
layer4 = resnet._modules.get('layer4')
layerfc = resnet._modules.get('avgpool')

def save_images(images, size, image_path):
    return imsave(images, size, image_path)

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

def generate_animation(path, num):
    images = []
    for e in range(num):
        img_name = path + '_epoch%03d' % (e+1) + '.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave(path + '_generate_animation.gif', images, fps=5)

def loss_plot(hist, path = 'Train_hist.png', model_name = '', cut=5):
    names = [n for n in hist]
    x = range(len(hist[names[0]]))

    #y1 = hist[names[0]]
    #y2 = hist[names[1]]
    y = [hist[names[i]] for i in range(len(names))]

    for i in range(len(names)):
        plt.plot(x[cut:], y[i][cut:], label=names[i])
    #plt.plot(x, y1, label=names[0])
    #plt.plot(x, y2, label=names[1])

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(path, model_name + '.png')

    plt.savefig(path)

    plt.close()

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

def get_vector(images, layer, model, num_ftrs, size, Flatten=False):
    """
    Get feature vector from ResNet
    layer: from ResNet
    model: ResNet
    """
    embedding = torch.zeros(images.shape[0], num_ftrs, size[0], size[1])
    def copy_data(m, i, o):
        embedding.copy_(o.data)
    h = layer.register_forward_hook(copy_data)
    hx = model(images)
    h.remove()
    if Flatten:
        embedding = torch.flatten(embedding, 1)
    return embedding

def get_feature(img, layer, resnet, filter_size, feature_size, flatten=False):
    """
    Input: one 255 range BGR image read from cv2
    Output: the feature from ResNet
    """
    features = []
    img = np.rollaxis(img, 1, 4)
    for i in range(len(img)):
        fimg = cv2.resize(img[i], (224,224), cv2.INTER_AREA)
        #fimg = cv2.merge([fimg, fimg, fimg]).astype(np.float32) / 255. #(224,224,3)
        #fimg = cv2.cvtColor(fimg, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        fimg = (fimg - mean) / std
        fimg = np.rollaxis(fimg, 2, 0)
        fimg = torch.from_numpy(fimg).float().cuda().unsqueeze(0)
        features.append( get_vector(fimg, layer, resnet, filter_size,
            [feature_size,feature_size]).squeeze().cpu().numpy() )
    return np.array(features)

def plot_predict_joint(im, fake_grid, epoch, opt, root_dir, num=2):
    """
    plot predicted joints overlapping on the original art
    and save the images
    """
    grid = fake_grid.clone().detach().cpu()
    img = im.copy().astype(np.float32)

    for i in range(num):
        idx = torch.argsort(grid[i,...].flatten(), descending=True)[:opt.anchor]
        centers = []
        for index in idx:
            ii, jj = index // opt.img_size, index % opt.img_size
            centers.append([float(ii) / float(opt.img_size), float(jj) / float(opt.img_size)])

        centers = np.array(centers) * opt.art_size

        color = np.array([0.,0.,255.])
        for k in range(len(centers)):
            x, y = int(centers[k,0]), int(centers[k,1])
            img[i,x,y,:] = color
            if x+1<opt.art_size and x-1>0:
                img[i,x-1,y,:]=img[i,x+1,y,:] = color
                if y+1<opt.art_size and y-1>0:
                    img[i,x-1,y-1,:]=img[i,x,y-1,:]=img[i,x+1,y-1,:]=color
                    img[i,x-1,y+1,:]=img[i,x,y+1,:]=img[i,x+1,y+1,:]=color

    for i in range(num):
        image_name = str(epoch) + '_' + str(i) + '_' + 'joints' + '.png'
        cv2.imwrite(os.path.join(root_dir, image_name), img[i,...].astype(np.uint8))
    return img

def img_normalize(img_batch):
    for i in range(len(img_batch)):
        img_batch[i] = (img_batch[i] - np.min(img_batch[i])) / np.max(img_batch[i])
    return img_batch

def ada_thre(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img,5)
    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    return img

def data_augmentation(img_batch, heatmap_batch, seed):
    for i in range(len(img_batch)):
        img_batch[i] = random_transform(img_batch[i], seed + i)
        #maskart_batch[i] = random_transform(maskart_batch[i], seed + i)
        for j in range(len(heatmap_batch[i])):
            heatmap_batch[i,j,...] = random_transform(heatmap_batch[i,j,...], seed + i)
    return img_batch, heatmap_batch

def data_generator(is_train, opt, seed):
    coco1 = np.load('../../share/data/coco1.npy', allow_pickle=True)
    coco2 = np.load('../../share/data/coco2.npy', allow_pickle=True)

    # merge npy list of dict
    coco = np.concatenate((coco1, coco2))
    random.shuffle(coco) #shuffle
    print('merge coco1 and coco2 length:', len(coco))
    split = int(len(coco) * 0.8)

    if is_train:
        data = coco[:split]
    else:
        data = coco[split:]

    counts = 0
    while True:
        np.random.seed(seed + counts)
        idx = np.random.randint(0, len(data), size=opt.batch_size)
        img_batch_ = np.array([ cv2.cvtColor(data[i]['img'], cv2.COLOR_BGR2RGB) for i in idx]) / 255.
        img_batch_ = np.rollaxis(img_batch_, 3, 1)
        heatmap_batch_ = np.array([data[i]['confidence_map'] for i in idx])
        heatmap_batch_ = heatmap_batch_[:,:,None]

        # data augmentation
        img_batch, heatmap_batch = data_augmentation(
            img_batch_, heatmap_batch_, seed + counts)

        feature = (img_batch.squeeze() * 255.).astype(np.uint8)
        feature_batch = get_feature(feature, layer2, resnet, 128, 28)

        img_batch = img_batch * 2. - 1. # -1~1
        heatmap_batch = heatmap_batch * 2. - 1. # -1~1

        img_batch = torch.from_numpy(np.array(img_batch)).float().cuda()
        feature_batch = torch.from_numpy(np.array(feature_batch)).float().cuda()
        heatmap_batch = torch.from_numpy(np.array(heatmap_batch)).float().cuda().squeeze()

        counts += opt.batch_size

        yield img_batch, heatmap_batch, feature_batch


# Transform functions from Keras
def random_transform(x, seed=None, channel_first=True):
    """Randomly augment a single image tensor.
    # Arguments
        x: 3D tensor, single image.
        seed: random seed.
    # Returns
        A randomly transformed version of the input (same shape).
    """
    np.random.seed(seed)

    if channel_first:
        img_row_axis = 1
        img_col_axis = 2
        img_channel_axis = 0
        H, W = x.shape[1], x.shape[2]
    else:
        img_row_axis = 0
        img_col_axis = 1
        img_channel_axis = 2
        H, W = x.shape[0], x.shape[1]

    if float(np.random.uniform(0.0, 1.0, 1)) < 0.5:
        if_flip = True
    else:
        if_flip = False

    rotation_range = 10
    theta = np.deg2rad(np.random.uniform(-rotation_range, rotation_range))

    height_shift_range = width_shift_range = 0.1
    if height_shift_range:
        try:  # 1-D array-like or int
            tx = np.random.choice(height_shift_range)
            tx *= np.random.choice([-1, 1])
        except ValueError:  # floating point
            tx = np.random.uniform(-height_shift_range,
                                   height_shift_range)
        if np.max(height_shift_range) < 1:
            tx *= x.shape[img_row_axis]
    else:
        tx = 0

    if width_shift_range:
        try:  # 1-D array-like or int
            ty = np.random.choice(width_shift_range)
            ty *= np.random.choice([-1, 1])
        except ValueError:  # floating point
            ty = np.random.uniform(-width_shift_range,
                                   width_shift_range)
        if np.max(width_shift_range) < 1:
            ty *= x.shape[img_col_axis]
    else:
        ty = 0

    zoom_range = (0.9, 1.1)
    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)

    transform_matrix = None
    if if_flip:
        flip_matrix = np.array([[-1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]])
        transform_matrix = flip_matrix if transform_matrix is None else np.dot(transform_matrix, flip_matrix)

    if theta != 0:
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        transform_matrix = rotation_matrix

    if tx != 0 or ty != 0:
        shift_matrix = np.array([[1, 0, tx],
                                 [0, 1, ty],
                                 [0, 0, 1]])
        transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

    if zx != 1 or zy != 1:
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])
        transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

    # apply transforming
    if transform_matrix is not None:
        h, w = x.shape[img_row_axis], x.shape[img_col_axis]
        transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
        x = apply_transform(x, transform_matrix, img_channel_axis, fill_mode='nearest')

    return x


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x, transform_matrix, channel_index=0, fill_mode='constant', cval=0.):
    x = np.rollaxis(x, channel_index, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
                                                         final_offset, order=0, mode=fill_mode, cval=cval) for x_channel
                      in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index + 1)
    return x


def rotation(x, rg):
    theta = np.deg2rad(rg)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    h, w = x.shape[0], x.shape[1]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(x, transform_matrix, 2, fill_mode='constant', cval=255)

    return x

def merge_heatmap(heatmap, normalize=True):
    """
    a batch of isolated heatmap tensor (b,j,h,w) -> (b,3,h,w)
    data distribution (-1, 1) -> (0, 1)
    """
    heatmap_ = heatmap.detach().clone()
    heatmap_ = (heatmap_ + 1. ) / 2.
    if len(heatmap_.shape) > 3:
        heatmap_ = torch.sum(heatmap_, 1, keepdim=True)
    elif len(heatmap_.shape) == 3:
        heatmap_ = heatmap_.unsqueeze(1)
    else:
        print('Invalid heatmap size!')
    heatmap_ = heatmap_.repeat(1,3,1,1)
    if normalize:
        heatmap_ = (heatmap_ - torch.min(heatmap_)) / torch.max(heatmap_)
    return heatmap_

def create_heatmap(im_map, im_cloud, kernel_size=(5,5),colormap=cv2.COLORMAP_TURBO,a1=0.5,a2=0.5,normalize=True):
    '''
    im_map: a batch of im_map tensor (b,3,h,w) in (-1,1)
    im_cloud: a batch of isolated heatmap tensor (b,j,h,w)
    return a batch of heatmap overlapped with img in tensor
    '''
    im_map_ = im_map.detach().clone()
    if im_map_.shape[1] == 1:
        im_map_ = im_map_.repeat(1,3,1,1)
    im_map_ = (((im_map_ + 1. ) / 2.) * 255.).cpu().numpy().astype(np.uint8)
    im_cloud_ = merge_heatmap(im_cloud, normalize=normalize).cpu().numpy() #0~1
    im_cloud_ = (im_cloud_ * 255.).astype(np.uint8)

    im_map_ = np.rollaxis(im_map_, 1, 4)
    im_cloud_ = np.rollaxis(im_cloud_, 1, 4)
    new = np.zeros_like(im_map_)

    for i in range(len(im_map_)):
        # create blur image, kernel must be an odd number
        im_cloud_blur = cv2.GaussianBlur(im_cloud_[i],kernel_size,0)

        # If you need to invert the black/white data image
        # im_blur = np.invert(im_blur)
        # Convert back to BGR for cv2
        #im_cloud_blur = cv2.cvtColor(im_cloud_blur,cv2.COLOR_GRAY2BGR)

        # Apply colormap
        im_cloud_clr = cv2.applyColorMap(im_cloud_blur, colormap)

        # blend images a1/a2
        new[i] = (a1*im_map_[i] + a2*im_cloud_clr)
    new = np.rollaxis(new, 3, 1)
    new = torch.from_numpy(new).float()
    return new
