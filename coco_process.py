import os
import cv2
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

IMG_HEIGHT, IMG_WIDTH = 320, 320

annFile='../../annotations/instances_train2017.json'
# initialize COCO api for instance annotations
coco=COCO(annFile)

kps_annFile = '../../annotations/person_keypoints_train2017.json'
coco_kps=COCO(kps_annFile)


# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

catIds = coco.getCatIds(catNms=['person'])
imgIds = coco.getImgIds(catIds=catIds )
print(len(imgIds))

def save_boximg(img, bbox):
    [bbox_x, bbox_y, bbox_w, bbox_h] = bbox
    bbox_x, bbox_y, bbox_w, bbox_h = int(bbox_x), int(bbox_y), int(bbox_w), int(bbox_h)
    new_img = np.zeros((bbox_w, bbox_h, 3))
    if len(img.shape) == 3:
        new_img = img[bbox_y:bbox_y+bbox_h, bbox_x:bbox_x+bbox_w, :]
    else:
        new_img = img[bbox_y:bbox_y+bbox_h, bbox_x:bbox_x+bbox_w]
    return new_img

def make_bg_white(im):
    gray = 0.2125*im[...,0] + 0.7154*im[...,1] + 0.0721*im[...,2]
    im[gray == 0] = 255
    return im

def make_square(img):
    H, W = img.shape[:2]
    maxb = max(H, W)
    deltah, deltaw = np.abs(H-maxb) // 2, np.abs(W-maxb) // 2
    new_im = cv2.copyMakeBorder(img, deltah, deltah, deltaw, deltaw,
        cv2.BORDER_CONSTANT, value=[255,255,255])
    new_im = cv2.resize(new_im, (maxb, maxb), cv2.INTER_AREA)
    return new_im, deltah, deltaw

def viz_one_img_w_joint_color(im, joints):
    color = np.array([0,0,255.])
    img = im.copy()

    for i in range(len(joints)):
        x, y, v = int(joints[i,0]), int(joints[i,1]), int(joints[i,2])
        if v == 2 and x > 0 and y > 0:
            x, y = y, x
            print('viz joints:', x,y, img.shape)
            img[x,y,:] = color
            if x+1<img.shape[0] and x-1>0:
                img[x-1,y,:]=img[x+1,y,:] = color
                if y+1<img.shape[1] and y-1>0:
                    img[x-1,y-1,:]=img[x,y-1,:]=img[x+1,y-1,:]=color
                    img[x-1,y+1,:]=img[x,y+1,:]=img[x+1,y+1,:]=color
    return img

def points_to_gaussian_heatmap(centers, height=IMG_HEIGHT, width=IMG_WIDTH, scale=64):
    """
    make a heatmap of gaussians taking as input the centers of the gaussians
    W = 400  # width of heatmap
    H = 400  # height of heatmap
    SCALE = 64  # increase scale to make larger gaussians
    CENTERS = [(100,100),
            (100,300),
            (300,100)]
    """
    gaussians = []
    for y,x in centers:
        s = np.eye(2)*scale
        g = multivariate_normal(mean=(x,y), cov=s)
        gaussians.append(g)

    # create a grid of (x,y) coordinates at which to evaluate the kernels
    x = np.arange(0, width)
    y = np.arange(0, height)
    xx, yy = np.meshgrid(x,y)
    xxyy = np.stack([xx.ravel(), yy.ravel()]).T

    # evaluate kernels at grid points
    zz = sum(g.pdf(xxyy) for g in gaussians)

    img = zz.reshape((height,width))
    img = (img - np.min(img)) / np.max(img)
    return img

# 0	:	nose
# 1	:	left_eye
# 2	:	right_eye
# 3	:	left_ear
# 4	:	right_ear
# 5	:	left_shoulder
# 6	:	right_shoulder
# 7	:	left_elbow
# 8	:	right_elbow
# 9	:	left_wrist
# 10	:	right_wrist
# 11	:	left_hip
# 12	:	right_hip
# 13	:	left_knee
# 14	:	right_knee
# 15	:	left_ankle
# 16	:	right_ankle


data = []

# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms=['person'])
imgIds = coco.getImgIds(catIds=catIds )
print(len(imgIds))

for i in range(0, len(imgIds)):
    img = coco.loadImgs(imgIds[i])[0]
    # use url to load image
    #I = io.imread(img['coco_url'])
    I = cv2.imread(os.path.join('../../images/train2017/', img['file_name']))

    # load instance annotations
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    # load keypoints anns
    annIds_kps = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns_kps = coco_kps.loadAnns(annIds_kps)

    # get masked bbox saved
    # 1. mask the human figure, 2. bbox crop human figure, 3. make image square and same size
    for j in range(len(anns)):
        if len(anns_kps[j]['segmentation'])==1 and anns_kps[j]['num_keypoints'] > 15:
        # restrict to one polygon seg &

            mask = np.zeros_like(I)
            mask = coco.annToMask(anns[j]) * anns[j]['category_id']
            if len(I.shape) == 3:
                mask = cv2.merge([mask,mask,mask])
            new_img = save_boximg(I*mask, anns[j]['bbox'])
            new_img = make_bg_white(new_img)

            # get the segmented single human figure
            if new_img.shape[0]*new_img.shape[1] > 2500:
                print('original img:', new_img.shape)
                # keypoints
                keypoints = np.array(anns_kps[j]['keypoints']).reshape(17, 3)
                keypoints = list(keypoints)
                del keypoints[1], keypoints[1], keypoints[1], keypoints[1] #delete ears&eyes

                # rescale keypoints, matching bbox crop
                [bbox_x, bbox_y, bbox_w, bbox_h] = anns_kps[j]['bbox']
                keypoints = np.array(keypoints)
                keypoints[:,0] = keypoints[:,0] - bbox_x
                keypoints[:,1] = keypoints[:,1] - bbox_y

                # make square
                new_img, deltah, deltaw = make_square(new_img)
                # matching make square img
                keypoints[:,0] += deltaw
                keypoints[:,1] += deltah

                # add frame to imgs
                frame_width = new_img.shape[0] // 5
                new_img = cv2.copyMakeBorder(new_img, frame_width, frame_width,
                    frame_width, frame_width, cv2.BORDER_CONSTANT, value=[255,255,255])

                keypoints[:,0] += frame_width
                keypoints[:,1] += frame_width

                # rescale imgs to same size
                h, w = new_img.shape[0], new_img.shape[1]
                new_img = cv2.resize(new_img, (IMG_HEIGHT, IMG_WIDTH), cv2.INTER_AREA)
                scaleh = IMG_HEIGHT / h
                scalew = IMG_WIDTH / w
                # matching resized img
                keypoints[:,0] = keypoints[:,0] * scaleh
                keypoints[:,1] = keypoints[:,1] * scalew

                confidence_map = []
                for k in range(len(keypoints)):
                    if keypoints[k, 2] == 2:
                        confidence_map.append(
                            points_to_gaussian_heatmap([(keypoints[k,1], keypoints[k,0])])
                            )
                    else:
                        confidence_map.append(
                            np.zeros((IMG_HEIGHT, IMG_WIDTH))
                        )

                # visualize 10 data
                # if i < 10:
                #     viz = viz_one_img_w_joint_color(new_img, keypoints)
                #     cv2.imwrite('./imgs/%s_%d.png' % (img['id'], j), viz.astype(np.uint8))
                #     sum_confi = np.sum(np.array(confidence_map), axis=0).squeeze()
                #     sum_confi = (sum_confi - np.min(sum_confi)) / np.max(sum_confi)
                #     print('sum confi', sum_confi.shape, np.max(sum_confi))
                #     cv2.imwrite('./imgs/%s_%d_c.png' % (img['id'], j), (sum_confi*255.).astype(np.float32))

                # img: 0~255, (h, w, 3)
                # confidence_map: 0~1, (keypoint_len, h, w)
                data.append({
                    'img': new_img,
                    'keypoints': keypoints.astype(np.float32),
                    'confidence_map': np.array(confidence_map).astype(np.float32)
                })

np.save(open('coco.npy', 'wb'), data)
