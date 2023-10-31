import cv2
from skimage import color
import numpy as np
from numpy import asarray
import os
import deepdish as dd
import shutil
from PIL import Image
from scipy.io import loadmat
from skimage import measure
from skimage.measure import regionprops
import pdb
import copy

class myobj(object):
    pass

def get_seed_name(threshhold, min_len):
    name  =('t_'   + '{:01.02f}'.format(threshhold) \
             + '_r_'+  '{:02.02f}'.format(min_len)).replace('.','_')
    return name

def get_labelmap_name(threshhold, area_thd):
    name  = ('t_' + '{:01.02f}'.format(threshhold) + '_a_'+ '{:02d}'.format(area_thd)).replace('.','_')
    return name

def getfilelist(Imagefolder, inputext):
    '''inputext: ['.json'] '''
    if type(inputext) is not list:
        inputext = [inputext]
    filelist = []
    filenames = []
    for f in sorted(os.listdir(Imagefolder)):
        if os.path.splitext(f)[1] in inputext and os.path.isfile(os.path.join(Imagefolder,f)):
               filelist.append(os.path.join(Imagefolder,f))
               filenames.append(os.path.splitext(os.path.basename(f))[0])
    return filelist, filenames

def imread(imgfile):
    assert os.path.exists(imgfile), '{} does not exist!'.format(imgfile)
    srcBGR = cv2.imread(imgfile)
    destRGB = cv2.cvtColor(srcBGR, cv2.COLOR_BGR2RGB)
    return destRGB

def overlayImg(img, mask,print_color =[5,119,72],alpha = 0.618,savepath = None):
    rows, cols = img.shape[0:2]
    color_mask = np.zeros((rows, cols, 3))
    assert len(mask.shape) == 2,'mask should be of dimension 2'
    color_mask[mask == 1] = print_color
    color_mask[mask == 0] = img[mask == 0]

    if len(img.shape) == 2:
       img_color = np.dstack((img, img, img))
    else:
       img_color = img

    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)

    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(img_hsv)
    img_masked = np.asarray((img_masked/np.max(img_masked)) * 255, dtype = np.uint8)

    if savepath is not None:
        im = Image.fromarray(img_masked)
        im.save(savepath)
    return img_masked

def removeBackground(origimg, cropheight=200, cropwidth=200, numclust=5):
    copyimg = copy.copy(origimg)
    imgheight, imgwidth, imgchannel = origimg.shape
    mask = np.ones((imgheight, imgwidth))
    numhistbin = 10
    numpatch = int(np.ceil(imgheight/cropheight) * np.ceil(imgwidth/cropwidth))
    colorhist = np.zeros((numhistbin*imgchannel, numpatch))
    k=0
    for i in range(0,imgheight,cropheight):
        for j in range(0,imgwidth,cropwidth):
            cropimg = origimg[i:i+cropheight, j:j+cropwidth, :]
            colorhist[0:numhistbin, k] = np.squeeze(cv2.calcHist([cropimg], [0], None, [numhistbin], [0, 256]))
            colorhist[numhistbin:numhistbin*2, k] = np.squeeze(cv2.calcHist([cropimg], [1], None, [numhistbin], [0, 256]))
            colorhist[numhistbin*2:, k] = np.squeeze(cv2.calcHist([cropimg], [2], None, [numhistbin], [0, 256]))
            k += 1

    kmeans = cluster.KMeans(n_clusters=numclust, random_state=0).fit(np.transpose(colorhist))
    background_label = kmeans.labels_[0]
    k = 0
    for i in range(0,imgheight,cropheight):
        for j in range(0,imgwidth,cropwidth):
            if kmeans.labels_[k] == background_label:
                mask[i:i+cropheight, j:j+cropwidth] = 0
            k += 1

    mask[i:,:] = 0
    mask[:,j:] = 0
    copyimg[mask==0] = 0
    return copyimg, mask

def printImage(Img=None, coordinates=None, savepath=None,  use_mask=True, **kwargs):
    param = myobj()
    param.linewidth = 3
    param.hlinewidth = (param.linewidth - 1) // 2
    param.color = [[255,0,255],[255,0,255],[255,0,255],[255,0,255]]
    param.alpha = 1
    for key in kwargs:
        setattr(param, key, kwargs[key])

    img_shape = Img.shape
    if use_mask == True:
        copyimg = copy.deepcopy(Img)
        _, mask = removeBackground(Img, cropheight=200, cropwidth=200, numclust=6)
    if coordinates.size != 0:
        coor_shape = coordinates.shape
        for idx in range(coor_shape[0]):
            x1 = int(round(coordinates[idx,0])) - param.hlinewidth
            x2 = int(round(coordinates[idx,0])) + param.hlinewidth + 1
            y1 = int(round(coordinates[idx,1])) - param.hlinewidth
            y2 = int(round(coordinates[idx,1]))+ param.hlinewidth + 1
            if x1 >= 0 and x2 <= img_shape[0] and y1 >= 0 and y2 <= img_shape[1]:
                label_idx = 0
                color_b = np.full((param.linewidth,param.linewidth),param.color[label_idx][0])
                color_g = np.full((param.linewidth,param.linewidth),param.color[label_idx][1])
                color_r = np.full((param.linewidth,param.linewidth),param.color[label_idx][2])
                color_dot = cv2.merge((color_b, color_g, color_r))
                Img[x1:x2, y1:y2] = color_dot

    if use_mask == True:
        Img[mask==0] = copyimg[mask==0]
    if savepath:
       cv2.imwrite(savepath, Img)
    return Img


def printImage_seg(Img=None, map_label=None, num_label=0, savepath=None, use_mask=True, **kwargs):
    param = myobj()
    param.linewidth = 3
    param.hlinewidth = (param.linewidth - 1) // 2
    param.color = [[0,0,255],[0,255,0],[255,0,0],[0,255,255],[255,0,255],[255,255,0],
                    [0,0,128],[0,128,0],[128,0,0],[0,128,128],[128,0,128],[128,128,0],
                    [0,128,255],[128,0,255],[0,255,128],[128,255,0],[255,0,128],[255,128,0]]
    param.alpha = 1
    for key in kwargs:
        setattr(param, key, kwargs[key])

    if len(Img.shape) == 2 or Img.shape[2] == 1:
        Img = cv2.cvtColor(Img,cv2.COLOR_GRAY2RGB)
    if use_mask == True:
        copyimg = copy.deepcopy(Img)
        _, mask = removeBackground(Img, cropheight=200, cropwidth=200, numclust=6)
    if num_label > 0:

        for idx in range(1,num_label+1):
            Img[map_label==idx] = param.color[idx-1]

    if use_mask == True:
        Img[mask==0] = copyimg[mask==0]
    if savepath:
       cv2.imwrite(savepath, Img)
    return Img

def printImage_seg2(Img=None, map_label=None, num_label=0, savepath=None, use_mask=False, alpha=0.85, **kwargs):

    assert Img is not None, 'input field not valid'
    param = myobj()
    param.color = [[0,0,255],[0,255,0],[255,0,0],[0,255,255],[255,0,255],[255,255,0],
                    [0,0,128],[0,128,0],[128,0,0],[0,128,128],[128,0,128],[128,128,0],
                    [0,128,255],[128,0,255],[0,255,128],[128,255,0],[255,0,128],[255,128,0]]
    param.alpha = alpha
    for key in kwargs:
        setattr(param, key, kwargs[key])

    if use_mask == True:
        copyimg = copy.deepcopy(Img)
        _, mask = removeBackground(Img, cropheight=200, cropwidth=200, numclust=6)
    overlaiedRes = Img
    if num_label > 0:
        for idx in range(1,num_label+1):
            label_mask = np.zeros(Img.shape[0:2])
            label_mask[map_label==idx] = 1
            overlaiedRes =  overlayImg(overlaiedRes, label_mask, print_color = param.color[idx-1], alpha = param.alpha)

    im_masked = Image.fromarray(overlaiedRes)
    if use_mask == True:
        im_masked[mask==0] = copyimg[mask==0]
    if savepath:
       im_masked.save(savepath)
    return overlaiedRes


def printCoords_seg_slc(savefolder, resultfolder, imgname, imgdir, imgext, threshhold=0.50, area_thd=0,mask=None, alpha=1):

    ol_folder = os.path.join(savefolder, get_labelmap_name(threshhold,area_thd))
    if not os.path.exists(ol_folder):
       os.makedirs(ol_folder)
    print('overlay image {ind}'.format(ind = imgname + '.png'))
    imgpath = os.path.join(imgdir, imgname + '.png')
    assert os.path.isfile(imgpath), 'image does not exist!'
    thisimg = cv2.imread(imgpath)
    savepath = os.path.join(ol_folder, imgname + '_ol.png' )
    resultDictPath = os.path.join(resultfolder, imgname +  '.mat')
    if os.path.isfile(resultDictPath):
       resultsDict = loadmat(resultDictPath)
    labelmapname = get_labelmap_name(threshhold,area_thd)
    labelnumname = get_labelmap_name(threshhold,area_thd) + '_number'
    map_label = resultsDict[labelmapname]
    num_label = int(resultsDict[labelnumname])
    if alpha == 1:
        printImage_seg(Img=thisimg, map_label=map_label, num_label=num_label, savepath=savepath, use_mask=False)
    else:
        printImage_seg2(Img=thisimg, map_label=map_label, num_label=num_label, savepath=savepath, use_mask=False, alpha=alpha)


def removeSmallRegions(label_image, area_threshold=0):
    label_img_copy = copy.deepcopy(label_image)
    props = measure.regionprops(label_image)
    for idx in range(len(props)):
        cur_label = props[idx].label
        cur_area = props[idx].area
        if cur_area <= area_threshold:
            label_img_copy[label_img_copy==cur_label] = 0
    label_img_copy[label_img_copy > 0] = 1
    map_label, num_label = measure.label(label_img_copy.astype(int), return_num = True)
    return map_label, num_label
