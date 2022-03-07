import os
import re
import shutil
import cv2
import imreg_dft as imr
import numpy as np
import skimage
import tifffile as tiff
from skimage import filters
from skimage.draw import disk
from skimage.measure import regionprops
from skimage.registration import phase_cross_correlation

print(skimage.__version__)

import matplotlib.pyplot as plt


# Utility functions
def natural_keys(text):
    f = (lambda x: int(x) if x.isdigit else 0)
    return [f(text)]


def listdir_nods(path):
    lst = os.listdir(path)
    if '.DS_Store' in lst:
        lst.remove('.DS_Store')

    # remove log files from processing
    for l in lst:
        if l.endswith('.log'):
            lst.remove(l)

    return lst


# Data processing utilities
def jitter_correct(video, **kwargs):
    lag = kwargs['lag']
    crop = kwargs['crop']
    upsample = kwargs['upsample']
    test_jitter = kwargs['test_jitter']

    alignment = np.empty((video.shape[0], video.shape[1] - 2 * crop, video.shape[2] - 2 * crop)).astype(np.uint16)

    save = np.empty((video.shape[0], video.shape[1], video.shape[2])).astype(np.uint16)

    def loop(frame, alignment, save, n: int = 0):
        i = np.asarray(frame).astype(np.uint16)

        i_gauss = np.asarray(skimage.filters.gaussian(i, preserve_range=True))

        i_cropped_gauss = skimage.util.crop(i_gauss, (
            (crop, crop), (crop, crop)))

        if n == 0:  # for the first loop
            alignment[n, :, :] = i_cropped_gauss
            save[n, :, :] = i
            shift = tuple([0, 0])

        else:
            if n > lag:
                bound = n - lag
            else:
                bound = 0
            shift, err, diffphase = phase_cross_correlation(np.mean(alignment[bound:n, :, :], axis=0),
                                                            i_cropped_gauss,
                                                            upsample_factor=upsample)  # calculate the drifts

            if test_jitter: print(err)

            # transform image based on the drift
            cg_transformed = np.asarray(
                imr.imreg.transform_img(i_cropped_gauss, mode='nearest',
                                        tvec=tuple(shift)))
            transformed = np.asarray(
                imr.imreg.transform_img(i, mode='nearest', tvec=tuple(shift)))

            alignment[n, :, :] = cg_transformed
            save[n, :, :] = transformed
       # print(tuple(shift))
        return alignment, save, tuple(shift)

    shift_arr = []
    for n in range(video.shape[0]):
        alignment, save, shift = loop(video[n, :, :], alignment, save, n)
        shift_arr.append([shift[0], shift[1]])

    #print(shift_arr)
    return save, shift_arr


# TODO Check functionality.
def apply_jitter_correct(video, shift_arr):
    save = np.empty((video.shape[0], video.shape[1], video.shape[2])).astype(np.uint16)

    for i in range(video.shape[0]):
        #print(shift_arr[i])
        transform = np.asarray(
            imr.imreg.transform_img(video[i, :, :], mode='nearest', tvec=shift_arr[i]))
        save[i, :, :] = transform

    return save


def segment(frame_in, **kwargs):
    s = frame_in.shape
    cropped = frame_in[kwargs['crop']:s[0] - kwargs['crop'], kwargs['crop']:s[1] - kwargs['crop']]

    lab = skimage.filters.threshold_local(cropped, block_size=kwargs['block_size']).astype(np.uint16)
    lab = skimage.morphology.label(lab)
    lab = skimage.morphology.remove_small_objects(lab, min_size=kwargs['min_size'])

    segmented_frame = np.zeros(s)
    disk_inds = []

    regions = regionprops(lab)
    for region in regions:
        c = region.centroid
        rr, cc = disk((int(c[1] + kwargs['crop']), int(c[0] + kwargs['crop'])), radius=kwargs['radius'])
        disk_inds.append([rr, cc])
        segmented_frame[cc, rr] = 1

    return segmented_frame, np.array(disk_inds)


def normalize(video):
    video = video / 65535  # uint16 to double
    mean = np.mean(video, axis=0)
    blurr = skimage.filters.gaussian(mean, sigma=(100, 100))
    video = np.divide(video, blurr)
    return video


# Oneshot Preprocessing
def process(data_path: str, seg_channel: int, dat_channel: int, test_jitter=False, test_seg=False):
    paths = listdir_nods(data_path)
    save_path = os.path.join(data_path, 'processed')

    if not os.path.exists(save_path):
        #print("Save directory not found. Partitioning...")
        os.mkdir(save_path)
        os.mkdir(os.path.join(save_path, 'segmentations'))
        os.mkdir(os.path.join(save_path, 'testing'))
        #print("Done")

    # Create a list of video names
    vlist = []
    for v in paths:
        if not v.endswith('.tif'):
            continue
        vlist.append(v[:-6])

    vlist = list(dict.fromkeys(vlist))

    for v in vlist:

        dat_suffix = 'C' + str(dat_channel) + '.tif'
        seg_suffix = 'C' + str(seg_channel) + '.tif'

        vsave = os.path.join(save_path, v + dat_suffix)
        if not os.path.exists(vsave):
            os.mkdir(vsave)

        seg_video_path = os.path.join(data_path, v + seg_suffix)
        dat_video_path = os.path.join(data_path, v + dat_suffix)

        seg_load = np.asarray(tiff.imread(seg_video_path))
        dat_load = np.asarray(tiff.imread(dat_video_path))

        # Only normalize segmentation video to make processing easier.
        seg_video = normalize(seg_load)

        seg_video, keys = jitter_correct(seg_video, lag=3, crop=100, upsample=100, test_jitter=False)
        dat_video = apply_jitter_correct(dat_load, keys)

        if test_jitter:
            path = os.path.join(os.path.join(save_path, 'testing'), 'test_video.tif')
            print("See test video here:")
            print(path)
            tiff.imwrite(path, seg_video)
            return

        seg, inds = segment(seg_video[5, :, :], crop=200, min_sigma=10, max_sigma=50, num_sigma=50,
                            threshold=.000001, overlap=0, radius=5, min_size=30, block_size=3)

        if os.path.exists(os.path.join(os.path.join(save_path, 'segmentations'), 'numpy')):
            n = 0
            for i in inds:
                np.save(os.path.join(os.path.join(save_path, 'segmentations'), str(n) + '.np'), str(i))
                n += 1

        if test_seg:
            path = os.path.join(os.path.join(save_path, 'testing'), 'test_segment.png')
            plt.imshow(seg)
            plt.show()
            print("See test segmentation here:")
            print(path)
            cv2.imwrite(path, seg)
            return

        if inds.ndim == 1:
            continue

        save_arr = np.empty((len(inds), dat_video.shape[0], len(inds[0, 0])))

        for nframe in range(dat_video.shape[0]):
            loaded = dat_video[nframe, :, :]
            for s in range(len(inds)):
                save_arr[s, nframe, :] = loaded[inds[s, 1], inds[s, 0]]

        for a in range(save_arr.shape[0]):
            fsave = os.path.join(vsave, str(a) + '.png')
            #print(fsave)
            cv2.imwrite(filename=fsave, img=save_arr[a, :, :].astype(np.uint16))

    print("Done Processing, saved in:")
    print(data_path)


def well_select(source_dir: str, dest_dir: str, row: str):
    for folder in listdir_nods(source_dir):
        if folder == 'testing' or folder == 'segmentations':
            continue
        if re.split("-", folder)[1].strip().startswith(row):
            folder_path = os.path.join(source_dir, folder)
            for file in listdir_nods(folder_path):
                file_source_path = os.path.join(folder_path, file)
                file_copy_path = os.path.join(dest_dir, folder + file)
                shutil.copy(file_source_path, file_copy_path)


def well_select_2000(source_dir: str, dest_dir: str, rows):
    for row in rows:
        well_select(source_dir, dest_dir, str(row))


