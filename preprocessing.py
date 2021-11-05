import os
import tifffile as tiff
import numpy as np
import cv2
from skimage.registration import phase_cross_correlation
from skimage.feature import blob_doh
import sklearn
from skimage.draw import disk
import imreg_dft as imr
import skimage

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


def jitter_correct(video, lag: int = 5, crop: int = 100, upsample: int = 100, test_jitter: bool = False):
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

        return alignment, save

    for n in range(video.shape[0]):
        alignment, save = loop(video[n, :, :], alignment, save, n)

    return save


def segment(frame_in, **kwargs):
    s = frame_in.shape
    cropped = frame_in[kwargs['crop']:s[0] - kwargs['crop'], kwargs['crop']:s[1] - kwargs['crop']]

    blobs = blob_doh(cropped, min_sigma=kwargs['min_sigma'], max_sigma=kwargs['max_sigma'],
                     num_sigma=kwargs['num_sigma'], threshold=kwargs['threshold'],
                     overlap=kwargs['overlap'])

    segmented_frame = np.empty(frame_in.shape)
    disk_inds = []
    for blob in blobs:
        print(blob)
        rr, cc = disk((int(blob[1] + kwargs.crop), int(blob[0] + kwargs.crop)), radius=kwargs.radius)
        disk_inds.append([rr, cc])
        segmented_frame[rr, cc] = 1

    return segmented_frame, disk_inds


def normalize(video):
    video = video / 65535  # uint16 to double
    mean = np.mean(video, axis=0)
    blurr = skimage.filters.gaussian(mean, sigma=(100, 100))
    video = np.divide(video, blurr)
    return video


# Oneshot Preprocessing
def process(data_path: str, test_jitter=False, test_seg=False):
    paths = listdir_nods(data_path)

    # paths.sort(key=natural_keys)

    save_path = os.path.join(data_path, 'processed')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        os.mkdir(os.path.join(save_path, 'segmentations'))
        os.mkdir(os.path.join(save_path, 'testing'))

    for v in paths:

        if not os.path.exists(os.path.join(save_path, v)):
            os.mkdir(os.path.join(save_path, v))
        video_path = os.path.join(data_path, v)

        video = np.asarray(tiff.imread(video_path))
        video = video / 65535

        video = normalize(video)

        video = jitter_correct(video)

        if test_jitter:
            path = os.path.join(os.path.join(save_path, 'testing'), 'video')
            print(path)
            cv2.imwrite(path + '.tif', video)
            return
        sh = skimage.morphology.remove_small_objects(video[0, :, :], min_size=128)
        plt.imshow(video[0, :, :])
        plt.show()
        plt.imshow(sh)
        plt.show()

        seg, inds = segment(video[0, :, :], crop=200, min_sigma=0.1, max_sigma=100, num_sigma=50,
                            threshold=0.00001, overlap=0, radius=5)

        if not os.path.exists(os.path.join(os.path.join(save_path, 'segmentations'), 'numpy')):
            n = 0
            for i in inds:
                np.save(os.path.join(os.path.join(save_path, 'segmentations'), n + '.np'), i)
                n += 1

        if test_seg:
            path = os.path.join(os.path.join(save_path, 'testing'), 'segment')
            print(path)
            plt.imshow(seg)
            plt.show()
            cv2.imwrite(path + '.png', seg)
            return

        print(inds)
        save_arr = np.empty((len(inds), video.shape[1], video.shape[2]))

        for nframe in range(video.shape[0]):

            loaded = video[nframe, :, :]

            for s in range(len(seg[0])):
                save_arr[s, :, :] = loaded[inds[s][1], inds[s][0]]

        for a in range(save_arr.shape[0]):
            cv2.imwrite(filename='save_path' + a + '.png', image=save_arr[a, :, :].astype(np.uint16))
    print("Done Processing", data_path)
