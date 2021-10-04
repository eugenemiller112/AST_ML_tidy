import re, os
from PIL import Image
import numpy as np
import cv2
from skimage.registration import phase_cross_correlation
import imreg_dft as imr
import skimage


# Utility functions
def natural_keys(text):
    f = (lambda x: int(x) if x.isdigit else 0)
    return [f(re.split(r'(\d+)', text)[1])]


def listdir_nods(path):
    lst = os.listdir(path)
    if '.DS_Store' in lst:
        lst.remove('.DS_Store')
    return lst


def jitter_correct(video, lag: int = 5, crop: int = 100, upsample: int = 100, test_jitter: bool = False):

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

    alignment = np.array(video.shape).astype(np.uint16)
    save = np.array(video.shape).astype(np.uint16)

    for n in range(video.shape[0]):
        alignment, save = loop(video[n, :, :], alignment, save, n)

    return save



# Preprocessing
def process(data_path: str, test_jitter=False, test_seg=False):
    paths = listdir_nods(data_path)
    paths.sort(key=natural_keys)

    save_path = os.path.join(data_path, 'processed')
    if not os.path.exits(save_path):
        os.mkdir(save_path)
        os.mkdir(os.path.join(save_path, 'segmentations'))
        os.mkdir(os.path.join(save_path, 'testing'))

    for v in paths:

        os.mkdir(save_path, v)
        video_path = os.path.join(data_path, v)
        video = np.asarray(Image.open(video_path))
        video = jitter_correct(video)  # todo write this

        if test_jitter:
            cv2.imwrite(os.path.join(os.path.join(save_path, 'testing'), 'video'), video)
            return

        seg = segment(video[0, :, :])  # todo define this

        if test_seg:
            cv2.imwrite(os.path.join(os.path.join(save_path, 'testing'), 'segment'), seg)
            return

        save_arr = np.array((len(seg), len(video.shape(0)), len(seg[0][0])))

        for nframe in range(video.shape[0]):

            loaded = video[nframe, :, :]

            for s in range(seg):
                save_arr[s, :, :] = loaded[seg[s][0], seg[s][1]]

        for a in range(save_arr.shape[0]):
            cv2.imwrite(filename='save_path' + a + '.png', image=save_arr[a, :, :].astype(np.uint16))
