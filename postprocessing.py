import os, re
import numpy as np
from preprocessing import natural_keys, listdir_nods
import cv2
import PIL as Image
import shutil
import random, math
import csv
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# todo make shuffle work by experiment, not by well
def perfect_shuffle(data_dir: str, save_dir: str, exp_list, res_wells: dict, sus_wells: dict):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    # make the necessary directories
    os.mkdir(save_dir)
    os.mkdir(os.path.join(save_dir, 'train'))
    os.mkdir(os.path.join(os.path.join(save_dir, 'train'), 'Res'))
    os.mkdir(os.path.join(os.path.join(save_dir, 'train'), 'Sus'))
    os.mkdir(os.path.join(save_dir, 'valid'))
    os.mkdir(os.path.join(os.path.join(save_dir, 'valid'), 'Res'))
    os.mkdir(os.path.join(os.path.join(save_dir, 'valid'), 'Sus'))
    os.mkdir(os.path.join(save_dir, 'test'))
    os.mkdir(os.path.join(os.path.join(save_dir, 'test'), 'Res'))
    os.mkdir(os.path.join(os.path.join(save_dir, 'test'), 'Sus'))

    # prob that a condition gets placed into validation (vprob) and test (tprob)
    vprob = 0.3
    tprob = 0.1

    for e in exp_list:
        e_path = os.path.join(os.path.join(data_dir, e), 'processed')

        res_names = res_wells.get(e)  # get names of resistant wells
        print(e)
        rintlist = []
        for w in res_names:

            well_names = []  # arr to save paths

            for well in listdir_nods(e_path):
                if well == 'testing' or well == 'segmentations':  # skip segmentations dir
                    pass

                elif re.split('-', well)[1].strip().startswith(w):  # check for
                    well_names.append(well)  # save path of well

            rarr = np.linspace(1, len(well_names), num=len(well_names) - 1).tolist()
            random.shuffle(rarr)

            test_ns = []  # arr to save which wells will be in test set
            valid_ns = []  # arr to save which wells will be in valid set

            # determine which of the paths will be in test, valid
            for i in range(math.ceil(tprob * len(res_names))):
                test_ns.append(rarr.pop())

            for i in range(math.ceil(vprob * len(res_names))):
                valid_ns.append(rarr.pop())

            # loop through and assign to correct directory (train, valid, or test)
            n = 1
            for name in well_names:
                w_path = os.path.join(e_path, name)

                if n in test_ns:
                    save_path = os.path.join(os.path.join(save_dir, 'test'), 'Res')
                elif n in valid_ns:
                    save_path = os.path.join(os.path.join(save_dir, 'valid'), 'Res')
                else:

                    save_path = os.path.join(os.path.join(save_dir, 'train'), 'Res')

                for dat in listdir_nods(w_path):
                    r = random.randint(1000000000, 9999999999)
                    while r in rintlist:
                        r = random.randint(1000000000, 9999999999)
                    rintlist.append(r)
                    sname = os.path.join(save_path, str(r) + '.png')
                    cname = os.path.join(w_path, dat)
                    # todo replace with flow_from_dataframe
                    shutil.copyfile(cname, sname)

                n += 1

        # same code but for susceptible wells
        sus_names = sus_wells.get(e)

        for w in sus_names:

            well_names = []

            for well in listdir_nods(e_path):
                if well == 'testing' or well == 'segmentations':
                    pass
                elif re.split('-', well)[1].strip().startswith(w):
                    well_names.append(well)

            rarr = np.linspace(1, len(well_names), num=len(well_names) - 1).tolist()
            random.shuffle(rarr)

            test_ns = []
            valid_ns = []

            for i in range(math.ceil(tprob * len(sus_names))):
                test_ns.append(rarr.pop())

            for i in range(math.ceil(vprob * len(sus_names))):
                valid_ns.append(rarr.pop())

            n = 1
            for name in well_names:
                w_path = os.path.join(e_path, name)

                if n in test_ns:
                    save_path = os.path.join(os.path.join(save_dir, 'test'), 'Sus')
                elif n in valid_ns:
                    save_path = os.path.join(os.path.join(save_dir, 'valid'), 'Sus')
                else:

                    save_path = os.path.join(os.path.join(save_dir, 'train'), 'Sus')

                for dat in listdir_nods(w_path):
                    r = random.randint(1000000000, 9999999999)
                    while r in rintlist:
                        r = random.randint(1000000000, 9999999999)
                    rintlist.append(r)
                    sname = os.path.join(save_path, str(r) + '.png')
                    cname = os.path.join(w_path, dat)

                    shutil.copyfile(cname, sname)

                n += 1


# uses saved .npy array to color cells based on model prediction
def color_seg_preds(path_to_seg_arr: str, path_to_seg_img: str, path_to_data: str, model):
    # find path and name
    img_save_dest = os.path.split(path_to_seg_img)[0]
    img_save_name = os.path.split(path_to_seg_img)[1]

    arr = np.load(path_to_seg_arr)  # load segmentation array
    im = Image.open(path_to_seg_img)  # open segmentation image
    color = cv2.cvtColor(np.asarray(im), cv2.COLOR_GRAY2RGB)  # turn to rgb in order to color

    paths = listdir_nods(path_to_data)
    paths.sort(key=natural_keys)

    n = 0  # coordinates segmentation arrays with data images (seg array 0 == data 0, seg array 1 == data 1, ...)
    for path in paths:

        # Labels:
        # Res = 0
        # Sus = 1

        i = np.asarray(Image.open(os.path.join(path_to_data, path)))  # image to be predicted
        print(i.shape)

        # some processing to get the model to take it
        i = np.expand_dims(i, axis=2)
        i = i[np.newaxis, :]
        print(i.shape)

        # predict on model
        pred = model.predict(i)

        # points to be colored
        subset = np.vstack(arr[n]).T

        # color the points
        if pred > 0.5:
            for point in subset:
                # Sus = red
                color[point[0], point[1]][0] = 255
                color[point[0], point[1]][1] = 0
                color[point[0], point[1]][2] = 0

        else:
            for point in subset:
                # Res = green
                color[point[0], point[1]][1] = 255
                color[point[0], point[1]][0] = 0
                color[point[0], point[1]][2] = 0

        n += 1
    print(color.shape)

    # save
    cv2.imwrite(uri=os.path.join(img_save_dest, img_save_name + 'preds_visualized.png'), im=color.astype(np.uint16))


# uses saved .npy array to color cells based on model prediction
def color_seg_preds(path_to_seg_arr: str, path_to_seg_img: str, path_to_data: str, model):
    # find path and name
    img_save_dest = os.path.split(path_to_seg_img)[0]
    img_save_name = os.path.split(path_to_seg_img)[1]

    arr = np.load(path_to_seg_arr)  # load segmentation array
    im = Image.open(path_to_seg_img)  # open segmentation image
    color = cv2.cvtColor(np.asarray(im), cv2.COLOR_GRAY2RGB)  # turn to rgb in order to color

    paths = listdir_nods(path_to_data)
    paths.sort(key=natural_keys)

    n = 0  # coordinates segmentation arrays with data images (seg array 0 == data 0, seg array 1 == data 1, ...)
    for path in paths:

        # Labels:
        # Res = 0
        # Sus = 1

        i = np.asarray(Image.open(os.path.join(path_to_data, path)))  # image to be predicted
        print(i.shape)

        # some processing to get the model to take it
        i = np.expand_dims(i, axis=2)
        i = i[np.newaxis, :]
        print(i.shape)

        # predict on model
        pred = model.predict(i)

        # points to be colored
        subset = np.vstack(arr[n]).T

        # color the points
        if pred > 0.5:
            for point in subset:
                # Sus = red
                color[point[0], point[1]][0] = 255
                color[point[0], point[1]][1] = 0
                color[point[0], point[1]][2] = 0

        else:
            for point in subset:
                # Res = green
                color[point[0], point[1]][1] = 255
                color[point[0], point[1]][0] = 0
                color[point[0], point[1]][2] = 0

        n += 1
    print(color.shape)

    # save
    cv2.imwrite(os.path.join(img_save_dest, img_save_name + 'preds_visualized.png'), color)


# wrapper for color_seg_preds that works iteratively
def color_all_preds(exp_dir: str, model):
    for exp in listdir_nods(exp_dir):

        exp_path = os.path.join(exp_dir, exp)

        seg_dir = os.path.join(exp_path, 'segmentations')

        well_list = listdir_nods(exp_path)
        well_list.remove('segmentations')

        for well in well_list:
            well_path = os.path.join(exp_path, well)

            color_seg_preds(os.path.join(seg_dir, well[:len(well) - 5]) + '.npy',
                            os.path.join(seg_dir, well[:len(well) - 5]) + '1.tif.png',
                            well_path, model)


def titration_pred(save_path: str, model_path: str, csv_path: str):
    PATH = save_path

    model = tf.keras.models.load_model(model_path)

    image_gen = ImageDataGenerator(rescale=1 / 255)
    c = image_gen.flow_from_directory(PATH, target_size=(120, 69), color_mode='grayscale',
                                      batch_size=1, class_mode='sparse', shuffle=False)

    d = {
        0: [],
        1: [],
        2: [],
        3: [],
        4: [],
        5: []
    }

    to_save = {
        'label': [],
        'prediction': []
    }

    n = 0
    print(len(c.__iter__()))
    for i in c.__iter__():
        if n > len(c.__iter__()):
            break
        pred = model.predict(i[0])[0][0]
        d[int(i[1][0])].append(pred)
        to_save['label'].append(int(i[1][0]))
        to_save['prediction'].append(pred)
        n += 1

    meanarr = []
    upperq = []
    lowerq = []
    # for k in d:
    # upperq.append(np.quantile(d[k], .975))
    # lowerq.append(np.quantile(d[k], .025))
    # meanarr.append(np.mean(d[k]))
    # print(k, lowerq[k], meanarr[k], upperq[k])
    print(c.class_indices)

    with open(csv_path, 'w') as file:
        writer = csv.writer(file, )
        writer.writerow(['label', 'prediction'])
        for i in range(len(to_save['label'])):
            writer.writerow([to_save['label'][i], to_save['prediction'][i]])
