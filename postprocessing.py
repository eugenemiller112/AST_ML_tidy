import os
import numpy as np
from preprocessing import natural_keys, listdir_nods
import cv2
import PIL as Image

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
