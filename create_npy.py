import os
from PIL import Image
import argparse
import numpy as np

def parse_data(datadir):
    print(datadir)
    img_label_dic = []
    size = 128
    for root, _, filenames in os.walk(datadir):
        img_label = []
        i = 0
        for filename in filenames:
            if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg') or filename.endswith('.tif'):
                filei = os.path.join(root, filename)
                img_label.append(np.asarray(Image.open(filei).resize((size,size))))
                i += 1
                if i == 100:
                    break
        if len(img_label) != 0:
            img_label_dic.append(np.asarray(img_label))

    img_npy = np.stack(img_label_dic, axis = 0)
    np.save('cancer.npy', img_npy)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str,)
    args = parser.parse_args()

    parse_data(args.dir)
