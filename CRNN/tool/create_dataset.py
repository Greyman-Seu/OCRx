import os
import lmdb
import cv2
import numpy as np
import argparse
import shutil
import sys


def checkImageIsValid(imageBin):  # chuan: imageBin 对应的词汇是什么？
    if imageBin is None:
        return False

    try:
        # imageBuf = np.fromstring(imageBin, dtype=np.uint8)
        imageBuf = np.frombuffer(imageBin, dtype=np.uint8)  # chuan
        # img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
        img = cv2.imdecode(imageBuf, cv2.IMREAD_COLOR)
        imgH, imgW = img.shape[0], img.shape[1]
    except:
        return False
    else:
        if imgH * imgW == 0:
            return False

    return True


def read_data_from_folder(folder_path, type):
    """
    :param folder_path:
    :param type:   raw: text_color.png
    :return:
    """

    image_path_list = []
    label_list = []
    pics = os.listdir(folder_path)
    # chuan: 按照长度排序，但等长的元素顺序不变(Timesort, stable)
    pics.sort(key=lambda i: len(i))
    for pic in pics:
        if pic.endswith(".png"):
            image_path_list.append(folder_path + '/' + pic)
            if len(pic.split('_')) == 3:
                if type=="red":
                    label_list.append(pic.split('_')[1])  # chuan
            elif len(pic.split('_')) == 2:
                if type == "text":
                    label_list.append(pic.split('_')[0])
                elif type == "color":
                    label_list.append(pic.split('_')[1].split(".png")[0])
                elif type == "both":
                    label_list.append(pic.split(".png")[0])
    return image_path_list, label_list


def read_data_from_file(file_path):
    image_path_list = []
    label_list = []
    f = open(file_path)  # chuan: 没有f.close()?
    while True:
        line1 = f.readline()
        line2 = f.readline()
        if not line1 or not line2:
            break
        line1 = line1.replace('\r', '').replace('\n', '')
        line2 = line2.replace('\r', '').replace('\n', '')
        image_path_list.append(line1)
        label_list.append(line2)

    return image_path_list, label_list


def show_demo(demo_number, image_path_list, label_list):
    print('\nShow some demo to prevent creating wrong lmdb data')
    print('The first line is the path to image and the second line is the image label')
    for i in range(demo_number):
        print('image: %s\nlabel: %s\n' % (image_path_list[i], label_list[i]))


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            if type(k) == str:
                k = k.encode()  # chuan: 将字符串存储为字节码形式
            if type(v) == str:
                v = v.encode()
            txn.put(k, v)


def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.

    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    # If lmdb file already exists, remove it. Or the new data will add to it.
    if os.path.exists(outputPath):
        shutil.rmtree(outputPath)
        os.makedirs(outputPath)
    else:
        os.makedirs(outputPath)

    assert (len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)

    # chuan: 1099511627776=1024**4, 1TB?
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in range(nSamples):
        imagePath = imagePathList[i]
        label = labelList[i]

        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt - 1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    env.close()
    print('Created dataset with %d samples' % nSamples)

# zyk
def write_imglist(path,imglist):
    with open(path,"w") as f:
        for line in imglist:
            f.write(line + '\n')
    print("imglist is saved to textfile:  successful!!!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # zyk
    parser.add_argument('--out', type=str, default="/home/chuan/dataset/captcha/raw/both/test",  # required=True,
                        help='lmdb data output path')
    parser.add_argument('--folder', type=str, default="/home/chuan/dataset/captcha/raw/fake_pic_test",
                        help='path to folder which contains the images')
    parser.add_argument('--type', type=str, default="both",
                        help='text;color;both;red;')
    parser.add_argument(
        '--file', type=str,
        help='path to file which contains the image path and label')
    args = parser.parse_args()

    if args.file is not None:
        image_path_list, label_list = read_data_from_file(args.file)
        createDataset(args.out, image_path_list, label_list)
        show_demo(2, image_path_list, label_list)
    elif args.folder is not None:
        image_path_list, label_list = read_data_from_folder(args.folder, type=args.type)

        write_imglist(os.path.join(args.out, "imglist.txt"), image_path_list)

        createDataset(args.out, image_path_list, label_list)
        show_demo(2, image_path_list, label_list)
    else:
        print('Please use --floder or --file to assign the input. Use -h to see more.')
        sys.exit()
