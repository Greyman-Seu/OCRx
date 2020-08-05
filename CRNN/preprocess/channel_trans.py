import os
from multiprocessing import Pool, cpu_count
from functools import partial
import time
import numpy as np
import cv2


def trans_name(filename: str, original: str, target: str) -> str:
    res = []
    labels, colors = filename.split('_')
    indices = [i for i, color in enumerate(colors) if color == original]

    if len(indices) == 0:
        return None

    for i in indices:
        res.append(labels[i])
    res.append('_')
    for _ in indices:
        res.append(target)
    res.extend(colors[-4:])  # ".png"
    res = ''.join(res)

    return res


def trans_image(filename, root, output):
    """Transform a single image."""

    # 读入原图
    # cv2 中默认读入图像是(B,G,R)
    image = cv2.imread(root + '/' + filename)
    red2red_name = trans_name(filename, '红', '红')  # 转换原图名称

    # 蓝色转红色
    # 蓝色R、G值较小，B值独大，B、G交换，接近红色分布
    blue2red = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    blue2red_name = trans_name(filename, '蓝', '红')

    # 黑色转红色
    # 黑色三个通道都比较小， 用255-R通道，则R通道独大，其余通道小，接近红色的分布
    black2red = image.copy()
    black2red[:, :, 2] = 255 - black2red[:, :, 2]
    black2red_name = trans_name(filename, '黑', '红')

    # 黄色转红色
    # 黄色R、G值较大，B值较小，先取反，则R、G值较小, B值独大，接近蓝色分布
    yellow2red = image.copy()
    yellow2red = cv2.bitwise_not(image)
    yellow2red = cv2.cvtColor(yellow2red, cv2.COLOR_RGB2BGR)  # 与蓝色转红色相同
    yellow2red_name = trans_name(filename, '黄', '红')

    # 写入结果
    names = [red2red_name, blue2red_name, black2red_name, yellow2red_name]
    imgs = [image, blue2red, black2red, yellow2red]
    global wc
    for name, img in zip(names, imgs):
        if name is not None:
            wc += 1
            cv2.imwrite(output+f'/{wc:0>6d}_'+name, img)
    # time.sleep(0.1)
    return


def trans_image0(filename, root, output):
    """Transform a single image."""

    # 读入原图
    # cv2 中默认读入图像是(B,G,R)
    image = cv2.imread(root + '/' + filename)
    red2red_name = trans_name(filename, '红', '红')  # 转换原图名称

    # 蓝色转红色
    # 蓝色R、G值较小，B值独大，B、G交换，接近红色分布
    blue2red = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    blue2red_name = trans_name(filename, '蓝', '红')

    # 黑色转红色
    # 黑色三个通道都比较小， 用255-R通道，则R通道独大，其余通道小，接近红色的分布
    black2red = image.copy()
    black2red[:, :, 2] = 255 - black2red[:, :, 2]
    black2red_name = trans_name(filename, '黑', '红')

    # 黄色转红色
    # 黄色R、G值较大，B值较小，先取反，则R、G值较小, B值独大，接近蓝色分布
    yellow2red = image.copy()
    yellow2red = cv2.bitwise_not(image)
    yellow2red = cv2.cvtColor(yellow2red, cv2.COLOR_RGB2BGR)  # 与蓝色转红色相同
    yellow2red_name = trans_name(filename, '黄', '红')

    # 写入结果
    names = [red2red_name, blue2red_name, black2red_name, yellow2red_name]
    imgs = [image, blue2red, black2red, yellow2red]
    zips = [(name, img) for name, img in zip(names, imgs) if name is not None]
    res = np.empty((len(zips), *imgs[0].shape), dtype=np.uint8)
    global wc
    for i, (name, img) in enumerate(zips):
        wc += 1
        res[i] = img
    return res, [ele[0] for ele in zips]


train_root = '/home/chuan/captcha/data/raw/fake_pic_train'
test_root = '/home/chuan/captcha/data/raw/fake_pic_test'
train_output = '/home/chuan/dataset/captcha/train'
test_output = '/home/chuan/dataset/captcha/test'

workdir = train_root
outputdir = train_output
filenames = os.listdir(workdir)
filenames = [filename for filename in filenames if filename.endswith('.png')]
filenames.sort()
wc = 0
base = 500
# trans = partial(trans_image, root=workdir, output=outputdir)
# with Pool(processes=cpu_count()) as pool:
#     pool.map(trans, filenames[:base])

imgs = []
labels = []
for filename in filenames:
    trans_image(filename, workdir, outputdir)
    # res = trans_image(filename, workdir, outputdir)
    # imgs.append(res[0])
    # labels.extend(res[1])
print(wc)
warning = f"Expect {wc} pictures, write only {len(os.listdir(outputdir))}!"
assert wc == len(os.listdir(outputdir)), warning

# data = np.concatenate(imgs, axis=0)
# labels = np.array(labels)
# cwd = os.getcwd()
# os.chdir(outputdir)
# np.save("images.npy", data)
# np.save("labels.npy", labels)
# os.chdir(cwd)
