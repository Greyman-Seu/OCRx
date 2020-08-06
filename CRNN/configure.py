# about data and net
with open("alphabet.txt", 'r') as f:
    text = f.readline()
color = "红黄黑蓝"
label_type = "text"
alphabet = {"text": text, "color": color, "both": (text, color)}[label_type]
manualSeed = 1234
imgH = 32  # the height of the input image to network
imgW = 90  # the width of the input image to network
crop = True
keep_ratio = False  # whether to keep ratio for image resize
nh = 256  # size of lstm hidden state
# nh = 128  # crnn_sim
nc = 3
weight_dir = 'weights'  # where to store model parameters
pretrained = ''  # path to pretrained model
dealwith_lossnan = False  # whether to replace all nan/inf in gradients to zero

# hardware
use_cuda = True  # enables cuda
multi_gpu = False  # whether to use multi gpu
ngpu = 1  # number of GPUs to use. Do remember to set multi_gpu to True!
workers = 4  # number of data loading workers

# training process
use_lr_scheduler = True
milestones = [10, 20]
gamma = 0.1
displayInterval = 100  # batch interval to print the training loss
testInterval = 1500  # batch interval to test the model on test set
nTestDisplay = 10  # number of samples to display when test the model

# finetune
nepoch = 100  # number of epochs to train with
batchSize = 64  # input batch size
lr = 0.0005  # learning rate for Critic, not used by adadealta
adam = False  # whether to use adam (default is rmsprop)
beta1 = 0.5  # beta1 for adam, default=0.5
adadelta = False  # whether to use adadelta (default is rmsprop)
