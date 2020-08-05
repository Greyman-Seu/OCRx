import alphabet

# about data and net
alphabet = alphabet.alphabet
alphabet += "abcdefghijklmnopqrstuvwxyz0123456789"  # chuan
keep_ratio = False  # whether to keep ratio for image resize
manualSeed = 1234  # reproduce experiemnt
random_sample = True  # whether to sample the dataset with random sampler
imgH = 32  # the height of the input image to network
imgW = 90  # the width of the input image to network
nh = 256  # size of the lstm hidden state
crop = True
# nc = 1
nc = 3
pretrained = ''  # path to pretrained model (to continue training)
expr_dir = 'expr'  # where to store samples and models
dealwith_lossnan = False  # whether to replace all nan/inf in gradients to zero

# hardware
cuda = True  # enables cuda
multi_gpu = False  # whether to use multi gpu
ngpu = 1  # number of GPUs to use. Do remember to set multi_gpu to True!
workers = 4  # number of data loading workers

# training process
displayInterval = 100  # interval to be print the train loss
valInterval = 1000  # interval to val the model loss and accuray
saveInterval = 1000  # interval to save model
n_val_disp = 10  # number of samples to display when val the model

# finetune
nepoch = 50  # number of epochs to train for
batchSize = 64  # input batch size
lr = 0.0005  # learning rate for Critic, not used by adadealta
beta1 = 0.5  # beta1 for adam. default=0.5
adam = False  # whether to use adam (default is rmsprop)
adadelta = False  # whether to use adadelta (default is rmsprop)

# logfile
log = "log.txt"
