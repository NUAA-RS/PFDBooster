
class args():

    # training args
    epochs = 10 #"number of training epochs, default is 2"
    batch_size = 2
    dataset = "./dataset"
    #trainNumber = 26295
    trainNumber = 84000
    #trainNumber = 40000
    HEIGHT = 256
    WIDTH = 256
    PATCH_SIZE = 128
    PATCH_STRIDE = 4

    save_model_dir = "models" #"path to folder where trained model will be saved."
    save_loss_dir = "models/loss"  # "path to folder where trained model will be saved."

    image_size = 256 #"size of training images, default is 256 X 256"
    cuda = 1 #"set it to 1 for running on GPU, 0 for CPU"
    seed = 42 #"random seed for training"
    ssim_weight = [1,10,100,1000,10000]
    ssim_path = ['1e0', '1e1', '1e2', '1e3', '1e4']

    lr = 1e-4 #"learning rate, default is 1e-4"
    lr_light = 1e-4  # "learning rate, default is 0.001"
    log_interval = 2 #"number of images after which the training loss is logged, default is 500"
    resume = None
    resume_auto_en = None
    resume_auto_de = None
    resume_auto_fn = None
    device = 0
    path = "Epoch_3_iters_3600_"

    model_path_gray = "./models/"+path+".model"
    #model_path_gray = "stage1.model"
    model_path_rgb = "./models/"+path+".model"



