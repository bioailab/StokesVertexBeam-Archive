# Libraries

# from __future__ import print_function
# from __future__ import division
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy
# import numpy as np
# import torchvision
# from torchvision import datasets, models, transforms
# import matplotlib.pyplot as plt
# import time
import os
import copy
import pickle
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--design', default=3, type=int, choices = [1, 2, 3], help='The type of experiment design.')
    parser.add_argument('--model_name', default="densenet", type=str,
                        choices=["resnet", "alexnet", "vgg", "squeezenet", "densenet", "inception"],
                        help = "Model to employ")
    parser.add_argument('--use_pretrained', default="True", type = bool,
                        help = "Use Pretrained model on ImageNet")
    parser.add_argument('--feature_extract', default="False", type = bool,
                        help = "Flag for feature extracting. When False, we finetune the whole model otherwise we only update the reshaped layer params")
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training (change depending on how much memory you have).')
    parser.add_argument('--num_epochs', default=100, type=int, help='Number of epochs to train for.')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate of the model.')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum of the model.')
    parser.add_argument('--pretrain_simulated', default=False, type = bool, 
                        help = "If design is 2, this is a must variable. \
                        It is true in case the model need to be pretrain on simulated data otherwise False.")

    args = parser.parse_args()

    if args.design == 1:
        from Design1.DockerHelperAllModelFromPytorch import model_fit
    if args.design == 2:
        from Design2.DockerHelperAllModelFromPytorchForSimulated import model_fit_simulated
        from Design2.DockerHelperAllModelFromPytorchForExperimental import model_fit_experimental
        from Design2.DockerHelperAllModelFromPytorchForExperimental import initialize_model
    if args.design == 3:
        from Design3.DockerHelperAllModelFromPytorch import model_fit
        

    # Top level data directory. Here we assume the format of the directory conforms to the ImageFolder structure
    data_dir = os.getcwd() + "/data/Design" +  str(args.design) + "/"
    
    # # Classes
    # class_dir = ["Class1_phi12_01/", "Class2_phi12_02/", "Class3_phi12_10/",
    #             "Class4_phi12_12/", "Class5_phi12_23/", "Class6_phi23_01/",
    #             "Class7_phi23_02/", "Class8_phi23_10/", "Class9_phi23_12/",
    #             "Class10_phi23_23/", "Class11_phi31_01/", "Class12_phi31_02/",
    #             "Class13_phi31_10/", "Class14_phi31_12/", "Class1_phi31_23/"]
    # Number of classes in the dataset
    num_classes = 15

    result_addr = os.getcwd() + "/results/Design" + str(args.design) + "/" + args.model_name + ".data"
    # ------------------------------------------------------------------------
    # Model fit
    if args.design == 1 or args.design == 3:
        model_ft, hist = model_fit(data_dir, args.batch_size, args.model_name, num_classes, 
                                    args.feature_extract, args.use_pretrained, args.lr, args.momentum, args.num_epochs)
    if args.design == 2:
        hist = dict()
        if args.pretrain_simulated == True:
            print("Starting model training in the simulated data")
            model, hist, input_size = model_fit_simulated(data_dir, args.batch_size, args.model_name,
                                        num_classes, args.feature_extract, args.use_pretrained, 
                                        args.lr, args.momentum, args.num_epochs)
            input_size = 224
            print("Finished model training in the simulated data")
        else:
            model, input_size = initialize_model(args.model_name, num_classes, args.feature_extract, args.use_pretrained)

        hist["input_size"] = input_size
        fold_list = ["1stFold", "2ndFold", "3rdFold", "4thFold", "5thFold", "6thFold", "7thFold", "8thFold", "9thFold", "10thFold"]
        fold_dir = data_dir + "experimental/"
        for i in range(10):
            fold = fold_list[i]
            data_dir = fold_dir + fold
            print("Starting model training in the experimental data: ", fold)
            hist[fold] = model_fit_experimental(data_dir, args.batch_size, copy.deepcopy(model), 
                            input_size, args.lr, args.momentum, args.num_epochs, args.model_name)
    

    # ------------------------------------------------------------------------
    # Save result
    with open(result_addr, 'wb') as file:
        pickle.dump(hist, file)
