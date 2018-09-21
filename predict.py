import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms, models
import argparse
import numpy as np
import json
import os
import random
from PIL import Image
from utils import load_checkpoint, load_cat_names

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', action='store', default='checkpoint.pth')
    parser.add_argument('--top_k', dest='top_k', default='5')
    parser.add_argument('--filepath', dest='filepath', default=None)
    parser.add_argument('--category_names', dest='category_names', default='cat_to_name.json')
    parser.add_argument('--gpu', action='store_true', default=True)
    return parser.parse_args()

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    expects_mean = [0.485, 0.456, 0.406]
    expects_std = [0.229, 0.224, 0.225]
    
    pil_image = Image.open(image)
    
    pil_image = pil_image.resize((256,256))
    
    value = 0.5*(256-224)
    pil_image = pil_image.crop((value,value,256-value,256-value))
    # Normalize:  0-255, but the model expected floats 0-1
    # Convert image to an array and divide each element
    pil_image = np.array(pil_image)/255

    pil_image = (pil_image - expects_mean) / expects_std
    
    return pil_image.transpose(2,0,1)
    

def predict(image_path, model, topk, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    
    cuda = torch.cuda.is_available()
    if gpu and cuda:
        model = model.cuda()
    else:
        model = model.cpu()
    
    image = process_image(image_path)
    
    # tranfer to tensor
    image = torch.from_numpy(np.array([image])).float()
    
    # The image becomes the input
    image = Variable(image)
    
    if gpu and cuda:
        image = Variable(image.cuda())
    else:       
        image = Variable(image)
        
    output = model.forward(image)
    
    probabilities = torch.exp(output).data
    
    # getting the topk (=5) probabilites and indexes
    # 0 -> probabilities
    # 1 -> index
    prob = torch.topk(probabilities, topk)[0].tolist()[0] # probabilities
    index = torch.topk(probabilities, topk)[1].tolist()[0] # index
    
    ind = []
    for i in range(len(model.class_to_idx.items())):
        ind.append(list(model.class_to_idx.items())[i][0])

    # transfer index to label
    label = []
    for i in range(5):
        label.append(ind[index[i]])
    return prob, label

def main(): 
    args = parse_args()
    gpu = args.gpu
    model = load_checkpoint(args.checkpoint)
    cat_to_name = load_cat_names(args.category_names)
    if args.filepath == None:
        img_num = random.randint(1, 102)
        image = random.choice(os.listdir('./flowers/test/' + str(img_num) + '/'))
        image_path = './flowers/test/' + str(img_num) + '/' + image
        prob, classes = predict(image_path, model, int(args.top_k), gpu)
        print('Image selected: ' + str(cat_to_name[str(img_num)]))
    else:
        image_path = args.filepath
        prob, classes = predict(image_path, model, int(args.top_k), gpu)
        print('File selected: ' + image_path)
    print(prob)
    print(classes)
    print([cat_to_name[x] for x in classes])

if __name__ == "__main__":
    main()