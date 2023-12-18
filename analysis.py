import json
import numpy as np
from matplotlib import pyplot as plt
import os


filePath1 = "C:\\Users\\evanj\\Desktop\\School\\Graduate\\Fall2023\\ComputerVision\\JigsawTermProject\\JigsawTrainingData\\using\\"
# filePath1 = "C:\\Users\\evanj\\Desktop\\School\\Graduate\\Fall2023\\ComputerVision\\JigsawTermProject\\JigsawTrainingData\\pieces\\test1"
filePath2 = "C:\\Users\\evanj\\Desktop\\School\\Graduate\\Fall2023\\ComputerVision\\JigsawTermProject\\outputs\\"
# filePath2 = "C:\\Users\\evanj\\Desktop\\School\\Graduate\\Fall2023\\ComputerVision\\JigsawTermProject\\"
fnames = os.listdir(filePath2)
# print(fnames)
pcts = []
num_pieces = []
# fnames = ['test.json']
for name in fnames:
    name = name[:-5]
    file = filePath1 + name + '\\adjacent.json'
    # file = filePath1 + '\\adjacent.json'
    file2 = filePath2 + name + '.json'
    f = open(file)
    json_file = json.load(f)
    adj_truth = np.zeros((len(json_file),len(json_file)))
    adj_gen = np.zeros((len(json_file),len(json_file)))
    for i in json_file:
        for j in json_file[i]:
            adj_truth[int(i)][int(j)] = 1
    # plt.figure(1)
    # plt.imshow(adj_truth)

    
    f = open(file2)
    json_file = json.load(f)
    for i in json_file:
        for j in json_file[i]:
            adj_gen[int(i)][int(j)] = 1

    # plt.figure(2)
    # plt.imshow(adj_gen)

    n=0
    p=0
    diff = np.zeros((len(json_file),len(json_file),3),dtype=np.uint8)

    for i in range(len(json_file)):
        for j in range(len(json_file)):
            if adj_truth[i][j] == adj_gen[i][j] and diff[i][j].all()==0:
                diff[i][j] =  (20,205,30)
                p+=1
            else:
                diff[i][j] = (255,0,0)
                diff[j][i] = (255,0,0)
                n+=1
    # plt.figure(3)
    # plt.imshow(diff)
    # plt.show()
    num_pieces.append(len(json_file))
    pct = 100*n/(n+p)
    pcts.append(pct)
    print(pct)

# print("\nError:")
# print(sum(pcts)/len(pcts))
# print("\nAvg Piece #:")
# print(sum(num_pieces)/len(num_pieces))
