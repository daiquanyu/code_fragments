from scipy.sparse import csc_matrix
from sklearn.neighbors import kneighbors_graph
import scipy.io as sio
import numpy as np



import numpy as np
import scipy.io as sio

datasets = ['books', 'dvd', 'electronics', 'kitchen']

for dataset in datasets:
    attrb = []
    group = []
    
    # training file
    fid = open('{}_train.svmlight'.format(dataset), 'r')
    line = fid.readline().strip()
    while line:
        line = line.split()
        label = int(line[0])
        label = [1, 0] if label==1 else [0, 1]
        bag_of_words = np.zeros((1, 5000))
        for j in range(len(line)-1):
            word_freq = line[j+1].split(':')
            bag_of_words[0, int(word_freq[0])] = int(word_freq[1])
        attrb.append(bag_of_words)
        group.append(label)
        line = fid.readline().strip()
    fid.close()
        
    # testing file
    fid = open('{}_test.svmlight'.format(dataset), 'r')
    line = fid.readline().strip()
    while line:
        line = line.split()
        label = int(line[0])
        label = [1, 0] if label==1 else [0, 1]
        bag_of_words = np.zeros((1, 5000))
        for j in range(len(line)-1):
            word_freq = line[j+1].split(':')
            bag_of_words[0, int(word_freq[0])] = int(word_freq[1])
        attrb.append(bag_of_words)
        group.append(label)
        line = fid.readline().strip()
    fid.close()
    
    attrb = np.concatenate(attrb, axis=0)
    group = np.array(group)
    
    print(attrb.shape)
    print(group.shape)
    
    sio.savemat('{}.mat'.format(dataset), {'attrb': attrb, 'group': group})

            
##################################################################################


datasets = ['books', 'dvd', 'kitchen', 'electronics']

for dataset in datasets:
    attrb = sio.loadmat('{}.mat'.format(dataset))['attrb']
    group = sio.loadmat('{}.mat'.format(dataset))['group']

    max_freq = np.max(attrb, 1)
    max_freq[np.where(max_freq==0)] = 1
    most_freq = np.concatenate([np.reshape(1/max_freq, (-1, 1))]*attrb.shape[1], axis=1)
    TF = np.multiply(attrb, most_freq)

    IDF = np.log(attrb.shape[0]/(np.sum(np.array(attrb>0.01, np.int32), axis=0)+1))
    IDF = np.concatenate([np.reshape(IDF, (1, -1))]*attrb.shape[0], axis=0)

    TF_IDF = np.multiply(TF, IDF)
    network = kneighbors_graph(TF_IDF, 5, mode='connectivity', metric='cosine', include_self=False)

    network = csc_matrix(network)
    attrb = csc_matrix(attrb)

    print(dataset)
    print(network.shape)
    print(attrb.shape)
    print(group.shape)

    sio.savemat('{}.mat'.format(dataset), {'network': network, 'group': group, 'attrb': attrb})
    