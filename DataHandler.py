from sklearn.model_selection import train_test_split
import numpy as np
import gzip
import pickle

class MnistData:
    def __init__(self, batch_size, num_classes,path,input_shape= (1,28,28), image_format="channel_first"):

        self.batch_size = batch_size
        self.input_shape = input_shape

        f = gzip.open(path, 'rb')
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        (self.trainImage,self.trainLabel), (self.valImage,self.valLabel), (self.testImage,self.testLabel) = u.load()
        f.close()

        X = np.concatenate((self.trainImage,self.testImage,self.valImage))
        y = np.concatenate((self.trainLabel,self.testLabel,self.valLabel))

        X = X.reshape((X.shape[0],) + self.input_shape)
        X = X.astype('float32')
        X /= 255

        y = np.eye(num_classes)[y.astype(np.int)].astype(np.int)
        
        self.trainImage,self.testImage, self.trainLabel, self.testLabel = train_test_split(X, y, test_size=10000, random_state=1)
        self.trainImage, self.valImage, self.trainLabel, self.valLabel = train_test_split(self.trainImage, self.trainLabel, test_size=10000, random_state=1)

        self.trainLen = len(self.trainImage)

        """self.trainImage = self.trainImage[:32]
        self.trainLabel = self.trainLabel[:32]
        self.valImage = self.valImage[:10]
        self.valLabel = self.valLabel[:10]
        self.testImage = self.testImage[:10]
        self.testLabel = self.testLabel[:10]
        self.batch_size = 5"""



        self.currentBatch = 0
        self.num_iterations = int(np.ceil(self.trainLen / self.batch_size))
        self.num_iterations_floor = int(np.floor(self.trainLen / self.batch_size))
        self.shuffleIndex()

    def shuffleIndex(self):
        self.currentBatch = 0
        Idx = np.arange(self.trainLen)
        np.random.shuffle(Idx)
        self.IdxMat = np.random.choice(Idx, self.num_iterations_floor * self.batch_size, replace=False)
        self.IdxFinal = np.setdiff1d(Idx, self.IdxMat)
        self.IdxMat = self.IdxMat.reshape(self.num_iterations_floor, self.batch_size)


    def forward(self):
        if self.currentBatch < self.num_iterations_floor:
            idx = self.IdxMat[self.currentBatch]
            self.currentBatch+=1
        else:
            idx = self.IdxFinal

        return self.trainImage[idx, :], self.trainLabel[idx, :]

    def get_test_set(self):

        idx = np.arange(self.testImage.shape[0])
        np.random.shuffle(idx)
        return self.testImage[idx], self.testLabel[idx]

    def get_val_set(self):
        idx = np.arange(self.valImage.shape[0])
        np.random.shuffle(idx)
        return self.valImage[idx], self.valLabel[idx]

    def get_train_set(self):
        idx = np.arange(self.trainImage.shape[0])
        np.random.shuffle(idx)
        return self.trainImage[idx], self.trainLabel[idx]

    def get_num_batches(self):
        return self.num_iterations

    def get_batch_size(self):
        return self.batch_size

    def get_input_shape(self):
        return self.input_shape

