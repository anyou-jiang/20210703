import scipy.io
import matplotlib.pyplot as plt
from LearnGraphAndCPDs import LearnGraphAndCPDs
from SamplePose import SamplePose
from ShowPose import ShowPose
import numpy as np

mat = scipy.io.loadmat('PA8Data.mat')
trainData = mat['trainData']
train_data_labels = trainData[0][0]
trainData_data = train_data_labels['data']
trainData_labels = train_data_labels['labels']

[P3, G3, likelihood3] = LearnGraphAndCPDs(trainData_data, trainData_labels)

np.random.seed(0)
pose = SamplePose(P3, G3, -1)

img = ShowPose(pose)

plt.cla()
plt.imshow(img, cmap='viridis')