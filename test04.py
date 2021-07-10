import scipy.io
from LearnGraphAndCPDs import LearnGraphAndCPDs
from ClassifyDataset import ClassifyDataset
from VisualizeModels import VisualizeModels

mat = scipy.io.loadmat('PA8Data.mat')
trainData = mat['trainData']
train_data_labels = trainData[0][0]
trainData_data = train_data_labels['data']
trainData_labels = train_data_labels['labels']

[P3, G3, likelihood3] = LearnGraphAndCPDs(trainData_data, trainData_labels)

testData = mat['testData']
test_data_labels = testData[0][0]
test_data = test_data_labels['data']
test_labels = test_data_labels['labels']

accuracy3 = ClassifyDataset(test_data, test_labels, P3, G3)

VisualizeModels(P3, G3)
