import scipy.io
from LearnCPDsGivenGraph import LearnCPDsGivenGraph
from ClassifyDataset import ClassifyDataset


mat = scipy.io.loadmat('PA8Data.mat')
G2 = mat['G2']

trainData = mat['trainData']
train_data_labels = trainData[0][0]
trainData_data = train_data_labels['data']
trainData_labels = train_data_labels['labels']
P2, likelihood2 = LearnCPDsGivenGraph(trainData_data, G2, trainData_labels)

testData = mat['testData']
test_data_labels = testData[0][0]
test_data = test_data_labels['data']
test_labels = test_data_labels['labels']

accuracy2 = ClassifyDataset(test_data, test_labels, P2, G2)
