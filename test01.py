import scipy.io


from VisualizeDataset import VisualizeDataset

mat = scipy.io.loadmat('PA8Data.mat')
trainData = mat['trainData']
train_data_labels = trainData[0][0]
data = train_data_labels['data']
labels = train_data_labels['labels']

VisualizeDataset(data)


