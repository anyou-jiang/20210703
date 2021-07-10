import matplotlib.pyplot as plt
from ShowPose import ShowPose

close_by_user = False

def on_close(event):
    print('Closed Figure!')
    global close_by_user
    close_by_user = True

def VisualizeDataset(Dataset):
    global close_by_user
    close_by_user = False
    fig = plt.figure()
    fig.canvas.mpl_connect('close_event', on_close)

    n_samples = Dataset.shape[0]
    for n in range(n_samples):
        pose = Dataset[n]
        img = ShowPose(pose)
        plt.cla()
        plt.imshow(img, cmap='viridis')
        plt.pause(0.01)
        if close_by_user:
            break