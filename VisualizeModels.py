import numpy as np
import matplotlib.pyplot as plt

from SamplePose import SamplePose
from ShowPose import ShowPose

close_by_user = False

def on_close(event):
    print('Closed Figure!')
    global close_by_user
    close_by_user = True

def VisualizeModels(P, G):
    #print(G.shape)
    K = len(P['c'])

    fig, axs = plt.subplots(1, K)
    fig.canvas.mpl_connect('close_event', on_close)
    while True:
        for k in range(K):
            if len(G.shape) == 2: #  same graph structure for all classes
                pos = SamplePose(P, G, k)
            else:
                pose = SamplePose(P, G[:, :, k], k)

            img = ShowPose(pose)
            axs[k].cla()
            axs[k].imshow(img, cmap='viridis')
            plt.pause(1)
            if close_by_user:
                return

