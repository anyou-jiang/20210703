import numpy as np
from math import pi
from math import sin
from math import cos
import matplotlib.pyplot as plt
import scipy.io
from func_DrawLine import func_DrawLine



def ShowPose(pose):
    '''
    
    :param pose: 10 x 3: body parts, each part is (y, x, alpha) 
    :return: 
    '''

    pose[:, 0] = pose[:, 0] + 100
    pose[:, 1] = pose[:, 1] + 150

    part_length = [60, 20, 32, 33, 32, 33, 46, 49, 46, 49]
    part_width = [18, 10, 7, 5, 7, 5, 10, 7, 10, 7]

    img = np.zeros((300, 300))
    for part in range(10):
        startpt = np.round(pose[part, 0:2]).astype(int)
        axis = np.array([sin(pose[part, 2] - pi / 2), cos(pose[part, 2] - pi / 2)])
        xaxis = np.array([cos(pose[part, 2] - pi / 2), -sin(pose[part, 2] - pi / 2)])
        endpt = np.round(startpt + part_length[part] * axis).astype(int)
        corner1 = np.round(startpt + xaxis * part_width[part]).astype(int)
        corner2 = np.round(startpt - xaxis * part_width[part]).astype(int)
        corner3 = np.round(endpt + xaxis * part_width[part]).astype(int)
        corner4 = np.round(endpt - xaxis * part_width[part]).astype(int)

        img = func_DrawLine(img, corner1[0], corner1[1], corner2[0], corner2[1], 1)
        img = func_DrawLine(img, corner1[0], corner1[1], corner3[0], corner3[1], 1)
        img = func_DrawLine(img, corner4[0], corner4[1], corner2[0], corner2[1], 1)
        img = func_DrawLine(img, corner4[0], corner4[1], corner3[0], corner3[1], 1)

        if startpt[0] > 2 and startpt[0] < 297 and startpt[1] > 2 and startpt[1] < 297:
            img[startpt[0]-3 : startpt[0]+3, startpt[1]-3 : startpt[1]+3] = 1

    return img




if __name__ == "__main__":
    mat = scipy.io.loadmat('PA9SampleCases.mat')
    example_input = mat['exampleINPUT']
    t1a1 = example_input['t1a1'][0][0]
    n_samples = t1a1.shape[0]
    for n in range(n_samples):
        pose = t1a1[n]
        img = ShowPose(pose)
        plt.cla()
        plt.imshow(img, cmap='viridis')
        plt.pause(1)



    # pose = np.array([
    #     [-0.1402, -0.3467, 2.8965],
    #     [- 0.1402, -0.3467, 0.0791],
    #     [3.3587, -15.7480, -3.1052],
    #     [29.9399, -16.7150, 0.5779],
    #     [3.0640, 17.7353, 2.7774],
    #     [29.7283, 27.9010, -0.8816],
    #     [56.6461, -5.7461, 3.0578],
    #     [105.1068, -1.6763, 3.1257],
    #     [55.4323, 14.4201, 3.0302],
    #     [103.3400, 19.7803, 3.0704],
    # ])
    # img = show_pose(pose)
    # plt.imshow(img, cmap='viridis')
    # plt.show()
    #
    # plt.close()

