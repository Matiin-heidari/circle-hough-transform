import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import maximum_filter

def smoothen(img,display):
    #Using a 3x3 gaussian filter to smoothen the image
    gaussian = np.array([[1/16.,1/8.,1/16.],[1/8.,1/4.,1/8.],[1/16.,1/8.,1/16.]])
    img.load(img.convolve(gaussian))
    if display:
        img.disp
    return img

def edge(img,threshold,display=False):
    #Using a 3x3 Laplacian of Gaussian filter along with sobel to detect the edges
    laplacian = np.array([[1,1,1],[1,-8,1],[1,1,1]])
    #Sobel operator (Orientation = vertical)
    sobel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

    #Generating sobel horizontal edge gradients
    G_x = img.convolve(sobel)

    #Generating sobel vertical edge gradients
    G_y = img.convolve(np.fliplr(sobel).transpose())

    #Computing the gradient magnitude
    G = pow((G_x*G_x + G_y*G_y),0.5)

    G[G<threshold] = 0
    L = img.convolve(laplacian)
    if L is None:
        return
    (M,N) = L.shape

    temp = np.zeros((M+2,N+2))
    temp[1:-1,1:-1] = L
    result = np.zeros((M,N))
    for i in range(1,M+1):
        for j in range(1,N+1):
            if temp[i,j]<0:
                for x,y in (-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1):
                        if temp[i+x,j+y]>0:
                            result[i-1,j-1] = 1
    img.load(np.array(np.logical_and(result,G),dtype=np.uint8))
    if display:
        img.disp
    return img


def detectCircles(img, threshold, region, radius=None):
    (M, N) = img.shape
    if radius is None:
        R_max = min(100, np.max((M, N)))  # Limit R_max for efficiency
        R_min = 3
    else:
        [R_max, R_min] = radius

    R = R_max - R_min
    A = np.zeros((R_max, M + 2 * R_max, N + 2 * R_max))
    B = np.zeros((R_max, M + 2 * R_max, N + 2 * R_max))

    theta = np.arange(0, 360) * np.pi / 180
    edges = np.argwhere(img[:, :])

    for val in range(R):
        r = R_min + val
        bprint = np.zeros((2 * (r + 1), 2 * (r + 1)))
        (m, n) = (r + 1, r + 1)

        for angle in theta:
            x = int(np.round(r * np.cos(angle)))
            y = int(np.round(r * np.sin(angle)))
            if 0 <= m + x < bprint.shape[0] and 0 <= n + y < bprint.shape[1]:  # Boundary check
                bprint[m + x, n + y] = 1

        constant = np.argwhere(bprint).shape[0]

        for x, y in edges:
            X = [x - m + R_max, x + m + R_max]
            Y = [y - n + R_max, y + n + R_max]

            X0, X1 = max(0, X[0]), min(A.shape[1], X[1])
            Y0, Y1 = max(0, Y[0]), min(A.shape[2], Y[1])

            bX0, bX1 = max(0, -X[0]), min(bprint.shape[0], A.shape[1] - X[0])
            bY0, bY1 = max(0, -Y[0]), min(bprint.shape[1], A.shape[2] - Y[0])

            A[r, X0:X1, Y0:Y1] += bprint[bX0:bX1, bY0:bY1]

        A[r] = np.where(A[r] < threshold * constant / r, 0, A[r])  # Proper thresholding

    local_max = maximum_filter(A, size=region)  # Faster local maxima detection
    B = (A == local_max) * A

    return B[:, R_max:-R_max, R_max:-R_max]  # Remove padding for final result


def displayCircles(A, file_path):
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib

    fig, ax = plt.subplots()
    ax.imshow(img)

    circleCoordinates = np.argwhere(A)
    for r, x, y in circleCoordinates:
        ax.add_artist(plt.Circle((y, x), r, color=(1, 0, 0), fill=False))

    plt.show()