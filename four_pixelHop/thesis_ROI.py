import numpy as np
import cv2
from scipy import signal

def getCost(cover):
    # 1D high pass decomposition filter
    hpdf = [-0.0544158422, 0.3128715909, -0.6756307363, 0.5853546837, 0.0158291053, -0.2840155430, -0.0004724846,
            0.1287474266, 0.0173693010, -0.0440882539,
            -0.0139810279, 0.0087460940, 0.0048703530, -0.0003917404, -0.0006754494, -0.0001174768]
    hpdf = np.array(hpdf)

    # 1D low pass decomposition filter
    lpdf = [(-1) ** i for i in range(len(hpdf))] * hpdf[::-1]
    # construction of 2D wavelet filters
    F = [0 for _ in range(3)]
    F[0] = lpdf[None, :] * hpdf[:, None]
    F[1] = hpdf[None, :] * lpdf[:, None]
    F[2] = hpdf[None, :] * hpdf[:, None]

    # Get embedding costs
    # initialization
    cover = cover.astype("double")
    p = -1
    wetCost = 10 ^ 10
    sizeCover = cover.shape

    # add padding
    padSize = np.max((F[0].shape, F[1].shape, F[2].shape))
    coverPadded = cv2.copyMakeBorder(cover, padSize, padSize, padSize, padSize, cv2.BORDER_REFLECT)

    # compute directional residual and suitability \xi for each filter
    xi = [0 for _ in range(3)]
    for fIndex in range(3):
        # compute residual
        R = signal.convolve2d(coverPadded,  F[fIndex], boundary='symm', mode='same')

        # compute suitability
        xi[fIndex] = signal.convolve2d(abs(R), np.rot90(abs(F[fIndex]), 2), boundary='symm', mode='same');
        # correct the suitability shift if filter size is even
        if F[fIndex].shape[0] % 2 == 0:
            xi[fIndex] = np.roll(xi[fIndex], 1, axis=0)
        if F[fIndex].shape[1] % 2 == 0:
            xi[fIndex] = np.roll(xi[fIndex], 1, axis=1)
        # remove padding
        xi[fIndex] = xi[fIndex][(xi[fIndex].shape[0] - sizeCover[0]) // 2: (xi[fIndex].shape[0] - sizeCover[0]) // 2 + sizeCover[0],
                                (xi[fIndex].shape[1] - sizeCover[1]) // 2: (xi[fIndex].shape[1] - sizeCover[1]) // 2 + sizeCover[1]]
    # compute embedding costs \rho
    rho = ((xi[0] ** p) + (xi[1] ** p) + (xi[2] ** p)) ** (-1 / p)

