import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

import numpy as np
class Gkernel():
    '''
    Focused region gradient operator class
    '''
    def __init__(self,kernelsize=3 ,kernelkind=0):
        super(Gkernel, self).__init__()
        self.ks=kernelsize
        self.kk=kernelkind
    def meanblur(self,img):
        img_mean = cv2.blur(img, (self.ks,self.ks))
        return img_mean

    def calculate_psnr(self,img1,
                       img2,
                       crop_border=4,):
        """Calculate PSNR (Peak Signal-to-Noise Ratio).

        Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

        Args:
            img1 (ndarray): Images with range [0, 255].
            img2 (ndarray): Images with range [0, 255].
            crop_border (int): Cropped pixels in each edge of an image. These
                pixels are not involved in the PSNR calculation.
            input_order (str): Whether the input order is 'HWC' or 'CHW'.
                Default: 'HWC'.
            test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

        Returns:
            float: psnr result.
        """

        assert img1.shape == img2.shape, (
            f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
        img1 = np.array(img1)
        img2 = np.array(img2)
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)

        if crop_border != 0:
            img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
            img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20. * np.log10(255. / np.sqrt(mse))

    def _ssim(self,img1, img2):
        """Calculate SSIM (structural similarity) for one channel images.

        It is called by func:`calculate_ssim`.

        Args:
            img1 (ndarray): Images with range [0, 255] with order 'HWC'.
            img2 (ndarray): Images with range [0, 255] with order 'HWC'.

        Returns:
            float: ssim result.
        """

        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) *
                    (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                           (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    def calculate_ssim(self,img1,
                       img2,
                       crop_border=4):
        """Calculate SSIM (structural similarity).

        Ref:
        Image quality assessment: From error visibility to structural similarity

        The results are the same as that of the official released MATLAB code in
        https://ece.uwaterloo.ca/~z70wang/research/ssim/.

        For three-channel images, SSIM is calculated for each channel and then
        averaged.

        Args:
            img1 (ndarray): Images with range [0, 255].
            img2 (ndarray): Images with range [0, 255].
            crop_border (int): Cropped pixels in each edge of an image. These
                pixels are not involved in the SSIM calculation.
            input_order (str): Whether the input order is 'HWC' or 'CHW'.
                Default: 'HWC'.
            test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

        Returns:
            float: ssim result.
        """

        assert img1.shape == img2.shape, (
            f'Image shapes are differnet: {img1.shape}, {img2.shape}.')

        img1 = np.array(img1)
        img2 = np.array(img2)
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)

        if crop_border != 0:
            img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
            img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

        ssims = self._ssim(img1, img2)

        return ssims

