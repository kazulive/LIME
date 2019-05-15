# -*- coding: utf-8 -*-

import numpy as np
import scipy.spase
import scipy.sparse.linalg
import cv2
import sys

# ゼロ除算を回避するパラメータ
lime_eps = 1e-3

# limeのmain処理
class LIME:
    def __init__(self, alpha, gamma, rho, win_size, weight_strategy):
        self.alpha = alpha
        self.gamma = gamma
        self.rho = rho
        self.win_size = win_size
        self.weight_strategy = weight_strategy

    def solve_linear_equation(self, img, Th, Wx, Wy):
        """
        :param Th: 初期 T^, shape=(h, w)
        :param Wx: 式(19)によるWd(x) (horizontal), shape=(h, w)
        :param Wy: 式(19)によるWd(x) (vertical), shape=(h, w)
        """
        H, W = img.shape[:2]
        N = H * W

        # ベクトル化
        Th_vec = Th.flatten('C')

        # 式(19)はAx=b (x=t, b=t~)で表現可能
        dx = self.alpha * Wx
        dy = self.alpha * Wy
        dxa = np.hstack([dx[:, -1].reshape(-1, 1), dy[:, 0:-1]])
        dya = np.vstack()

    def lime(self, img):
        # 画像サイズ確保
        H, W = img.shape[:2]

        # ダウンサンプリング
        scale_factor = 0.5
        dH = int(H * scale_factor)
        dW = int(W * scale_factor)
        down_img = cv2.resize(img, (dH, dW), interpolation=cv2.INTER_AREA)

        # 画素値の正規化 [0, 1]
        down_img = down_img / 255.

        # Bright Channel による初期Tの推定
        Th = np.max(down_img, axis = 2)

        # ∇Tの計算
        Th_h = np.hstack([np.diff(Th, axis=1), (Th[:, 0] - Th[0, -1]).reshape(-1, 1)])
        Th_v = np.vstack([np.diff(Th, axis=0), (Th[0, :] - Th[-1, :]).reshape(1, -1)])

        # 行列Wの構成 (パターンは3つ)
        if self.weight_strategy == 1:
            Wh = np.ones((dH, dW))
            Wv = np.ones((dH, dW))

        elif self.weight_strategy == 2:
            Wh = 1. / (np.abs(Th_h) + lime_eps)
            Wv = 1. / (np.abs(Th_v) + lime_eps)

        elif self.weight_strategy == 3:
            hk = np.exp(-np.array(range(self.win_size)) / (2. * (self.rho ** 2))).reshape(1, self.win_size)
            vk = hk.reshapw(self.win_size, -1)
            sum_val = np.sum(hk)
            Wh = sum_val / (np.abs(cv2.filter2D(Th_h, -1, hk, anchor=(0, 0))) + lime_eps)
            Wv = sum_val / (np.abs(cv2.filter2D(Th_v, -1, vk, anchor=(0, 0))) + lime_eps)

        # Wを計算
        Wx = Wh / (np.abs(Th_h) + lime_eps)
        Wy = Wv / (np.abs(Th_v) + lime_eps)

        # 照明画像を更新
        T = self.solve_linear_equation(Th, Wx, Wy)

        # γ関数
        T = np.power(T, self.gamma)
        T = np.expand_dims(T, axis=2) + lime_eps

        R = down_img / T

        # float -> uint8
        R = R * 255
        # [0, 255]に範囲設定
        R = np.clip(R, 0, 255)
        R = np.fix(R).astype(dtype=np.uint8)

        return R