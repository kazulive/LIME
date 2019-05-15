# -*- coding: utf-8 -*-

"""
z = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])
z.reshape(-1, 2)
array([[1, 2],
       [3, 4],
       [5, 6],
       [7, 8],
       [9, 10],
       [11, 12]])
"""
import numpy as np
import scipy.sparse
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

    def solve_linear_equation(self, Th, Wx, Wy):
        """
        :param Th: 初期 T^, shape=(h, w)
        :param Wx: 式(19)によるWd(x) (horizontal), shape=(h, w)
        :param Wy: 式(19)によるWd(x) (vertical), shape=(h, w)
        """
        H, W = Th.shape[:2]
        N = H * W

        # ベクトル化
        Th_vec = Th.flatten('C')

        # 式(19)はAx=b (x=t, b=t~)で表現可能
        dx = self.alpha * Wx
        dy = self.alpha * Wy
        dxa = np.hstack([dx[:, -1].reshape(-1, 1), dx[:, 0:-1]]) # dx ahead
        dya = np.vstack([dy[-1, :].reshape(1, -1), dy[0:-1, :]]) # dy ahead

        # ベクトル化
        dy_vec = dy.flatten('C')
        dx_vec = dx.flatten('C')
        dxa_vec = dxa.flatten('C')
        dya_vec = dya.flatten('C')

        dyd1 = -np.vstack([dy[-1, :].reshape(1, -1), np.zeros((H-1, W))]).flatten('C')
        dyd2 = -np.vstack([dya[1:, :], np.zeros((1, W))]).flatten('C')
        dyd3 = -np.vstack([np.zeros((1, W)), dy[0:-1, :]]).flatten('C')
        dyd4 = -np.vstack([np.zeros((H-1, W)), dya[0, :].reshape(1, -1)]).flatten('C')
        ay = scipy.sparse.spdiags(np.array([dyd1, dyd2, dyd3, dyd4]), np.array([-N+W, -W, W, N-W]), N, N)

        dxd1 = -np.hstack([dx[:, -1].reshape(-1, 1), np.zeros((H, W - 1))]).flatten('C')
        dxd2 = -np.hstack([dxa[:, 1:], np.zeros((H, 1))]).flatten('C')
        dxd3 = -np.hstack([np.zeros((H, 1)), dx[:, 0:-1]]).flatten('C')
        dxd4 = -np.hstack([np.zeros((H, W - 1)), dxa[:, 0].reshape(-1, 1)]).flatten('C')
        ax = scipy.sparse.spdiags(np.array([dxd1, dxd2, dxd3, dxd4]), np.array([-W + 1, -1, 1, W - 1]), N, N)

        dig = scipy.sparse.spdiags(np.array([dx_vec + dy_vec + dxa_vec + dya_vec + 1]), np.array([0]), N, N)
        a = ax + ay + dig

        # 逆行列Aの近似
        m = scipy.sparse.linalg.spilu(a.tocsc())
        # 線形関数を構成
        m2 = scipy.sparse.linalg.LinearOperator((N, N), m.solve)
        # 前処理付き共役勾配法
        T, info = scipy.sparse.linalg.bicgstab(a, Th_vec, tol=1e-1, maxiter=2000, M=m2)

        if info != 0:
            print("収束不可能でした")

        T = T.reshape((H, W), order='C')

        T = np.clip(T, 0, sys.maxsize)
        T = T / (np.max(T) + lime_eps)

        return T

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
        Th = np.max(down_img, axis=2)

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

        R = cv2.resize(R, (W, H), interpolation=cv2.INTER_CUBIC)
        return R

if __name__ == '__main__':
    img = cv2.imread("01.bmp")
    cv2.imshow("input", img)
    cv2.waitKey(0)
    #hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #h, s, v = cv2.split(hsv)
    output = LIME(alpha=0.5, gamma=0.8, rho=2, win_size=5, weight_strategy=2).lime(img)
    cv2.imshow("output", output)
    cv2.waitKey(0)
