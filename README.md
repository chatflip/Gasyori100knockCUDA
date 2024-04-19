# 画像処理100本ノック with CUDA

## Description

This repository is dedicated to implementing [画像処理100本ノック](https://github.com/ryoppippi/Gasyori100knock) in C++ and CUDA C++.The primary goal is personal learning and enhancing skills in image processing techniques using these programming languages.

## Requirements

- NVIDIA GPU
- OS: Windows
- Software
  - Visual Studio 2022
  - CUDA 11.8

## Contribution

We appreciate your interest, but we are not seeking contributions to this repository. It is intended for individual learning and development. Thank you for understanding.

## Link

| No.| Title(JP) | Title(EN) |
| - | -------- | ------- |
| 1 | [チャネル入れ替え](src/Question_01)  | Channel Swapping |
| 2 | グレースケール化 | Grayscale Conversion |
| 3 | 二値化 | Binarization |
| 4 | 大津の二値化 | Otsu's Binarization |
| 5 | HSV変換 | HSV Conversion |
| 6 | 減色処理 | Color Reduction |
| 7 | 平均プーリング | Average Pooling |
| 8 | Maxプーリング | Max Pooling |
| 9 | ガウシアンフィルタ | Gaussian Filter |
| 10 | メディアンフィルタ | Median Filter |
| 11 | 平滑化 | Smoothing |
| 12 | モーションフィルタ | Motion Filter |
| 13 | MAX-MINフィルタ | MAX-MIN Filter |
| 14 | 微分フィルタ | Differential Filter |
| 15 | Sobelフィルタ | Sobel Filter |
| 16 | Prewittフィルタ | Prewitt Filter |
| 17 | Laplacianフィルタ | Laplacian Filter |
| 18 | Embossフィルタ | Emboss Filter |
| 19 | LoGフィルタ | LoG Filter |
| 20 | ヒストグラム表示 | Histogram Display |
| 21 | ヒストグラム正規化 | Histogram Normalization |
| 22 | ヒストグラム操作 | Histogram Operation |
| 23 | ヒストグラム平坦化 | Histogram Flattening |
| 24 | ガンマ補正 | Gamma Correction |
| 25 | 最近傍補間 | Nearest-neighbor Interpolation |
| 26 | Bi-linear補間 | Bi-linear Interpolation |
| 27 | Bi-cubic補間 | Bi-cubic Interpolation |
| 28 | アフィン変換(平行移動) | Affine Transformation (Translation) |
| 29 | アフィン変換(拡大縮小) | Affine Transformation (Scaling) |
| 30 | アフィン変換(回転) | Affine Transformation (Rotation) |
| 31 | アフィン変換(スキュー) | Affine Transformation (Skew) |
| 32 | フーリエ変換 | Fourier Transformation |
| 33 | フーリエ変換 ローパスフィルタ | Fourier Transformation (Low-pass Filter) |
| 34 | フーリエ変換 ハイパスフィルタ | Fourier Transformation (High-pass Filter) |
| 35 | フーリエ変換 バンドパスフィルタ | Fourier Transformation (Band-pass Filter) |
| 36 | JPEG圧縮 (Step.1)離散コサイン変換 | JPEG Compression (Step.1) Discrete Cosine Transformation |
| 37 | PSNR | PSNR |
| 38 | JPEG圧縮 (Step.2)DCT+量子化 | JPEG Compression (Step.2) DCT + Quantization |
| 39 | JPEG圧縮 (Step.3)YCbCr表色系 | JPEG Compression (Step.3) YCbCr Color Space |
| 40 | JPEG圧縮 (Step.4)YCbCr+DCT+量子化 | JPEG Compression (Step.4) YCbCr + DCT + Quantization |
| 41 | Cannyエッジ検出 (Step.1) エッジ強度 | Canny Edge Detection (Step.1) Edge Strength |
| 42 | Cannyエッジ検出 (Step.2) 細線化 | Canny Edge Detection (Step.2) Thinning |
| 43 | Cannyエッジ検出 (Step.3) ヒステリシス閾処理 | Canny Edge Detection (Step.3) Hysteresis Threshold Processing |
| 44 | Hough変換 直線検出 (Step.1) Hough変換 | Hough Transformation Line Detection (Step.1) Hough Transformation |
| 45 | Hough変換 直線検出 (Step.2) NMS | Hough Transformation Line Detection (Step.2) NMS |
| 46 | Hough変換 直線検出 (Step.3) Hough逆変換 | Hough Transformation Line Detection (Step.3) Hough Inverse Transformation |
| 47 | モルフォロジー処理(膨張) | Morphological Processing (Dilation) |
| 48 | モルフォロジー処理(収縮) | Morphological Processing (Erosion) |
| 49 | オープニング処理 | Opening Processing |
| 50 | クロージング処理 | Closing Processing |
| 51 | モルフォロジー勾配 | Morphological Gradient |
| 52 | トップハット変換 | Top-hat Transformation |
| 53 | ブラックハット変換 | Black-hat Transformation |
| 54 | テンプレートマッチング SSD | Template Matching SSD |
| 55 | テンプレートマッチング SAD | Template Matching SAD |
| 56 | テンプレートマッチング NCC | Template Matching NCC |
| 57 | テンプレートマッチング ZNCC | Template Matching ZNCC |
| 58 | ラベリング 4近傍 | Labeling 4-neighbor |
| 59 | ラベリング 8近傍 | Labeling 8-neighbor |
| 60 | アルファブレンド | Alpha Blend |
| 61 | 4-連結数 | 4-Connected Component |
| 62 | 8-連結数 | 8-Connected Component |
| 63 | 細線化 | Thinning |
| 64 | ヒルディッチの細線化 | Hilditch's Thinning |
| 65 | Zhang-Suenの細線化 | Zhang-Suen's Thinning |
| 66 |  HOG (Step.1) 勾配強度・勾配角度 | HOG (Step.1) Gradient Strength and Gradient Angle |
| 67 | HOG (Step.2) 勾配ヒストグラム | HOG (Step.2) Gradient Histogram |
| 68 | HOG (Step.3) ヒストグラム正規化 | HOG (Step.3) Histogram Normalization |
| 69 | HOG (Step.4) 特徴量の描画 | HOG (Step.4) Drawing Features |
| 70 | カラートラッキング | Color Tracking |
| 71 |  マスキング | Masking |
| 72 | マスキング(カラートラッキングとモルフォロジー) | Masking (Color Tracking and Morphology) |
| 73 | 縮小と拡大 | Reduction and Expansion |
| 74 | ピラミッド差分による高周波成分の抽出 | Extraction of High Frequency Components by Pyramid Difference |
| 75 | ガウシアンピラミッド | Gaussian Pyramid |
| 76 | 顕著性マップ | Saliency Map |
| 77 | ガボールフィルタ | Gabor Filter |
| 78 | ガボールフィルタの回転 | Rotation of Gabor Filter |
| 79 | ガボールフィルタによるエッジ抽出 | Edge Extraction by Gabor Filter |
| 80 | ガボールフィルタによる特徴抽出 | Feature Extraction by Gabor Filter |
| 81 | Hessianのコーナー検出 | Hessian Corner Detection |
| 82 | Harrisのコーナー検出 (Step.1) Sobel + Gaussian | Harris Corner Detection (Step.1) Sobel + Gaussian |
| 83 | Harrisのコーナー検出 (Step.2) コーナー検出 | Harris Corner Detection (Step.2) Corner Detection |
| 84 | 簡単な画像認識 (Step.1) 減色化 + ヒストグラム | Simple Image Recognition (Step.1) Color Reduction + Histogram |
| 85 | 簡単な画像認識 (Step.2) クラス判別 | Simple Image Recognition (Step.2) Class Discrimination |
| 86 | 簡単な画像認識 (Step.3) 評価(Accuracy) | Simple Image Recognition (Step.3) Evaluation (Accuracy) |
| 87 | 簡単な画像認識 (Step.4) k-NN | Simple Image Recognition (Step.4) k-NN |
| 88 | K-means (Step.1) 重心作成 | K-means (Step.1) Creating Centroids |
| 89 | K-means (Step.2) クラスタリング | K-means (Step.2) Clustering |
| 90 | K-means (Step.3) 初期ラベルの変更 | K-means (Step.3) Changing Initial Labels |
| 91 | K-meansによる減色処理 (Step.1) 色の距離によるクラス分類 | Color Reduction by K-means (Step.1) Class Classification by Color Distance |
| 92 | K-meansによる減色処理 (Step.2) 減色処理 | Color Reduction by K-means (Step.2) Color Reduction |
| 93 | 機械学習の学習データの用意 (Step.1) IoUの計算 | Preparing Machine Learning Training Data (Step.1) Calculating IoU |
| 94 | 機械学習の学習データの用意 (Step.2) ランダムクラッピング | Preparing Machine Learning Training Data (Step.2) Random Cropping |
| 95 | ニューラルネットワーク (Step.1) ディープラーニングにする | Neural Network (Step.1) Make it Deep Learning |
| 96 | ニューラルネットワーク (Step.2) 学習 | Neural Network (Step.2) Learning |
| 97 | 簡単な物体検出 (Step.1) スライディングウィンドウ + HOG | Simple Object Detection (Step.1) Sliding Window + HOG |
| 98 | 簡単な物体検出 (Step.2) スライディングウィンドウ + NN | Simple Object Detection (Step.2) Sliding Window + NN |
| 99 | 簡単な物体検出 (Step.3) Non-Maximum Suppression | Simple Object Detection (Step.3) Non-Maximum Suppression |
| 100 | 簡単な物体検出 (Step.4) 評価(Accuracy) | Simple Object Detection (Step.4) Evaluation (Accuracy) |

















  










## Citation

```bash
@article{yoyoyo-yoGasyori100knock,
    Author = {yoyoyo-yo},
    Title = {Gasyori100knock},
    Journal = {https://github.com/yoyoyo-yo/Gasyori100knock},
    Year = {2019}
}
```
