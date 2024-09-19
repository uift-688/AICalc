# Addition/Subtraction Neural Network

このプロジェクトは、加算および減算を学習するシンプルなニューラルネットワークの実装です。PythonのAIモジュールを使用せず、`numpy`と`pickle`を利用しています。

## 必要なモジュール

このプロジェクトでは、以下のモジュールが必要です：

- `numpy`
- `pickle`
- `threading`
- `os`

これらのモジュールは、次のコマンドでインストールできます：

```bash
pip install numpy
```

## 技術概要

### 活性化関数

- **Swish**: `swish(x) = x / (1.0 + exp(-x))`  
  Swish関数は、ニューラルネットワークの非線形性を高めるために使用されます。従来のReLU関数よりも滑らかな変化を提供します。

### 損失関数

- **Huber損失**:  
  Huber損失は、回帰問題において、絶対誤差と二乗誤差の組み合わせを用いてロバストな損失を計算します。`delta`パラメータによって絶対誤差と二乗誤差の切り替え点を調整します。

- **Elastic Net損失**:  
  Elastic Net損失は、L1正則化（Lasso）とL2正則化（Ridge）の組み合わせを用いた損失関数です。`alpha`と`l1_ratio`パラメータで正則化の強さと種類を調整します。

### 最適化アルゴリズム

- **Nadam**:  
  Nadam（Nesterov-accelerated Adaptive Moment Estimation）は、Adamの改良版で、Nesterov加速勾配法を組み合わせた最適化アルゴリズムです。モーメンタムとスケールの調整を行い、学習を効率化します。

## 概要

このニューラルネットワークは、加算と減算の演算を学習し、与えられた入力に対して正しい出力を予測することを目的としています。モデルのトレーニングと評価には`numpy`を使用し、モデルの保存と読み込みには`pickle`を使用しています。

## 使用方法

1. [AICalc.py](https://raw.githubusercontent.com/uift-688/AICalc/main/AICalc.py)をダウンロードします。
2. コマンドラインから次のコマンドを実行して、ニューラルネットワークを開始します：

    ```bash
    python AICalc.py
    ```

3. ユーザーからの入力に基づいて予測を行うことができます。

## ファイル構成

- `AICalc.py`: すべての機能が集まったニューラルネットワークプログラム
- `model_params.pkl`: 学習済みモデルのパラメーターが保存されたファイル（`AICalc.py`で生成）

## ライセンス

このプロジェクトには特定のライセンスはありません。自由に利用、変更、配布してください。

## 注意事項

このニューラルネットワークはバージョン1.0であり、まだ開発段階です。特に引き算モデルに関しては誤差が多いことが確認されています。実用として使用するには、ベータ版になるまでお待ちください。

## バージョン履歴

[1.2](https://raw.githubusercontent.com/uift-688/AICalc/main/AICalc-1.2.py)
[1.0](https://raw.githubusercontent.com/uift-688/AICalc/main/AICalc.py)

---

ご質問や提案がある場合は、[Issues](https://github.com/uift-688/AICalc/issues)でご連絡ください。
