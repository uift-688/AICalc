import numpy as np
import threading
import pickle
import os

# Swish活性化関数
def swish(x):
    return x / (1.0 + np.exp(-x))

# スケーリング関数（小数対応）
def scale_input(X):
    return X / 10.0

def unscale_output(output):
    return output * 10.0

# ハッバー損失関数
def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    abs_error = np.abs(error)
    quadratic = np.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return 0.5 * np.mean(np.square(quadratic)) + delta * np.mean(linear)

# Elastic Net損失関数
def elastic_net_loss(y_true, y_pred, weights, alpha=0.5, l1_ratio=0.5):
    error = y_true - y_pred
    mse_loss = np.mean(np.square(error))
    
    l1_loss = l1_ratio * np.sum(np.abs(weights))
    l2_loss = (1 - l1_ratio) * np.sum(np.square(weights))
    
    return mse_loss + alpha * (l1_loss + l2_loss)

# 学習関数（Elastic Net 搭載）
def fine_tune_with_elastic_net(X, y, weights, bias, batch_size=32, learning_rate=0.0001, epochs=1000, 
                            beta1=0.9, beta2=0.999, epsilon=1e-8, alpha=3.0, l1_ratio=0.5):
    num_samples = X.shape[0]

    # Nadamのパラメータの初期化
    m_w = np.zeros_like(weights)
    v_w = np.zeros_like(weights)
    m_b = np.zeros_like(bias)
    v_b = np.zeros_like(bias)
    t = 0

    for epoch in range(epochs):
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        epoch_loss = 0
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            # フォワードプロパゲーション
            output = np.dot(X_batch, weights) + bias
            error = y_batch - output

            # Elastic Net 損失の計算
            batch_loss = elastic_net_loss(y_batch, output, weights, alpha, l1_ratio)
            epoch_loss += batch_loss * len(batch_indices)

            # 勾配の計算
            grad_w = -np.dot(X_batch.T, error) / batch_size
            grad_b = -np.sum(error, axis=0, keepdims=True) / batch_size

            # Nadamの更新
            t += 1
            m_w = beta1 * m_w + (1 - beta1) * grad_w
            v_w = beta2 * v_w + (1 - beta2) * np.square(grad_w)
            m_b = beta1 * m_b + (1 - beta1) * grad_b
            v_b = beta2 * v_b + (1 - beta2) * np.square(grad_b)

            m_w_hat = m_w / (1 - beta1 ** t)
            v_w_hat = v_w / (1 - beta2 ** t)
            m_b_hat = m_b / (1 - beta1 ** t)
            v_b_hat = v_b / (1 - beta2 ** t)

            weights -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
            bias -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

            # NaNやinfのチェック
            if np.any(np.isnan(weights)) or np.any(np.isinf(weights)) or \
               np.any(np.isnan(bias)) or np.any(np.isinf(bias)) or \
               np.any(np.isnan(error)) or np.any(np.isinf(error)):
                print("NaN/inf発生時の weights: ", weights)
                print("NaN/inf発生時の bias: ", bias)
                print("NaN/inf発生時の error: ", error)
                raise ValueError("学習中にNaNまたはinfが発生しました。")

    return weights, bias

# パラメーターを保存する関数
def save_parameters(weights_dict, bias_dict, filename="./model_params.pkl"):
    with open(filename, "wb") as f:
        pickle.dump({"weights": weights_dict, "bias": bias_dict}, f)

# パラメーターを読み込む関数
def load_parameters(filename="./model_params.pkl"):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    else:
        return None, None

# モデルの初期化
np.random.seed(1)
weights_dict, bias_dict = load_parameters()
if weights_dict is None or bias_dict is None:
    weights_dict = {"+": np.random.rand(2, 1), "-": np.random.rand(2, 1)}
    bias_dict = {"+": np.random.rand(1, 1), "-": np.random.rand(1, 1)}

# 学習スレッドの関数
def learning_thread(operation, X, y):
    global weights_dict, bias_dict

    # 正しい型が設定されていることを確認
    if not isinstance(weights_dict, dict) or not isinstance(bias_dict, dict):
        print("weights_dict または bias_dict が辞書ではありません。")
        return
    
    try:
        weights = weights_dict[operation]
        bias = bias_dict[operation]
    except KeyError as e:
        print(f"KeyError: {e} - operation '{operation}' が辞書に存在しません。")
        return
    except TypeError as e:
        print(f"TypeError: {e} - weights_dict または bias_dict の型に問題があります。")
        return

    try:
        if operation == "+":
            fine_tune_with_elastic_net(X, y, weights, bias, epochs=50000)
        elif operation == "-":
            fine_tune_with_elastic_net(X, y, weights, bias, epochs=2000)
        save_parameters(weights_dict, bias_dict)
    except ValueError as e:
        print(f"エラー発生: {e}")
    print("学習が完了しました: " + operation)

# 学習スレッドの開始
def start_learning_threads(X, y_plus, y_minus):
    thread_plus = threading.Thread(target=learning_thread, args=("+", X, y_plus), daemon=True)
    thread_minus = threading.Thread(target=learning_thread, args=("-", X, y_minus), daemon=True)
    thread_plus.start()
    thread_minus.start()
    print("学習が開始されました。学習は非同期で実行されます。")

# 予測関数
def predict(X, weights, bias):
    return swish(np.dot(X, weights) + bias)

# パラメーターを使用して予測する関数（小数対応）
def predict_with_trained_model(test_input, operation):
    scaled_input = scale_input(test_input)
    scaled_output = predict(scaled_input, weights_dict[operation], bias_dict[operation])
    output = unscale_output(scaled_output)
    return int(round(output[0][0]))

# ユーザーからの入力を受け取る関数
def get_user_input():
    print("2つの数をスペースで区切って入力し、演算子を選択してください (例: 500 200 +)。終了するには「exit」と入力してください。")
    user_input = input()
    if user_input.lower() == "exit":
        return None, None, None
    try:
        numbers = list(map(float, user_input.split()[:2]))
        operation = user_input.split()[2]
        if operation not in ["+", "-"]:
            print("無効な演算子です。'+' または '-' を入力してください。")
            return None, None, None
        if len(numbers) != 2:
            print("無効な入力です。数値を2つ入力してください。")
            return None, None, None
        return np.array([numbers]), operation
    except (ValueError, IndexError):
        print("無効な入力です。数値と演算子を正しく入力してください。")
        return None, None, None

# メイン処理
if __name__ == "__main__":
    # デモ用のデータ生成
    X, y_plus, y_minus = np.array([[1, 2], [3, 5], [4, 7], [2, 3]]), np.array([[3], [8], [11], [5]]), np.array([[0], [1], [3], [1]])
    
    start_learning_threads(X, y_plus, y_minus)
    
    while True:
        test_input, operation = get_user_input()
        if test_input is None:
            break
        result = predict_with_trained_model(test_input, operation)
        print(f"予測結果: {result}")
