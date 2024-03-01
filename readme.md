# Reducing PAPR of Multitone Signals using PPO
PPOを用いてPAPRが低減できる初期位相を求め、結果の出力を行う。  
学習データ・計算結果のまとめ（テキスト）・計算結果データ（CSV）の出力を行う。  
包絡線電力の波形については出力を行わないため、必要であれば計算した初期位相を確認して [calc-papr](https://github.com/solightail/calc-papr) を利用すること。

## 設定ファイルについて
本プログラムではコマンドライン引数ではなく、設定ファイル `config.toml` を使用して各値の設定を行う。  
リポジトリをクローンしただけでは `config.toml` は存在しない状態であるため、`config_sample.toml` をコピーして利用すること。  
各値の詳細はコメントアウトしてあるので、ここでは省略する。

## 使用方法
*各コマンドは参考程度で、各環境の状態に応じて実行すること。
1. venvで仮想環境を作成
```bash
python -m venv venv
venv\Scripts\activate.bat
```
2. 必須パッケージをインストール
```bash
python -m pip install --upgrade pip wheel
pip install torch torchvision torchaudio gymnasium matplotlib pandas scipy
```
3. 設定ファイルの準備
```bash
copy repapr-ppo\config_sample.toml repapr-ppo\config.toml
```
4. コマンドラインにて実行
```bash
python -m repapr-ppo
```
