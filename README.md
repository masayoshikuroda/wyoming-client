# Wyoming Client

Wyoming プロトコルを使った音声対話クライアントです。マイク入力から音声認識、ウェイクワード検出、LLM への問い合わせ、テキスト読み上げまでを1つの Python クライアントで扱います。

このリポジトリは、以下のような構成を想定しています。

- STT サーバー: Whisper などの音声認識サービス
- WWD サーバー: openWakeWord などのウェイクワード検出サービス
- TTS サーバー: Piper などの音声合成サービス
- LLM サーバー: Wyoming 互換の LLM/Intent サービス

## 機能概要

- マイク入力を継続的に監視し、ウェイクワード検出を待機
- 検出後に音声認識を開始し、無音検出で発話終了を判定
- 認識したテキストを LLM に送信
- LLM 応答を TTS で音声に変換して再生
- 各モジュールを個別に実行して動作確認可能

## 依存関係

- Python 3.12+
- `wyoming`
- `sounddevice`
- `numpy`
- `webrtcvad-wheels`
- `websockets`

## インストール

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

または `uv` を使う場合:

```bash
uv sync
```

## 主要ファイル

- `main.py`: 全体のフロー制御
- `wwd.py`: ウェイクワード検出クライアント
- `stt.py`: 音声認識クライアント
- `llm.py`: LLM への問い合わせクライアント
- `tts.py`: TTS 再生クライアント

## 実行方法

起動時のデフォルト接続先は `localhost` で、各サーバーのポートは次のとおりです。

- TTS: 10200
- STT: 10300
- WWD: 10400
- LLM: 10500

### 全体フローの実行

```bash
python main.py \
  --host localhost \
  --tts-port 10200 \
  --stt-port 10300 \
  --wwd-port 10400 \
  --llm-port 10500 \
  --wwd-timeout 60.0 \
  --stt-silence-duration 3.0
```

### 個別モジュールのテスト

#### TTS

```bash
python tts.py --host localhost --port 10200 --text "こんにちは、私はWyomingクライアントです。"
```

#### STT

```bash
python stt.py --host localhost --port 10300
```

デフォルトではマイク入力を待ち、発話開始後に無音が一定時間続くと認識結果を返します。音声ソースは `WyomingSTTClient.transcribe(source=...)` に `AudioSource` を渡して切り替えられます。標準の `MicrophoneAudioSource` と `WavAudioSource` は `stt.py` で利用できます。

WAV ファイルを送信する場合は `--source wav --wav` を指定します。従来どおり `--wav` のみでも WAV 入力として動作します。非圧縮 PCM 形式かつクライアント設定と同じサンプルレート・チャンネル数・16-bit 幅の WAV に対応しています。

```bash
python stt.py --host localhost --port 10300 --source wav --wav test.wav
```

#### WWD

```bash
python wwd.py --host localhost --port 10400
```

#### LLM

```bash
python llm.py --host localhost --port 10500 --text "こんばんは、月がきれいですね。"
```

## 動作の流れ

1. `main.py` が `WyomingWWDClient` を起動し、ウェイクワード待機状態になる
2. 検知されると `WyomingSTTClient` で発話内容を録音
3. 無音検出で発話終了を判定して STT サーバーへ送信
4. 認識結果を `WyomingLLMClient` が受け取り、LLM に問い合わせ
5. 応答テキストを `WyomingTTSClient` が音声に変換して再生
