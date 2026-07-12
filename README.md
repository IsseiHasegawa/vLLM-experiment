# vLLM-learn

## Methods決定ログ（2026-07-12）
- vLLM v0.25.0 固定（commit 702f4814fe54fabff350d43cb753ae3e47c0c276）、
  fork IsseiHasegawa/vllm の instrumentation ブランチで計装
- 環境: RunPod A40 48GB を全実験で統一（1x / tp=2は2x）。
  ベンチクライアントはサーバと同一Pod上（localhost）
- D1: 全ランで --ignore-eos（モデル間で生成トークン数を統制）
- D2: --num-warmups 10。サーバ起動直後は捨てラン1本
- D3: サーバ起動は（モデル×TP構成）ごと。レート・データセット・反復は同一サーバで実行
- D4: run帰属はmanifest方式（開始/終了時刻でphase JSONLをスライス）
- D5: 主指標p95。p99はn=200のため3反復の誤差棒付き参考値
- D6: 反復は同一seed=42（ワークロード固定、誤差棒は系統ノイズのみを表す）
- D7: 1GPU実験（S1,S2,S3,I1,I2）は同一インスタンスの1セッションに統合
- D8: --no-enable-prefix-caching 必須 / --disable-log-stats 禁止 /
  async scheduling 無効（step計装の帰属を明確にするため）
