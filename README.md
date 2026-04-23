# Parking-spot-detection-and-counter

## Run inference

Project now reuses existing files:

- `main.py`
- `util.py`

`util.py` will auto-load model in this order:

1. `model/best_cnn_model.pth` (PyTorch CNN)
2. `model/model.p` (sklearn SVM fallback)

Run:

```bash
python main.py
```

Press `q` to quit.