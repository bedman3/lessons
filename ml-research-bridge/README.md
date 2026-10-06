# ML Research Bridge lab

The examples use generated data only. They teach model comparison and research
hygiene; no score here represents alpha, a backtest, or P&L.

From this directory, install the compact baseline dependencies and run:

```sh
python -m pip install -r requirements.txt
python lab.py --seed 19 --output result.json
python -m unittest -v test_lab.py
```

`lab.py` creates a known regime shift, purges labels at chronological
boundaries, fits its scaler on development rows only, chooses from a small
declared candidate set using validation MAE, then evaluates the frozen choice
on the original future holdout once. Its invalid future-target control is
reported separately and never participates in selection. JSON includes split
boundaries, versions, metrics, and local timing.

Use the [research memo template](research-memo-template.md) to record the
question, target clock, tried variants, results, and limitations. The synthetic
four-chapter project is a learning exercise; it may take roughly 50–90 focused
hours to complete thoughtfully and does not by itself establish professional
ML experience or qualification for roles requiring production model work.

For the optional CPU neural-network exercise, install `requirements-torch.txt`
and run `python torch_lab.py --seed 19`. It reuses the split and train-only
scaler, demonstrates tensors and autograd, and reports local training and
inference timing. Timing depends on the machine and is not a GPU comparison.
