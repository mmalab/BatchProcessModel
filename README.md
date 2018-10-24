# BatchProcessModel
モデルの学習から評価までを一括して実行する

## How To Work
`bp_main.py`を実行することで, 学習から評価までを行う

```
$ python3 bp_main.py
```

- 学習のみ行う場合(training only)  
`config.py`内で`stage='train'`とする
- 評価のみ行う場合(evaluation only)  
`config.py`内で`stage='eval'`とする

## Now Avaliable
- Classification
  - LeNet

### Training
#### Classification
- flow_from_directory

### Evaluate
#### Classification
- modelによるevaluate
- evaluateに対するPrecision, Recall, F-measure
- ROC曲線とAUC
