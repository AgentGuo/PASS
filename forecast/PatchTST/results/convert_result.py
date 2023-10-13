import numpy as np
pred = np.load('pred.npy')
print(pred.shape)
true = np.load('true.npy')
print(true.shape)
preds, trues = [], []
for i in range(pred.shape[2]):
    idx = 115
    tmp_pred, tmp_true = np.array([]), np.array([])
    while idx < len(pred):
        tmp_pred = np.concatenate((tmp_pred, pred[idx, :, i]))
        tmp_true = np.concatenate((tmp_true, true[idx, :, i]))
        idx += len(pred[idx, :, i])
    preds.append(tmp_pred)
    trues.append(tmp_true)
print(len(preds), len(trues))
preds = np.stack(preds, axis=0)
trues = np.stack(trues, axis=0)
print('preds.shape = {}, trues.shape = {}'.format(str(preds.shape), str(trues.shape)))
np.save('new_pred.npy', preds)
np.save('new_true.npy', trues)
