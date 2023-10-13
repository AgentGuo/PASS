import numpy as np
pred = np.load('pred.npy')
print(pred.shape)
true = np.load('true.npy')
print(true.shape)
pred_len = 3
svc_len = pred.shape[2]
new_pred = np.vstack((pred[0, :-1, :], pred[:, pred_len-1, :])).T
new_true = np.vstack((true[0, :-1, :], true[:, pred_len-1, :])).T
print(new_pred.shape, new_true.shape)
np.save('new_pred.npy', new_pred)
np.save('new_true.npy', new_true)
