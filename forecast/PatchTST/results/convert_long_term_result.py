import numpy as np
pred = np.load('pred.npy')
print(pred.shape)
true = np.load('true.npy')
print(true.shape)
pred_list = []
true_list = []
for i in range(0, pred.shape[0], 288):
    pred_list.append(pred[i, :, :])
    true_list.append(true[i, :, :])
new_pred = np.vstack(pred_list).T
new_true = np.vstack(true_list).T
print(new_pred.shape, new_true.shape)
np.save('new_pred.npy', new_pred)
np.save('new_true.npy', new_true)
