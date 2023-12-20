from src.yan_yan_et_al import *
from plotting import *
from majority import majority

np.random.seed(1234)

a_real = np.ones(10)
x, y = generate_data(100, a_real, n=4, minc=-1, maxc=1)
x = x[:, 0:-1]
v_real = np.array(
    [
        [3, -1, 0, 0, 0],
        [1, 1, 0, 1, -1],
        [2, 2, 1, 0, -1],
        [-2, -2, 0, 1, 0],
    ]
)
advice = expert_advice(y, x, v_real)

x_1 = np.column_stack((x, np.ones(x.shape[0])))

a, v, l = yan_yan_et_al(x, advice, 1e-3, 0)
w_majority = majority(x, advice, 0)

print(w_majority / w_majority[0])
print(a / a[0])

x, y = generate_data(100, a_real, n=4, minc=-1, maxc=1)
yan_yan_votes_logistically = dot_sigmoid(x, -a)
majority_votes = dot_sigmoid(x, -w_majority)
ts = np.linspace(0, 1, 100)
roc_majority = -np.ones((len(ts), 2))
roc_yan = -np.ones((len(ts), 2))
for j, t in enumerate(ts):
    TPR, FPR = eval_classifier(y, majority_votes, t)
    roc_majority[j, :] = (FPR, TPR)
    TPR, FPR = eval_classifier(y, yan_yan_votes_logistically, t)
    roc_yan[j, :] = (FPR, TPR)

print(roc_yan)
print(roc_majority)

plt.plot(roc_majority[:, 0], roc_majority[:, 1], label="Majority")
plt.plot(roc_yan[:, 0], roc_yan[:, 1], label="Yan")
plt.legend()
plt.show()
