from src.yan_yan_et_al import *
from plotting import *

np.random.seed(1234)

a_real = np.array([1, 1, 1, 1, 1, 1, 1])
x, y = generate_data(10000, a_real, n=6, minc=-1, maxc=1)
x = x[:, 0:-1]
v_real = np.array(
    [
        [3, -1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [2, 2, 1, 0, 0, 0, 0],
        [-2, -2, 0, 0, 0, 0, 1],
    ]
)
advice = expert_advice(y, x, v_real)

a, v, l = yan_yan_et_al(x, advice, 1e-3)
print(f"a: {a / a[0]}")
print(f"a real: {a_real / a_real[0]}")
for t in range(v_real.shape[0]):
    print(f"v: {v[t, :]}")
    print(f"v real: {v_real[t, :]}")
plt.plot(l)
plt.show()
