plt.figure()
plt.scatter(hsphere[:, 0], hsphere[:, 1], c=y, cmap="coolwarm_r")

p_1 = np.hstack((hsphere, np.ones((n_samples, 1))))
plt.figure()
for i in range(n_experts):
    plt.subplot(2, 3, i + 1)
    plt.scatter(
        hsphere[:, 0],
        hsphere[:, 1],
        c=dot_sigmoid(p_1, v_real[i, :]),
        cmap="coolwarm_r",
    )
