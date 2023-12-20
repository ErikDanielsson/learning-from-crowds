from src.yan_yan_et_al import *
from plotting import *

np.random.seed(1234)

a_real = np.array([1, -2, 0])
x, y = generate_data(1000, a_real)
x = x[:, 0:-1]
v_real = 3 * np.array(
    [
        [3, -1, 0],
        [1, 1, -1],
        [2, 2, -1],
        [-2, -2, 1],
    ]
)
advice = expert_advice(y, x, v_real)

x_1 = np.column_stack((x, np.ones(x.shape[0])))

plt.figure()
plot_bin_datapoints(x, y, plt.gca(), marker="o", markersize=20)
plot_line(a_real, plt.gca(), color="green")
plt.gcf().suptitle("Ground truth", fontsize=16)
plt.gca().set_xlabel("Feature 1")
plt.gca().set_ylabel("Feature 2")
plt.savefig("yan-yan-et-al-ground-truth.png")

annotator_inds = [(0, 0), (1, 0), (0, 1), (1, 1)]

fig, axs = plt.subplots(2, 2)
for i, ind in enumerate(annotator_inds):
    plot_bin_datapoints(x, advice[:, i], axs[*ind], marker="o")
    plot_line(v_real[i, :], axs[*ind], color="green")
    axs[*ind].set_title(f"Expert {i + 1}")
    axs[*ind].set_xlabel("Feature 1")
    axs[*ind].set_ylabel("Feature 2")
fig.suptitle("Expert predictions", fontsize=16)
fig.tight_layout()
plt.savefig("yan-yan-et-al-expert-predictions.png")


fig, axs = plt.subplots(2, 2)
for i, ind in enumerate(annotator_inds):
    conf = create_color_classifier(x_1, v_real[i, :])
    plot_cont_datapoints(x, conf, axs[*ind], marker="o")
    axs[*ind].set_title(f"Expert {i + 1}")
    axs[*ind].set_xlabel("Feature 1")
    axs[*ind].set_ylabel("Feature 2")
fig.suptitle("Expert bias", fontsize=16)
fig.tight_layout()
plt.savefig("yan-yan-et-al-expert-bias.png")


a, v = yan_yan_et_al(x, advice, 1e-3, 0)
print(a, v)

plt.figure()
x_1 = np.column_stack((x, np.ones(x.shape[0])))
pred_confidence = create_color_classifier(x_1, a, threshold=0.5)
plot_bin_datapoints(x, pred_confidence, plt.gca(), markersize=20)
plot_line(a_real, plt.gca(), color="green")
plot_line(a, plt.gca(), color="red")

plt.gcf().suptitle("Estimated ground truth", fontsize=16)
plt.gca().set_xlabel("Feature 1")
plt.gca().set_ylabel("Feature 2")
plt.savefig("yan-yan-et-al-estimated-ground-truth.png")


fig, axs = plt.subplots(2, 2)
for i, ind in enumerate(annotator_inds):
    conf = create_color_annotator(x_1, pred_confidence, v[i, :], 0.5)
    plot_bin_datapoints(x, conf, axs[*ind])
    plot_line(v_real[i, :], axs[*ind], color="green")
    plot_line(v[i, :], axs[*ind], color="red")
    axs[*ind].set_title(f"Expert {i + 1}")
    axs[*ind].set_xlabel("Feature 1")
    axs[*ind].set_ylabel("Feature 2")
fig.suptitle("Estimated expert predictions", fontsize=16)
fig.tight_layout()
plt.savefig("yan-yan-et-al-estimated-expert-predictions.png")

fig, axs = plt.subplots(2, 2)
for i, ind in enumerate(annotator_inds):
    conf = create_color_classifier(x_1, v[i, :])
    plot_cont_datapoints(x, conf, axs[*ind])
    axs[*ind].set_title(f"Expert {i + 1}")
    axs[*ind].set_xlabel("Feature 1")
    axs[*ind].set_ylabel("Feature 2")
fig.suptitle("Estimated expert bias", fontsize=16)
fig.tight_layout()
plt.savefig("yan-yan-et-al-estimated-expert-bias.png")
