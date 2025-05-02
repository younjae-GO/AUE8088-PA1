import matplotlib.pyplot as plt

raw = [
    ("EfficientNet‑B0",  4.3,   26.0, "EfficientNet"),
    ("EfficientNet‑B1",  6.8,   39.2, "EfficientNet"),
    ("EfficientNet‑B2",  8.0,   39.2, "EfficientNet"),
    ("EfficientNet‑B3", 11.0,   30.4, "EfficientNet"),
    ("EfficientNet‑B4", 17.9,   29.0, "EfficientNet"),
    ("EfficientNet‑B5", 28.8,   27.0, "EfficientNet"),
    ("EfficientNet‑B6", 41.2,   23.9, "EfficientNet"),
    ("EfficientNet‑B7", 67.3,   21.2, "EfficientNet"),

    ("Densenet201",        18.5,   43.3, "Densenet"),
    ("Densenet169",    12.8,   45.0, "Densenet"),
    ("Densenet161",   26.9,   42.9, "Densenet"),
    ("Densenet161",   7.2,   43.3, "Densenet"),

    ("ResNet18",   11.3,   29.2, "Resnet"),
    ("ResNet34",   21.8,   27.3, "Resnet"),
    ("ResNet50",   25.5,   23.6, "Resnet"),
    ("ResNet101",   44.5,   19.2, "Resnet"),
    ("ResNet152",   60.1,   16.5, "Resnet"),


    # 기타 단일 점
    ("MyNetwork",      2.0,   40.8, "Other"),
    ("Mobilenet-v2",      2.5,   20.7, "Other")
]

fig, ax = plt.subplots(figsize=(7, 5))

markers = {"EfficientNet": "s", "Densenet": "o", "Other": "p", "Resnet": "d"}
linestyles = {"EfficientNet": "-", "Densenet": "--", "Other": ":", "Resnet": "-."}

groups = {}
for name, params, acc, family in raw:
    groups.setdefault(family, []).append((params, acc, name))

for family, points in groups.items():
    points.sort(key=lambda t: t[0])
    xs, ys, labels = zip(*points)

    ax.scatter(xs, ys, marker=markers.get(family, "o"), label=family)

    if family in linestyles:
        ax.plot(xs, ys, linestyles[family])

    for x, y, label in points:
        ax.annotate(label, (x, y), textcoords="offset points",
                    xytext=(4, 4), ha="left", fontsize=8)

ax.set_xlabel("Number of Parameters (Millions)")
ax.set_ylabel("ImageNet Top‑1 Accuracy (%)")
ax.set_title("Accuracy vs. Model Size (ImageNet‑1k)")
ax.grid(True, linestyle=":")
ax.legend(frameon=False)
fig.tight_layout()

plt.show()
