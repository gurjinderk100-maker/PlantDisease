import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(4, 6))
ax.axis('off')

# Box function
def add_box(ax, x, y, w, h, text, color):
    rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='black', facecolor=color)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, horizontalalignment='center', verticalalignment='center', fontsize=12, fontweight='bold')

# Draw blocks
add_box(ax, 1, 4.5, 2, 0.8, "Weight Layer", "#a0c4ff")
add_box(ax, 1, 2.5, 2, 0.8, "Weight Layer", "#a0c4ff")
add_box(ax, 1.5, 1.0, 1, 0.8, "ReLU", "#fdffb6")
add_box(ax, 1.5, 3.5, 1, 0.8, "ReLU", "#fdffb6")

# Arrows
ax.annotate("", xy=(2, 4.5), xytext=(2, 5.5), arrowprops=dict(arrowstyle="->", lw=2))
ax.text(2, 5.6, "x", horizontalalignment='center', fontsize=14, fontweight='bold')

ax.annotate("", xy=(2, 3.5), xytext=(2, 4.3), arrowprops=dict(arrowstyle="->", lw=2)) # down to relu
ax.annotate("", xy=(2, 2.5), xytext=(2, 3.3), arrowprops=dict(arrowstyle="->", lw=2)) # down to weight
ax.annotate("", xy=(2, 1.8), xytext=(2, 2.3), arrowprops=dict(arrowstyle="->", lw=2)) # down to relu

# Skip connection
ax.annotate("", xy=(1.8, 2.0), xytext=(2, 5.2), arrowprops=dict(arrowstyle="->", lw=2, connectionstyle="angle,angleA=0,angleB=-90,rad=10"))
ax.text(0.5, 3.5, "Identity\nx", horizontalalignment='center', fontsize=12, fontweight='bold')

# F(x) label
ax.text(3, 3.5, "F(x)", horizontalalignment='center', fontsize=14, fontweight='bold')

# Plus symbol
circle = patches.Circle((2, 2), 0.2, edgecolor='black', facecolor='white', lw=1.5)
ax.add_patch(circle)
ax.text(2, 2, "+", horizontalalignment='center', verticalalignment='center', fontsize=16, fontweight='bold')
ax.annotate("", xy=(2, 1.8), xytext=(2, 2.0), arrowprops=dict(arrowstyle="-", lw=2))

# Final output
ax.annotate("", xy=(2, 0), xytext=(2, 1.0), arrowprops=dict(arrowstyle="->", lw=2))
ax.text(2, -0.2, "F(x) + x", horizontalalignment='center', fontsize=14, fontweight='bold')

plt.xlim(0, 4)
plt.ylim(-0.5, 6)
plt.savefig("relazione_latex/figure/resnet_block.png", dpi=300, bbox_inches='tight')
