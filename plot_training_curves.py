import matplotlib.pyplot as plt
import numpy as np

epochs = list(range(1, 24))
train_loss = [0.3440, 0.1344, 0.1147, 0.0964, 0.0892, 0.0800, 0.0706, 0.0637, 0.0682, 0.0631, 
              0.0617, 0.0547, 0.0565, 0.0554, 0.0510, 0.0507, 0.0486, 0.0507, 0.0400, 0.0436, 
              0.0456, 0.0503, 0.0394]
val_loss = [0.0749, 0.0476, 0.0394, 0.0462, 0.0321, 0.0684, 0.0359, 0.0519, 0.0456, 0.0224, 
            0.0395, 0.0517, 0.0201, 0.0185, 0.0229, 0.0178, 0.0198, 0.0148, 0.0348, 0.0266, 
            0.0274, 0.0353, 0.0181]

train_acc = [0.9057, 0.9645, 0.9709, 0.9757, 0.9785, 0.9801, 0.9829, 0.9848, 0.9840, 0.9856, 
             0.9853, 0.9875, 0.9875, 0.9878, 0.9880, 0.9888, 0.9889, 0.9892, 0.9913, 0.9903, 
             0.9903, 0.9899, 0.9915]
val_acc = [0.9785, 0.9872, 0.9897, 0.9877, 0.9917, 0.9832, 0.9913, 0.9861, 0.9900, 0.9950, 
           0.9904, 0.9902, 0.9950, 0.9949, 0.9944, 0.9964, 0.9945, 0.9962, 0.9929, 0.9952, 
           0.9950, 0.9915, 0.9964]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot Loss
ax1.plot(epochs, train_loss, 'o-', color='#e74c3c', label='Train Loss', linewidth=2, markersize=5)
ax1.plot(epochs, val_loss, 's--', color='#2980b9', label='Val Loss', linewidth=2, markersize=5)
ax1.scatter([18], [0.0148], color='#27ae60', s=120, zorder=5, label='Best Checkpoint (Epoch 18, 0.0148)')
ax1.axvline(x=18, color='#27ae60', linestyle=':', alpha=0.7)
ax1.set_title('Cross-Entropy Loss vs. Epochs', fontsize=13, fontweight='bold', pad=12)
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Loss', fontsize=11)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(frameon=True, facecolor='#f8f9fa')

# Plot Accuracy
ax2.plot(epochs, [a * 100 for a in train_acc], 'o-', color='#e67e22', label='Train Accuracy (%)', linewidth=2, markersize=5)
ax2.plot(epochs, [a * 100 for a in val_acc], 's--', color='#27ae60', label='Val Accuracy (%)', linewidth=2, markersize=5)
ax2.scatter([16], [99.64], color='#8e44ad', s=120, zorder=5, label='Peak Val Acc (Epoch 16, 99.64%)')
ax2.set_title('Classification Accuracy vs. Epochs', fontsize=13, fontweight='bold', pad=12)
ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('Accuracy (%)', fontsize=11)
ax2.set_ylim(88, 101)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(frameon=True, facecolor='#f8f9fa')

plt.tight_layout()
plt.savefig('training_curves.png', dpi=300)
print("Saved training_curves.png successfully!")
