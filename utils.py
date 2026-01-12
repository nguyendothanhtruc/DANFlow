import matplotlib
import matplotlib.pyplot as plt
from torchvision import transforms
from skimage import morphology
from skimage.segmentation import mark_boundaries

# Suppress matplotlib warnings
plt.rcParams.update({'figure.max_open_warning': 0})

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def plot_fig(vmax, vmin, original_im, prediction, label, threshold, saved_path):
    """
    Plot and save visualization of anomaly detection results.
    
    Args:
        vmax (float): Maximum value for normalization
        vmin (float): Minimum value for normalization
        original_im: Original input image
        prediction: Predicted anomaly map
        label: Ground truth label
        threshold (float): Threshold for binary segmentation
        saved_path (str): Path to save the figure
    """
    vmax *= 255.
    vmin *= 255.
    img = (original_im * 255).astype("uint8")
    gt = (label * 255).astype("uint8")
    heat_map = (prediction * 255).astype("uint8")

    # Create binary mask
    mask = prediction.copy()
    mask[mask > threshold] = 1
    mask[mask <= threshold] = 0
    kernel = morphology.disk(4)
    mask = morphology.opening(mask, kernel)
    mask = (mask * 255).astype("uint8")

    # Create visualization
    vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
    fig_img, ax_img = plt.subplots(1, 5, figsize=(12, 3))
    fig_img.subplots_adjust(right=0.9)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    
    for ax_i in ax_img:
        ax_i.axes.xaxis.set_visible(False)
        ax_i.axes.yaxis.set_visible(False)

    # Plot subplots
    ax_img[0].imshow(img)
    ax_img[0].title.set_text('Image')

    ax_img[1].imshow(gt[:, :, 0], cmap='gray')
    ax_img[1].title.set_text('GroundTruth')

    ax = ax_img[2].imshow(heat_map, cmap='jet', norm=norm)
    ax_img[2].imshow(img, cmap='gray', interpolation='none')
    ax_img[2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
    ax_img[2].title.set_text('Predicted heat map')

    ax_img[3].imshow(mask, cmap='gray')
    ax_img[3].title.set_text('Predicted mask')

    ax_img[4].imshow(vis_img)
    ax_img[4].title.set_text('Segmentation result')

    # Add colorbar
    left = 0.92
    bottom = 0.15
    width = 0.015
    height = 1 - 2 * bottom
    rect = [left, bottom, width, height]
    cbar_ax = fig_img.add_axes(rect)
    cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
    cb.ax.tick_params(labelsize=8)
    
    font = {
        'family': 'serif',
        'color': 'black',
        'weight': 'normal',
        'size': 8,
    }
    cb.set_label('Anomaly Score', fontdict=font)

    plt.savefig(saved_path, bbox_inches='tight')
    plt.close()

def denormalize(image):
    inv_transform = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
    ])
    return inv_transform(image)