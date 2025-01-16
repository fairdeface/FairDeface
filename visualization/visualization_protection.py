#### Visualization during protection
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from fawkes.protection import Fawkes
from fawkes.utils import load_image, load_extractor
from skimage import filters
from scipy import ndimage
from tqdm import tqdm
import multiprocessing

class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            del kwargs['groups']
        super().__init__(*args, **kwargs)

def get_img_array(img_path, size=(112, 112)):
    img = load_image(img_path)
    img = cv2.resize(img, size)
    array = np.expand_dims(img, axis=0) / 255.0
    return array

def generate_saliency_map(model, image):
    image = tf.convert_to_tensor(image)
    with tf.GradientTape() as tape:
        tape.watch(image)
        features = model(image)
        target = tf.reduce_sum(tf.square(features))

    gradients = tape.gradient(target, image)
    saliency_map = tf.reduce_max(tf.abs(gradients), axis=-1)
    return saliency_map.numpy()

def create_saliency_heatmap(saliency_map, target_size):
    saliency_map = cv2.resize(saliency_map, target_size)
    saliency_norm = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
    saliency_smooth = filters.gaussian(saliency_norm, sigma=5)
    saliency_enhanced = (saliency_smooth - saliency_smooth.min()) / (saliency_smooth.max() - saliency_smooth.min())
    saliency_enhanced = ndimage.gaussian_filter(saliency_enhanced, sigma=3)
    heatmap = plt.cm.jet(saliency_enhanced)
    return (heatmap[:,:,:3] * 255).astype(np.uint8)

def visualize_protection_and_saliency(original_img, protected_img, saliency_map, output_path):
    target_size = original_img.shape[:2]
    protected_img = cv2.resize(protected_img, (target_size[1], target_size[0]))
    heatmap = create_saliency_heatmap(saliency_map, (target_size[1], target_size[0]))
    overlay = cv2.addWeighted(protected_img, 0.6, heatmap, 0.4, 0)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original_img)
    axes[0].set_title("Original")
    axes[0].axis('off')

    axes[1].imshow(protected_img)
    axes[1].set_title("Protected")
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title("Obfuscation Focus Areas")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def process_image(image_path, visualization_output_dir, input_dir, gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        protector = Fawkes("extractor_2", gpu=str(gpu_id), batch_size=8, mode="high")
        extractor = load_extractor(protector.feature_extractor)

        original_img = cv2.imread(image_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        original_array = get_img_array(image_path)

        protection_result = protector.run_protection([image_path])

        protected_img_path = image_path.replace('.jpg', '_cloaked.png')
        if not os.path.exists(protected_img_path):
            raise FileNotFoundError(f"Protected image not found at {protected_img_path}")

        protected_img = cv2.imread(protected_img_path)
        protected_img = cv2.cvtColor(protected_img, cv2.COLOR_BGR2RGB)
        protected_array = get_img_array(protected_img_path)

        protected_saliency = generate_saliency_map(extractor.model, protected_array)

        relative_path = os.path.relpath(os.path.dirname(image_path), start=input_dir)
        visualization_subdir = os.path.join(visualization_output_dir,
                                            f"{os.path.basename(os.path.dirname(os.path.dirname(image_path)))}_visualization",
                                            f"{os.path.basename(os.path.dirname(image_path))}_visualization")
        os.makedirs(visualization_subdir, exist_ok=True)

        visualization_filename = os.path.basename(image_path).replace('.jpg', '_visualization.png').replace('_cloaked', '')
        visualization_path = os.path.join(visualization_subdir, visualization_filename)

        visualize_protection_and_saliency(
            original_img,
            protected_img,
            protected_saliency[0],
            visualization_path
        )

        return True
    except Exception as e:
        print(f"Error processing {image_path} on GPU {gpu_id}: {str(e)}")
        return False




def process_directory(directory, visualization_output_dir, input_dir, num_gpus):
    image_files = [os.path.join(directory, f) for f in os.listdir(directory)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    pool = multiprocessing.Pool(processes=num_gpus)
    results = []

    for i, image_path in enumerate(image_files):
        gpu_id = i % num_gpus
        results.append(pool.apply_async(process_image, args=(image_path, visualization_output_dir, input_dir, gpu_id)))
    
    pool.close()
    pool.join()
    
    successful_count = sum(1 for r in results if r.get())
    return successful_count, len(image_files)

def print_directory_structure(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")

def main():
    try:
        input_dir = "/home/poojitha/fawkes-new/fawkes/BFW"
        visualization_output_dir = "BFW_Visualizations"
        num_gpus = 4

        os.makedirs(visualization_output_dir, exist_ok=True)

        print("Input directory structure:")
        print_directory_structure(input_dir)

        total_successful = 0
        total_images = 0

        for root, dirs, files in os.walk(input_dir):
            if files:
                print(f"Processing directory: {root}")
                successful, total = process_directory(root, visualization_output_dir, input_dir, num_gpus)
                total_successful += successful
                total_images += total

        print(f"\nProcessing complete. {total_successful}/{total_images} images processed successfully.")
        print(f"Cloaked images saved in the same directory structure as the original images.")
        print(f"Visualizations saved in: {visualization_output_dir}")

        print("\nVisualization output directory structure:")
        print_directory_structure(visualization_output_dir)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    main()
