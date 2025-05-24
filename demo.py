
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from typing import List, Tuple, Optional
import os


class BRIEFDescriptor:
    def __init__(self, descriptor_length: int = 256, patch_size: int = 31):
        self.descriptor_length = descriptor_length
        self.patch_size = patch_size
        
        # Generate random test pairs
        np.random.seed(42)
        center = patch_size // 2
        sigma = patch_size / 5.0
        
        self.test_pairs = []
        for _ in range(descriptor_length):
            x1 = int(np.clip(np.random.normal(center, sigma), 0, patch_size - 1))
            y1 = int(np.clip(np.random.normal(center, sigma), 0, patch_size - 1))
            x2 = int(np.clip(np.random.normal(center, sigma), 0, patch_size - 1))
            y2 = int(np.clip(np.random.normal(center, sigma), 0, patch_size - 1))
            self.test_pairs.append((x1, y1, x2, y2))
    
    def compute_descriptor(self, image: np.ndarray, keypoint: Tuple[int, int]) -> Optional[np.ndarray]:
        x, y = keypoint
        half_patch = self.patch_size // 2
        
        # Check bounds
        if (x - half_patch < 0 or x + half_patch >= image.shape[1] or
            y - half_patch < 0 or y + half_patch >= image.shape[0]):
            return None
        
        # Extract patch
        patch = image[y - half_patch:y + half_patch + 1,
                     x - half_patch:x + half_patch + 1]
        
        # Gaussian smoothing
        patch = cv2.GaussianBlur(patch, (5, 5), 1.0)
        
        # Binary tests
        descriptor = np.zeros(self.descriptor_length, dtype=np.uint8)
        for i, (x1, y1, x2, y2) in enumerate(self.test_pairs):
            descriptor[i] = 1 if patch[y1, x1] < patch[y2, x2] else 0
        
        return descriptor
    
    def compute_descriptors(self, image: np.ndarray, keypoints: List[Tuple[int, int]]) -> List[np.ndarray]:
        """Compute descriptors for multiple keypoints"""
        descriptors = []
        for kp in keypoints:
            desc = self.compute_descriptor(image, kp)
            if desc is not None:
                descriptors.append(desc)
        return descriptors
    
    def match_descriptors(self, desc1: List[np.ndarray], desc2: List[np.ndarray], 
                         threshold: int = 64) -> List[Tuple[int, int]]:
        matches = []
        for i, d1 in enumerate(desc1):
            best_match = -1
            best_distance = float('inf')
            
            for j, d2 in enumerate(desc2):
                distance = np.sum(d1 != d2)  # Hamming distance
                if distance < best_distance and distance <= threshold:
                    best_distance = distance
                    best_match = j
            
            if best_match != -1:
                matches.append((i, best_match))
        
        return matches


def detect_keypoints(image: np.ndarray, max_keypoints: int = 200) -> List[Tuple[int, int]]:
    """Detect keypoints using FAST"""
    fast = cv2.FastFeatureDetector_create(threshold=20)
    kp = fast.detect(image, None)
    return [(int(k.pt[0]), int(k.pt[1])) for k in kp[:max_keypoints]]


def create_test_images():
    """Create synthetic test images if real ones don't exist"""
    print("Creating synthetic test images...")
    
    # Base image
    img1 = np.zeros((400, 600), dtype=np.uint8)
    cv2.rectangle(img1, (50, 50), (150, 120), 255, -1)
    cv2.circle(img1, (300, 100), 40, 200, -1)
    cv2.putText(img1, 'BRIEF', (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)
    
    # Add noise
    noise = np.random.randint(0, 50, img1.shape, dtype=np.uint8)
    img1 = cv2.add(img1, noise)
    
    # Transform for second image
    M = cv2.getRotationMatrix2D((300, 200), 10, 1.0)
    M[0, 2] += 100  # Translation
    img2 = cv2.warpAffine(img1, M, (600, 400))
    
    cv2.imwrite('image1.jpg', img1)
    cv2.imwrite('image2.jpg', img2)
    return img1, img2


def stitch_images(img1: np.ndarray, img2: np.ndarray, 
                 kp1: List[Tuple[int, int]], kp2: List[Tuple[int, int]], 
                 matches: List[Tuple[int, int]]) -> Optional[np.ndarray]:
    """Stitch two images using matched keypoints"""
    if len(matches) < 10:
        print(f"Not enough matches for stitching: {len(matches)}")
        return None
    
    # Extract matched points
    pts1 = np.array([kp1[m[0]] for m in matches], dtype=np.float32)
    pts2 = np.array([kp2[m[1]] for m in matches], dtype=np.float32)
    
    # Find homography
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
    if H is None:
        print("Failed to find homography")
        return None
    
    inliers = np.sum(mask)
    print(f"Homography found with {inliers}/{len(matches)} inliers")
    
    # Warp and stitch
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    
    # Calculate output size
    corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners, H)
    
    all_corners = np.concatenate((warped_corners, 
                                 np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)))
    
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel())
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel())
    
    # Translation matrix
    translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    
    # Warp image1
    output_size = (x_max - x_min, y_max - y_min)
    warped_img1 = cv2.warpPerspective(img1, translation @ H, output_size)
    
    # Create result
    result = np.zeros((y_max - y_min, x_max - x_min), dtype=np.uint8)
    
    # Place image2
    y_offset, x_offset = -y_min, -x_min
    result[y_offset:y_offset + h2, x_offset:x_offset + w2] = img2
    
    # Blend with warped image1
    mask1 = warped_img1 > 0
    mask2 = result > 0
    overlap = mask1 & mask2
    
    result[mask1 & ~overlap] = warped_img1[mask1 & ~overlap]
    result[overlap] = (warped_img1[overlap].astype(float) + result[overlap].astype(float)) / 2
    
    return result.astype(np.uint8)


def draw_matches(img1: np.ndarray, kp1: List[Tuple[int, int]], 
                img2: np.ndarray, kp2: List[Tuple[int, int]], 
                matches: List[Tuple[int, int]], max_show: int = 20) -> np.ndarray:
    """Draw matches between two images"""
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    
    # Create side-by-side image
    combined = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    combined[:h1, :w1] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    combined[:h2, w1:] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    
    # Draw matches
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    
    for i, (idx1, idx2) in enumerate(matches[:max_show]):
        color = colors[i % len(colors)]
        pt1 = kp1[idx1]
        pt2 = (kp2[idx2][0] + w1, kp2[idx2][1])
        
        cv2.circle(combined, pt1, 4, color, -1)
        cv2.circle(combined, pt2, 4, color, -1)
        cv2.line(combined, pt1, pt2, color, 2)
    
    return combined


def run_demo():
    """Main demo function"""
    print("=" * 60)
    print("BRIEF Descriptor Demo with Image Stitching")
    print("By: lilfermat | Date: 2025-05-24 09:40:55")
    print("=" * 60)
    
    # Load images
    if os.path.exists('image1.jpg') and os.path.exists('image2.jpg'):
        img1 = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)
        print("Loaded image1.jpg and image2.jpg")
    else:
        img1, img2 = create_test_images()
    
    print(f"Image shapes: {img1.shape}, {img2.shape}")
    
    # Initialize BRIEF
    brief = BRIEFDescriptor(descriptor_length=256)
    
    # Detect keypoints
    print("\n1. Detecting keypoints...")
    kp1 = detect_keypoints(img1)
    kp2 = detect_keypoints(img2)
    print(f"   Found {len(kp1)} and {len(kp2)} keypoints")
    
    # Compute descriptors
    print("2. Computing BRIEF descriptors...")
    desc1 = brief.compute_descriptors(img1, kp1)
    desc2 = brief.compute_descriptors(img2, kp2)
    print(f"   Computed {len(desc1)} and {len(desc2)} descriptors")
    
    # Match descriptors
    print("3. Matching descriptors...")
    matches = brief.match_descriptors(desc1, desc2, threshold=64)
    print(f"   Found {len(matches)} matches")
    
    # Calculate match statistics
    if matches:
        distances = [np.sum(desc1[m[0]] != desc2[m[1]]) for m in matches]
        print(f"   Average Hamming distance: {np.mean(distances):.1f}")
    
    # Stitch images
    print("\n4. Stitching images...")
    stitched = stitch_images(img1, img2, kp1, kp2, matches)
    
    # Visualization
    print("\n5. Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Original images
    axes[0, 0].imshow(img1, cmap='gray')
    axes[0, 0].set_title(f'Image 1\n{len(kp1)} keypoints')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img2, cmap='gray')
    axes[0, 1].set_title(f'Image 2\n{len(kp2)} keypoints')
    axes[0, 1].axis('off')
    
    # Matches
    matches_img = draw_matches(img1, kp1, img2, kp2, matches)
    axes[1, 0].imshow(cv2.cvtColor(matches_img, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f'Feature Matches\n{len(matches)} total')
    axes[1, 0].axis('off')
    
    # Stitched result
    if stitched is not None:
        axes[1, 1].imshow(stitched, cmap='gray')
        axes[1, 1].set_title(f'Stitched Panorama\n{stitched.shape[1]}x{stitched.shape[0]}')
        cv2.imwrite('panorama.jpg', stitched)
        print("Saved: panorama.jpg")
    else:
        axes[1, 1].text(0.5, 0.5, 'Stitching Failed\nNot enough matches', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Stitching Failed')
    axes[1, 1].axis('off')
    
    plt.suptitle('BRIEF Descriptor Demo Results', fontsize=16)
    plt.tight_layout()
    plt.savefig('brief_results.png', dpi=150, bbox_inches='tight')
    print("Saved: brief_results.png")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Keypoints detected: {len(kp1)}, {len(kp2)}")
    print(f"Descriptors computed: {len(desc1)}, {len(desc2)}")
    print(f"Matches found: {len(matches)}")
    print(f"Stitching: {'SUCCESS' if stitched is not None else 'FAILED'}")
    
    if stitched is not None:
        print(f"Panorama size: {stitched.shape[1]} x {stitched.shape[0]}")
    
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    
    try:
        run_demo()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()