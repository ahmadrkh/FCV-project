# Autonomous Vehicle Surroundings Mapping using Image Stitching and Depth Estimation

## Project Overview
This project is a course assignment for the **3D Computer Vision** class at **Sharif University of Technology, Faculty of Computer Engineering**, during the first semester of 1403-1402 (2023-2024). The goal is to create a comprehensive 3D map of a vehicle's surroundings by generating a panoramic image from multiple camera views and estimating depth maps. This system is designed to enhance environmental perception for autonomous driving applications, addressing challenges like blind spots and collision prevention.

### Key Components
- **Image Stitching**: Combines images from rear, left, and right vehicle cameras to form a 360-degree panoramic view.
- **Depth Estimation**: Generates a depth map from the stitched image using deep learning models, enabling better object detection and distance measurement.

The project leverages existing datasets and simulators to simulate real-world driving scenarios. Results are demonstrated through Jupyter notebooks.

## Authors
- Ahmadreza Khanari
- Ramtin Moslemi
- Amirhossein Haji Mohammad Rezaie


## Professor
- Pr. Shohreh Kasaei

## Dependencies
The project is implemented in Python 3.x and relies on standard computer vision and deep learning libraries. Install via pip:
```bash
pip install opencv-python numpy matplotlib scikit-image
pip install torch torchvision  # For depth estimation models
pip install albumentations  # Optional for image augmentations
pip install jupyter  # For running notebooks
```

## For Specific Models
- Monodepth2: Clone from [GitHub](https://github.com/nianticlabs/monodepth2)
- Panodepth: Refer to the paper for implementation details.

**Note**: No additional installations like CUDA are assumed; run on CPU if needed, but GPU is recommended for depth models.

## Installation and Setup

### Clone the Repository
```bash
git clone <your-repo-url>
cd project-directory
```

## Installation and Setup

### Set Up a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt  # If requirements.txt is provided
```

### Download Datasets
- nuScenes: Register and download from the official site (use mini version for testing). [Link](https://www.nuscenes.org/)
- Oxford RobotCar: Download via their toolkit. [Link](https://robotcar-dataset.robots.ox.ac.uk/)
- CARLA: Install CARLA simulator and use the provided GitHub script to generate data. [CARLASim Repo](https://github.com/rnett/CARLASim)

### For CARLA Data Generation
- Follow the repo instructions to set up sensor positions (e.g., front, rear, left, right cameras).
- Run the simulation to output images.

## Usage
The project is structured around Jupyter notebooks for step-by-step execution.

### Running the Notebooks

#### Launch Jupyter
```bash
jupyter notebook
```

#### Open the Main Notebooks
- `image_stitching.ipynb`: Performs stitching on sample frames. Inputs: Images from datasets. Outputs: Panoramic views.
  - Example: Load left/rear/right images, compute homography iteratively, visualize results.
- `depth_estimation.ipynb`: Applies depth models to stitched images.
  - Example: Use Monodepth2 on panoramic output; visualize depth heatmaps.
- `full_pipeline.ipynb`: End-to-end pipeline combining both steps, tested on CARLA-generated data.

### Example Code Snippet (Image Stitching)
```bash
import cv2
import numpy as np

# Load images (e.g., left and rear)
img_left = cv2.imread('left.jpg')
img_rear = cv2.imread('rear.jpg')

# SIFT feature detection and matching
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img_left, None)
kp2, des2 = sift.detectAndCompute(img_rear, None)

# Match features and compute homography
matcher = cv2.BFMatcher()
matches = matcher.knnMatch(des1, des2, k=2)
good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Warp and stitch
stitched = cv2.warpPerspective(img_left, H, (img_left.shape[1] + img_rear.shape[1], img_left.shape[0]))
# ... (add rear image to stitched)
```

#### For Iterative Updates (to Handle Drift)
```bash
# Weighted update for consecutive frames
alpha = 1 / cnt  # cnt is frame counter
H_new = alpha * H_current + (1 - alpha) * H_previous
```

### Testing on Custom Data
- Place your images in the `data/` folder.
- Update notebook paths and run cells sequentially.
- For video input: Process frames iteratively to maintain consistency.

## Results
- **Stitching Example**: Panoramic views eliminate blind spots, showing ~360-degree surroundings.
- **Depth Maps**: Accurate distance estimation (e.g., using LIDAR supervision in nuScenes), visualized as heatmaps.
- Sample outputs are included in the notebooks. For real-time performance, optimize with GPU.

See the project report (`CV_Project_Report.pdf`) for detailed methodology, literature review, and figures.

## Challenges
- **Dataset Size**: Large volumes (e.g., 15+ GB) required selective downloading.
- **Field of View (FoV)**: Limited FoV in real datasets led to fewer keypoint matches; mitigated with iterative estimation.
- **Drift in Sequences**: Addressed via window-based updates and outlier rejection (e.g., if matrix difference > 100).
- **Non-Ideal Conditions**: Handled non-ideal rotations using SIFT; synthetic data from CARLA helped validation.

## Future Work
- Integrate with real-time onboard computers for live processing.
- Enhance with more sensors (e.g., full LIDAR fusion).
- Commercialize for driver assistance systems to reduce accidents.
- Scale to full 360-degree coverage with additional cameras.

## References
- Blind Spot Zone: [ResearchGate Figure](https://www.researchgate.net/figure/The-blindspot-zone-description-We-define-the-blindspot-of-a-driver-as-the-zone-he-can_fig1_221355854)
- Brown & Lowe (2007). Automatic Panoramic Image Stitching Using Invariant Features. IJCV.
- Lin et al. (2015). Adaptive As-Natural-As-Possible Image Stitching. CVPR.
- Ho et al. (2017). 360-Degree Video Stitching for Dual-Fisheye Lens Cameras. ICIP.
- Godard et al. (2019). Digging into Self-Supervised Monocular Depth Estimation. ICCV.
- Johnston & Carneiro (2020). Self-Supervised Monocular Trained Depth Estimation. CVPR.
- Kumar et al. (2018). Monocular Fisheye Camera Depth Estimation Using Sparse LiDAR Supervision. ITSC.
- Kumar et al. (2020). FisheyeDistanceNet: Self-Supervised Scale-Aware Distance Estimation. ICRA.
- Li et al. (2021). PanoDepth: A Two-Stage Approach for Monocular Omnidirectional Depth Estimation. 3DV.
- Maddern et al. (2017). 1 Year, 1000 km: The Oxford RobotCar Dataset. IJRR.
- Caesar et al. (2020). nuScenes: A Multimodal Dataset for Autonomous Driving. CVPR.
- Dosovitskiy et al. (2017). CARLA: An Open Urban Driving Simulator. CoRL.
- Nett (2020). CARLASim. [GitHub](https://github.com/rnett/CARLASim).

<!-- ## License
 This project is for educational purposes. Code is open-source under MIT License. Datasets are subject to their respective licenses. -->

## Last Updated
02:57 AM EDT, October 21, 2025
