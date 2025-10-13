import numpy as np
import cv2
import pandas as pd
from typing import Dict, Optional


def calculate_matching_metrics(xlsx_path: str, scene_list: list = None) -> Optional[Dict[str, Dict]]:
    """
    Calculate performance metrics for feature matching algorithms.

    Args:
        xlsx_path: Path to the Excel result file
        scene_list: List of scene types to analyze. If None, uses default list

    Returns:
        metrics_dict: Dictionary containing performance metrics for each scene.
                     Returns None if file doesn't exist or error occurs.
    """
    # Default scene list if not provided
    if scene_list is None:
        scene_list = ['urban', 'plain', 'rural', 'water', 'mountain', 'hills', 'desert', 'frozen']

    try:
        # Read Excel file
        df = pd.read_excel(xlsx_path)

        # Initialize result dictionary
        metrics_dict = {}

        # Iterate through all scene types
        for scene in scene_list:
            # Filter rows where first column contains the scene name
            filtered_df = df[df.iloc[:, 0].str.contains(scene, na=False)]

            # Get the 6th column (error column) data
            error_column = filtered_df.iloc[:, 5]

            # Calculate error distribution statistics
            total_count = len(error_column)
            if total_count > 0:
                # Count matches within different error thresholds
                count_3 = (error_column <= 3).sum()  # Number of samples with error ≤ 3 pixels
                count_5 = (error_column <= 5).sum()  # Number of samples with error ≤ 5 pixels
                count_10 = (error_column <= 10).sum()  # Number of samples with error ≤ 10 pixels

                # Calculate success rates
                ratio_3 = count_3 / total_count  # Success rate for ≤ 3 pixels
                ratio_5 = count_5 / total_count  # Success rate for ≤ 5 pixels
                ratio_10 = count_10 / total_count  # Success rate for ≤ 10 pixels

                # Store metrics for this scene
                metrics_dict[scene] = {
                    'total_samples': total_count,
                    'success_rate_3px': ratio_3 * 100,
                    'success_rate_5px': ratio_5 * 100,
                    'success_rate_10px': ratio_10 * 100,
                    'count_3px': count_3,
                    'count_5px': count_5,
                    'count_10px': count_10
                }
            else:
                # If no data for this scene
                metrics_dict[scene] = {
                    'total_samples': 0,
                    'success_rate_3px': 0.0,
                    'success_rate_5px': 0.0,
                    'success_rate_10px': 0.0,
                    'count_3px': 0,
                    'count_5px': 0,
                    'count_10px': 0
                }

        return metrics_dict

    except FileNotFoundError:
        print(f"Error: File {xlsx_path} does not exist")
        return None
    except Exception as e:
        print(f"Error processing file {xlsx_path}: {str(e)}")
        return None


def print_metrics_latex_format(metrics_dict: Dict[str, Dict], satellite: str, method: str):
    """
    Print performance metrics in LaTeX table format.

    Args:
        metrics_dict: Dictionary containing performance metrics
        satellite: Satellite name for labeling
        method: Matching method name for labeling
    """
    if metrics_dict is None:
        return

    for scene, metrics in metrics_dict.items():
        if metrics['total_samples'] > 0:
            # Print in LaTeX table row format
            print(f"{satellite}_{scene} & {method} & {metrics['success_rate_3px']:.2f} & "
                  f"{metrics['success_rate_5px']:.2f} & {metrics['success_rate_10px']:.2f} \\\\")
            print("-" * 50)

def rotate_and_translate_image_new(image, angle, tx, ty):
    """
    Apply rotation and translation transformation to an image.

    Args:
        image: Input image (numpy array)
        angle: Rotation angle in degrees
        tx: Translation in x-direction (pixels)
        ty: Translation in y-direction (pixels)

    Returns:
        img_b_transformed: Transformed image
        T: Combined transformation matrix (3x3)
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Create rotation matrix (2x3)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Create translation matrix (2x3)
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])

    # Combine transformations (rotation then translation)
    T = np.dot(np.vstack([rotation_matrix, [0, 0, 1]]),
               np.vstack([translation_matrix, [0, 0, 1]]))

    # Apply the combined transformation
    img_b_transformed = cv2.warpAffine(image, T[:2, :], (w, h))

    return img_b_transformed, T


def corner_error(img_size, H_pred, H_true):
    """
    Calculate the mean corner error between predicted and true homographies.

    Args:
        img_size: Size of the square image (width/height)
        H_pred: Predicted homography matrix (3x3)
        H_true: Ground truth homography matrix (3x3)

    Returns:
        mean_error: Average Euclidean distance between transformed corners
    """
    center = [0, 0]  # Using image origin as center

    # Extract rotation and translation components
    rotation_pred = H_pred[:2, :2]
    translate_pred = H_pred[:2, 2].T
    rotation_true = H_true[:2, :2]
    translate_true = H_true[:2, 2].T

    # Define four image corners
    old_pts_all = np.array([
        [0, 0],
        [0, img_size],
        [img_size, 0],
        [img_size, img_size]
    ])

    error = 0
    for old_pts in old_pts_all:
        # Transform points using predicted homography
        npt_pred = np.matmul((old_pts - center), rotation_pred) + center + translate_pred
        # Transform points using true homography
        npt_true = np.matmul((old_pts - center), rotation_true) + center + translate_true
        # Accumulate Euclidean distance between transformed points
        error += np.linalg.norm(npt_pred - npt_true)

    # Calculate mean error across all four corners
    mean_error = error / 4
    return mean_error


def rigid_transform_2D(A, B):
    """
    Compute optimal rigid transformation between two 2D point sets.

    Args:
        A: Source points (Nx2)
        B: Target points (Nx2)

    Returns:
        H: Homography matrix (3x3) representing the transformation
        rmse: Root mean square error after transformation
    """
    assert A.shape[0] == B.shape[0], 'Point sets must have same number of points'

    N = A.shape[0]  # Number of points
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # Center the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # Compute rotation matrix using SVD
    S = np.matmul(np.transpose(AA), BB)
    U, _, V = np.linalg.svd(S)
    R = np.matmul(V, U.T)

    # Handle reflection case
    if np.linalg.det(R) < 0:
        V[1, :] *= -1
        R = np.matmul(V, U.T)

    # Compute translation vector
    t = -np.matmul(R, centroid_A) + centroid_B

    # Construct homography matrix
    H = np.vstack((np.hstack((R, t.reshape(-1, 1))), np.array([0, 0, 1])))

    # Calculate RMSE
    rmse = B - np.matmul(A, R.T) - t

    return H, rmse


def FSC(cor1, cor2, error_t):
    """
    Fast Sample Consensus (FSC) for 2D point matching with RANSAC.

    Args:
        cor1: Source points (Nx2)
        cor2: Target points (Nx2)
        error_t: Inlier threshold (pixels)

    Returns:
        cor1_new: Inlier points from source (as KeyPoint objects)
        cor2_new: Inlier points from target (as KeyPoint objects)
        H: Best homography matrix (3x3)
    """
    M, N = cor1.shape
    n = 2  # Minimum points needed for estimation
    max_iteration = M * (M - 1)

    # Limit iterations for efficiency
    iterations = min(10000, max_iteration)

    most_consensus_number = 0
    cor1_new = np.zeros((M, N))
    cor2_new = np.zeros((M, N))

    # RANSAC loop
    for i in range(1, iterations + 1):
        count = 0
        while True:
            # Randomly select sample points
            a = np.random.randint(1, M, size=n)
            cor11 = cor1[a, :2]
            cor22 = cor2[a, :2]

            # Ensure valid sample (distinct points)
            if (n == 2 and (a[0] != a[1]) and
                    not np.array_equal(cor11[0, :2], cor11[1, :2]) and
                    not np.array_equal(cor22[0, :2], cor22[1, :2])):
                break

            count += 1
            if M <= 4 or count >= 100:
                break

        # Estimate transformation
        solution, _ = rigid_transform_2D(cor11, cor22)

        # Transform all source points
        match1_xy = cor1[:, 0:2].T
        match1_xy = np.vstack((match1_xy, np.ones((1, match1_xy.shape[1]))))
        t_match1_xy = np.dot(solution, match1_xy)

        # Calculate distances to target points
        match2_xy = cor2[:, 0:2].T
        match2_xy = np.vstack((match2_xy, np.ones((1, match2_xy.shape[1]))))
        diff_match2_xy = np.linalg.norm(t_match1_xy - match2_xy, axis=0)

        # Find inliers
        index_in = np.where(diff_match2_xy < error_t)[0]
        consensus_num = len(index_in)

        # Update best solution if better consensus found
        if consensus_num > most_consensus_number:
            most_consensus_number = consensus_num
            cor1_new = cor1[index_in, :]
            cor2_new = cor2[index_in, :]

    # Remove duplicate points
    uni1 = cor1_new[:, :2]
    _, i = np.unique(uni1, axis=0, return_index=True)
    cor1_new = cor1_new[np.sort(i), :]
    cor2_new = cor2_new[np.sort(i), :]

    uni1 = cor2_new[:, :2]
    _, i = np.unique(uni1, axis=0, return_index=True)
    cor1_new = cor1_new[np.sort(i), :]
    cor2_new = cor2_new[np.sort(i), :]

    # Compute final homography using all inliers
    H, _ = rigid_transform_2D(cor1_new[:, :2], cor2_new[:, :2])

    # Convert to OpenCV KeyPoint objects
    cor1_new = [cv2.KeyPoint(point[0], point[1], 1) for point in cor1_new]
    cor2_new = [cv2.KeyPoint(point[0], point[1], 1) for point in cor2_new]

    return cor1_new, cor2_new, H