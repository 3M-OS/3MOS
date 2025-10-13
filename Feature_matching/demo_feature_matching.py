import argparse
import torch
from tqdm import tqdm
from utils import *
from torchvision import transforms
import xlwt
from src.utils.load_model import load_model
import os

if __name__ == '__main__':
    def add_common_arguments(parser):
        """Add common arguments for all matching methods"""
        parser.add_argument('--satellite_list', type=str, default='ALOS,GF3,SEN,RCM,Radarsat',
                            help='Comma-separated list of satellite types')
        parser.add_argument('--data_base_path', type=str,
                            default=r'E:\temp\ubuntu\crop_data',
                            help='Base directory containing image data')
        parser.add_argument('--test_data_file', type=str,
                            default=r'test_feature_matching.txt',
                            help='Path to test data file containing image pairs and transformations')


    def add_method_arguments(parser, method):
        """Add method-specific arguments"""
        if method == "xoftr":
            parser.add_argument('--match_threshold', type=float, default=0.3,
                                help='Initial matching threshold for XOFTR')
            parser.add_argument('--fine_threshold', type=float, default=0.1,
                                help='Fine matching threshold for XOFTR')
            parser.add_argument('--ckpt', type=str,
                                default="./weights/minima_xoftr.ckpt",
                                help='Checkpoint path for XOFTR model')

        elif method == "loftr":
            parser.add_argument('--ckpt', type=str,
                                default="./weights/minima_loftr.ckpt",
                                help='Checkpoint path for LoFTR model')
            parser.add_argument('--thr', type=float, default=0.2,
                                help='Matching threshold for LoFTR')
        elif method == "sp_lg":
            parser.add_argument('--ckpt', type=str,
                                default="./weights/minima_lightglue.pth",
                                help='Checkpoint path for SuperPoint+LightGlue')
        elif method == "roma":
            parser.add_argument('--ckpt2', type=str,
                                default="large",
                                help='ROMA model size')
            parser.add_argument('--ckpt', type=str,
                                default='./weights/minima_roma.pth',
                                help='Checkpoint path for ROMA model')

        else:
            raise ValueError(f"Unknown matching method: {method}")

        add_common_arguments(parser)


    # Initialize argument parser
    parser = argparse.ArgumentParser(description='Benchmark Relative Pose Estimation')
    parser.add_argument('--method', type=str, default='roma',
                        choices=["xoftr", 'sp_lg', 'loftr', 'roma'],
                        help="Feature matching method to use: xoftr, sp_lg, loftr, roma")

    # Parse known arguments first to determine method
    args, remaining_args = parser.parse_known_args()
    add_method_arguments(parser, args.method)
    args = parser.parse_args()
    args.satellite_list = args.satellite_list.split(',')
    print("Configuration parameters:", args)

    # Load the selected matching model
    matcher = load_model(args.method, args)

    # Read test data file
    with open(args.test_data_file, 'r') as file:
        lines = file.readlines()

    # Define image preprocessing pipeline
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.46], std=[0.225]),
    ])

    # Process each satellite type
    for satellite_type in args.satellite_list:
        # Filter and prepare test data for current satellite
        test_data = []
        for line in lines:
            elements = line.strip().split()
            if len(elements) == 5 and satellite_type in elements[0]:
                test_data.append(elements)

        # Prepare output Excel file
        xls_path = fr'./result/3MOS_{satellite_type}_MINIMA_{args.method}_new_1013_256.xlsx'
        os.makedirs(r'./result', exist_ok=True)
        book = xlwt.Workbook(encoding="utf-8", style_compression=0)
        sheet1 = book.add_sheet('Sheet1', cell_overwrite_ok=True)
        test_num = len(test_data)
        print(f'Satellite: {satellite_type}, Test pairs: {test_num}')

        # Process each test pair
        for i in tqdm(range(test_num), desc=f'{args.method} {satellite_type}'):
            try:
            # Load and transform images
                img0_path = os.path.join(args.data_base_path, test_data[i][0])
                img1_path = os.path.join(args.data_base_path, test_data[i][1])
                tx, ty, angle = int(test_data[i][2]), int(test_data[i][3]), int(test_data[i][4])

                # Read and warp SAR image
                image1 = cv2.imread(img1_path, 0)  # Read as grayscale
                image1, H = rotate_and_translate_image_new(image1, angle, tx, ty)
                cv2.imwrite(f'./data/sar_warp_{args.method}.jpg', image1)

                # Prepare image pair for matching
                pair = {'im0': img0_path, 'im1': f'./data/sar_warp_{args.method}.jpg'}

                # Perform feature matching
                with torch.no_grad():
                    match_res = matcher(pair['im0'], pair['im1'])
                    mkpts0 = match_res['mkpts0']  # Matched keypoints in image 0
                    mkpts1 = match_res['mkpts1']  # Matched keypoints in image 1

                # Compute matching quality metrics
                error_threshold = 3  # pixels
                inlier_kpts_left, inlier_kpts_right, H_pred = FSC(mkpts0, mkpts1, error_threshold)
                n_inliers = len(inlier_kpts_left)

                # Changed variable name to avoid conflict with function name
                corner_err_value = corner_error(256, H_pred, H)  # Compute corner error

                print(f'Pair {i}: Matches={len(mkpts0)}, Inliers={n_inliers}, Corner Error={corner_err_value}')

                # Save results to Excel
                for col_index, data in enumerate(test_data[i]):
                    sheet1.write(i, col_index, data)
                sheet1.write(i, len(test_data[i]), corner_err_value)  # Use the new variable name here
                book.save(xls_path)

            except Exception as e:
                # Handle failures by writing default values
                print(f"Error processing pair {i}: {str(e)}")
                for col_index, data in enumerate(test_data[i]):
                    sheet1.write(i, col_index, data)
                sheet1.write(i, len(test_data[i]), 10000)  # Large error value for failures
                book.save(xls_path)
                continue

        # Calculate performance metrics for feature matching algorithms.
        metrics = calculate_matching_metrics(xls_path)
        print_metrics_latex_format(metrics, satellite_type, args.method)