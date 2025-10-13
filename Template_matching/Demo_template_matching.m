clc;
clear;
close all;

% Add required toolboxes to path
addpath HOPC_FFT

% Configuration parameters
Template_size = 128;      % Template size for matching
base_path = 'E:\temp\ubuntu\crop_data\'; % Base path for image data
txt_file_path = 'E:\temp\ubuntu\crop_data\test_template_matching.txt'; % Path to your text file

% Method to use for matching
Method = {'HOPC'};        % Matching method: HOPC

% Satellite types to test (modify this list as needed)
satellite_types = {'ALOS'}; % Test all satellites
% satellite_types = {'ALOS', 'GF3', 'SEN', 'RCM', 'Radarsat'}; % Test specific satellites only

% Read all data from text file
fid = fopen(txt_file_path, 'r');
if fid == -1
    error('Cannot open file: %s', txt_file_path);
end

% Read all lines
file_content = textscan(fid, '%s %s %d %d');
fclose(fid);

% Extract data from file
img_opt_all = file_content{1};
img_sar_all = file_content{2};
coor_x_all = file_content{3};
coor_y_all = file_content{4};

% Get total number of test cases
total_test_num = length(img_opt_all);
fprintf('Total test cases in file: %d\n', total_test_num);

% Filter test cases based on satellite types
selected_indices = [];
for i = 1:total_test_num
    % Check if the optical image path contains any of the specified satellite types
    opt_path = img_opt_all{i};
    for j = 1:length(satellite_types)
        if contains(opt_path, satellite_types{j})
            selected_indices = [selected_indices, i];
            break;
        end
    end
end

% Get filtered test cases
test_num = length(selected_indices);
fprintf('Selected test cases for satellites %s: %d\n', strjoin(satellite_types, ', '), test_num);

if test_num == 0
    error('No test cases found for the specified satellite types.');
end

% Initialize error matrix
All_error = zeros(test_num, length(Method));

% Start timing
tic

% Simple progress bar implementation
fprintf('Progress: [');
progress_bar_length = 50; % Length of the progress bar
progress_step = max(1, round(test_num / progress_bar_length));

% Main processing loop
for idx = 1:test_num
    % Get the actual index in the original data
    index = selected_indices(idx);
    
    % Construct full image paths with base path
    opt_path = fullfile(base_path, img_opt_all{index});
    sar_path = fullfile(base_path, img_sar_all{index});
    x_true = coor_x_all(index);
    y_true = coor_y_all(index);
    
    % Read images
    im_opt = imread(opt_path);          % Optical image
    im_SAR_origin = imread(sar_path);   % SAR image
    
    % Convert optical image to grayscale if needed
    [height, width, num_channels] = size(im_opt);
    if num_channels >= 3
        im_opt = im_opt(:, :, 1:3);     % Keep only RGB channels
        im_opt = rgb2gray(im_opt);      % Convert to grayscale
    end
    
    % Extract template from SAR image using given coordinates
    im_SAR = im_SAR_origin(y_true:y_true+Template_size-1, x_true:x_true+Template_size-1);
    
    % Initialize errors for current test case
    errors = [];
    
    % Process with each matching method
    for i = 1:length(Method)
        try
            switch Method{i}
                case 'HOPC'
                    % Perform HOPC matching
                    [~, Match_Position] = toc_FFT_HOPC_match(im_SAR, im_opt);
                    
                    % Calculate Euclidean distance error
                    error_value = sqrt((Match_Position(1) - double(x_true))^2 + (Match_Position(2) - double(y_true))^2);
                    
                    errors = [errors; error_value];
            end
            
        catch
            % If matching fails, assign large error value
            errors = [errors; 10000];
        end
    end
    
    % Store errors for current test case
    All_error(idx, :) = errors;
    
    % Update progress bar
    if mod(idx, progress_step) == 0 || idx == test_num
        progress_percent = round(idx / test_num * progress_bar_length);
        fprintf(repmat('=', 1, progress_percent - floor((idx-1) / test_num * progress_bar_length)));
    end
    
    % Display detailed progress every 100 cases
    if mod(idx, 100) == 0
        fprintf('] %d/%d (%.1f%%)', idx, test_num, idx/test_num*100);
        fprintf('\nProgress: [');
    end
end

% Complete progress bar
fprintf('] 100%%\n');

% Display elapsed time
elapsed_time = toc;
fprintf('Processing completed in %.2f seconds\n', elapsed_time);

% Calculate and display basic statistics
fprintf('\n=== Results Summary ===\n');
fprintf('Satellites tested: %s\n', strjoin(satellite_types, ', '));
fprintf('Total cases processed: %d\n', test_num);

for i = 1:length(Method)
    method_errors = All_error(:, i);
    success_3px = sum(method_errors <= 3);
    success_6px = sum(method_errors <= 6);
    success_10px = sum(method_errors <= 10);
    success_20px = sum(method_errors <= 20);
    fprintf('\nMethod: %s\n', Method{i});
    fprintf('Success rate (≤3px): %.2f%% (%d/%d)\n', success_3px/test_num*100, success_3px, test_num);
    fprintf('Success rate (≤6px): %.2f%% (%d/%d)\n', success_6px/test_num*100, success_6px, test_num);
    fprintf('Success rate (≤10px): %.2f%% (%d/%d)\n', success_10px/test_num*100, success_10px, test_num);
    fprintf('Success rate (≤20px): %.2f%% (%d/%d)\n', success_20px/test_num*100, success_20px, test_num);

end

% Save results to file
result_filename = sprintf('matching_results_%s.mat', strjoin(satellite_types, '_'));
save(result_filename, 'All_error', 'test_num', 'Method', 'satellite_types');
fprintf('\nResults saved to %s\n', result_filename);