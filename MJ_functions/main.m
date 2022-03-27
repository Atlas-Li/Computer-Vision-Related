close all; clear all; clc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% visualize thermal images
thermal_root_path = 'E:\calibration\Azure\ir\';
suffix = 'png';
vis_out_root = 'E:\calibration\Azure\vis_ir\';
vis_out = adjust_images(thermal_root_path, suffix, vis_out_root);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% target_size = [320,320];
% crop_out_root = 'E:\calibration\crop\';
% crop_out = crop_img(vis_out_root, target_size, suffix, crop_out_root);
