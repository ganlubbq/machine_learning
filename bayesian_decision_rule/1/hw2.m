clear all; close all; clc;

%% Setup and import data
data_dir = fullfile(cd, 'homework2');
img = imread(fullfile(data_dir, 'cheetah.bmp'));
img = double(img)/255;
img_mask = imread(fullfile(data_dir, 'cheetah_mask.bmp'));
load(fullfile(data_dir,'TrainingSamplesDCT_8_new.mat'));
dimensions = 64;
fig = 1;

bg = TrainsampleDCT_BG;
fg = TrainsampleDCT_FG;

%% Collect 8x8 blocks and calculate DCT
block_size = 8;
[img_rows, img_cols] = size(img);

block_rows = img_rows-block_size+1;
block_cols = img_cols-block_size+1;
blocks = cell(block_rows, block_cols);
blocks_dct = blocks;

for i = 1:block_rows
    for j = 1:block_cols
        blocks{i,j} = img(i:i+block_size-1, j:j+block_size-1);
        blocks_dct{i,j} = dct2(blocks{i,j});
    end
end

%% Convert 8x8 blocks to vectors with zig-zag pattern
vector_stack = zeros(block_rows*block_cols, block_size*block_size);
index = 1;
for i = 1:block_rows
    for j = 1:block_cols
        vector_stack(index,:) = zigzag_index(blocks_dct{i,j});
        index = index+1;
    end
end

%% a) Calculate Py(FG) and Py(BG)
fprintf('> Part a\n');
[rows_bg, ~] = size(bg);
[rows_fg, ~] = size(fg);
rows_total = rows_bg + rows_fg;
p_bg = rows_bg/rows_total;
p_fg = rows_fg/rows_total;
fprintf('Py(BG) = %.3f\n', p_bg);
fprintf('Py(FG) = %.3f\n', p_fg);

figure; fig = fig + 1;
histogram('Categories', {'BG', 'FG'}, 'BinCounts', [p_bg, p_fg]);
title('Prior Probabilities');

%% b) 
% PDF of grass background
[mu_bg, cov_bg] = maximum_likelihood(bg);
for i = 1:dimensions
    x_bg(:,i)   = min(bg(:,i)):(max(bg(:,i))-min(bg(:,i)))/1000:max(bg(:,i));
    pdf_bg(:,i) = normpdf(x_bg(:,i), mu_bg(1,i), sqrt(cov_bg(i,i)));
end

% PDF of cheetah foreground
[mu_fg, cov_fg] = maximum_likelihood(fg);
for i = 1:dimensions
    x_fg(:,i)   = (min(fg(:,i)):(max(fg(:,i))-min(fg(:,i)))/1000:max(fg(:,i)))';
    pdf_fg(:,i) = normpdf(x_fg(:,i), mu_fg(1,i), sqrt(cov_fg(i,i)));
end

i = 1;
while i <= dimensions
    figure; fig = fig + 1;
    for j = 1:16
        subplot(4,4,j)
        hold on; grid on;
        plot(x_bg(:,i), pdf_bg(:,i));
        plot(x_fg(:,i), pdf_fg(:,i));
        hold off;
        title(['Feature ' num2str(i)]);
        i = i + 1;
    end    
end

% 8 best/worst
delta  = abs(mu_fg - mu_bg) + abs(cov_fg - cov_bg);
[~, delta_index ] = sort(delta(1,:), 'descend' );
best = delta_index(1:8)
[~, delta_index ] = sort(delta(1,:), 'ascend' );
worst = delta_index(1:8)

% Plot 8 best
index = 1;
figure; fig = fig + 1;
for i = best
    subplot(4,2,index);
    hold on
    plot(x_bg(:,i), pdf_bg(:,i));
    plot(x_fg(:,i), pdf_fg(:,i));
    hold off
    title(['#' num2str(index) ' Best - Index: ' num2str(i)]);
    legend('bg', 'fg')
    index = index + 1;
end

% Plot 8 worst
index = 1;
figure; fig = fig + 1;
for i = worst
    subplot(4,2,index);
    hold on
    plot(x_bg(:,i), pdf_bg(:,i));
    plot(x_fg(:,i), pdf_fg(:,i));
    hold off
    title(['#' num2str(index) ' Worst - Index: ' num2str(i)]);
    index = index + 1;
end

%% c)
% Make classifier using 64 features
[rows, cols] = size(vector_stack);

W_bg = -0.5.*inv(cov_bg);
w_bg = inv(cov_bg)*mu_bg';
w0_bg = -0.5.*(mu_bg*inv(cov_bg)*mu_bg')-0.5*log(det(cov_bg))+log(p_bg);

W_fg = -0.5.*inv(cov_fg);
w_fg = inv(cov_fg)*mu_fg';
w0_fg = -0.5.*(mu_fg*inv(cov_fg)*mu_fg')-0.5*log(det(cov_fg))+log(p_fg);

for i = 1:rows
    g64_bg(i,:) = vector_stack(i,:)*W_bg*vector_stack(i,:)'+w_bg'*vector_stack(i,:)'+w0_bg;
    g64_fg(i,:) = vector_stack(i,:)*W_fg*vector_stack(i,:)'+w_fg'*vector_stack(i,:)'+w0_fg;
%     i64_bg(i,:) = (vector_stack(i,:)-mu_bg)*inv(cov_bg)*(vector_stack(i,:)-mu_bg)' + (det(cov_bg) - 2*log(p_bg));
%     i64_fg(i,:) = (vector_stack(i,:)-mu_fg)*inv(cov_fg)*(vector_stack(i,:)-mu_fg)' + (det(cov_fg) - 2*log(p_fg));
end

% Make classifier using 8 features
[mu8_bg, cov8_bg] = maximum_likelihood(bg(:,best));
[mu8_fg, cov8_fg] = maximum_likelihood(fg(:,best));
[rows, cols] = size(vector_stack);

W_bg = -0.5.*inv(cov8_bg);
w_bg = inv(cov8_bg)*mu8_bg';
w0_bg = -0.5.*(mu8_bg*inv(cov8_bg)*mu8_bg')-0.5*log(det(cov8_bg))+log(p_bg);

W_fg = -0.5.*inv(cov8_fg);
w_fg = inv(cov8_fg)*mu8_fg';
w0_fg = -0.5.*(mu8_fg*inv(cov8_fg)*mu8_fg')-0.5*log(det(cov8_fg))+log(p_fg);

for i = 1:rows
    g8_bg(i,:) = vector_stack(i,best)*W_bg*vector_stack(i,best)'+w_bg'*vector_stack(i,best)'+w0_bg;
    g8_fg(i,:) = vector_stack(i,best)*W_fg*vector_stack(i,best)'+w_fg'*vector_stack(i,best)'+w0_fg;
%     i8_bg(i,:) = (vector_stack(i,best)-mu8_bg)*inv(cov8_bg)*(vector_stack(i,best)-mu8_bg)' + (det(cov8_bg) - 2*log(p_bg));
%     i8_fg(i,:) = (vector_stack(i,best)-mu8_fg)*inv(cov8_fg)*(vector_stack(i,best)-mu8_fg)' + (det(cov8_fg) - 2*log(p_fg));
end


[A64, img_A64] = create_mask(g64_fg, g64_bg, img, block_rows, block_cols);
figure
imagesc(A64);
colormap(gray(255));
title('Algorithm Result w/64 Features');
figure
imagesc(img_A64);
colormap(gray(255));
title('Algorithm Result w/64 Features Overlay');

[A8, img_A8] = create_mask(g8_fg, g8_bg, img, block_rows, block_cols);
figure
imagesc(A8);
colormap(gray(255));
title('Algorithm Result w/8 Features');
figure
imagesc(img_A8);
colormap(gray(255));
title('Algorithm Result w/8 Features Overlay');

%Probability of error 64
p_error_64 = probability_error(A64, img_mask, block_rows, block_cols);
fprintf('Probability of Error (64) = %.3f\n', p_error_64);

%Probability of error 8
p_error_8 = probability_error(A8, img_mask, block_rows, block_cols);
fprintf('Probability of Error (8) = %.3f\n', p_error_8);

%% Helper functions

% Turns array into vector ordered in the provided zig-zag pattern
function return_vector = zigzag_index(array)
    [rows, cols] = size(array);
    vector_length = rows*cols;
    return_vector = zeros(1, vector_length);
    index = 0;
    for k = 2:vector_length
    state = mod(k,2);
        for i = 1:rows
            for j = 1:cols
                if((i+j) == k)
                    index = index+1;
                    if(state == 0)
                        return_vector(index) = array(j,k-j);
                    else          
                        return_vector(index) = array(k-j,j);
                    end
                end    
            end
        end
    end
end

function [mu, cov] = maximum_likelihood(array)
    [rows, cols] = size(array);
    mu = sum(array)/rows;
    data = zeros(rows,cols);
    for i=1:cols
        data(:,i)= ((array(:,i) - mu(i)));
    end
    cov = (data.'*data)/rows;
end

function [A, img_A] = create_mask(BG, FG, img, block_rows, block_cols)
    index = 1;
    img_A = img;
    A = zeros(block_rows, block_cols);
    [rows, ~] = size(BG);
    while index <= rows
        for i = 1:block_rows
            for j = 1:block_cols
                if FG(index,:) < BG(index,:)
                    A(i,j) = 1;
                    img_A(i,j) = 1;
                end
                index = index+1;
            end
        end
    end
end

function p_error = probability_error(array, img_mask, block_rows, block_cols)
    [rows_A, cols_A] = size(array);
    img_mask = img_mask(1:rows_A, 1:cols_A)/255;
    figure
    imagesc(img_mask);
    colormap(gray(255));
    title('Provided Mask (Trimmed)');
    p_g1y1 = 0;
    p_g1y0 = 0;
    p_g0y1 = 0;
    p_g0y0 = 0;
    cmp_total = rows_A * cols_A;
    for i = 1:rows_A
        for j = 1:cols_A
            if(array(i,j) == 1 && img_mask(i,j) == 1)
                p_g1y1 = p_g1y1 + 1;
            end
            if(array(i,j) == 1 && img_mask(i,j) == 0)
                p_g1y0 = p_g1y0 + 1;
            end
            if(array(i,j) == 0 && img_mask(i,j) == 1)
                p_g0y1 = p_g0y1 + 1;
            end
            if(array(i,j) == 0 && img_mask(i,j) == 0)
                p_g0y0 = p_g0y0 + 1;
            end
        end
    end
    p_g1y0a = p_g1y0 / ((block_rows*block_cols) - sum(sum(img_mask)));
    p_g0y1a = p_g0y1 / sum(sum(img_mask));
    p_fg = sum(sum(img_mask)) / (block_rows*block_cols);
    p_bg = 1 - p_fg;
    p_error = p_g1y0a*p_bg + p_g0y1a*p_fg;
end