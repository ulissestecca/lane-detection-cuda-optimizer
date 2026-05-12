#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <vector>

using namespace cv;

// Convert input image to grayscale
__global__ void rgb2gray(unsigned char *input, unsigned char *gray, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;
    int i3 = idx * 3;

    // Compute grayscale value using standard RGB weights
    gray[idx] = 0.114f * input[i3] + 0.587f * input[i3 + 1] + 0.299f * input[i3 + 2];
}

// Apply Gaussian blur
__global__ void gaussian_blur(unsigned char *input, unsigned char *output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || y < 1 || x >= width - 1 || y >= height - 1)
        return;

    int filter[3][3] = {
        {1, 2, 1},
        {2, 4, 2},
        {1, 2, 1}};

    // Iterate over 3x3 filter to compute weighted sum
    int sum = 0;
    for (int ky = -1; ky <= 1; ky++)
    {
        for (int kx = -1; kx <= 1; kx++)
        {
            sum += input[(y + ky) * width + (x + kx)] * filter[ky + 1][kx + 1];
        }
    }
    output[y * width + x] = sum / 16;
}

// Detect edges using Sobel operator
__global__ void sobel_edges(unsigned char *gray, unsigned char *edges, int width, int height, float threshold)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || y < 1 || x >= width - 1 || y >= height - 1)
        return;

    int sobelX[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int sobelY[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    float Gx = 0;
    float Gy = 0;

    // Apply the 3x3 Sobel filters to compute X and Y gradients

    for (int ky = -1; ky <= 1; ky++)
    {
        for (int kx = -1; kx <= 1; kx++)
        {
            float pixel = gray[(y + ky) * width + (x + kx)];
            Gx += pixel * sobelX[ky + 1][kx + 1];
            Gy += pixel * sobelY[ky + 1][kx + 1];
        }
    }

    // Calculate gradient magnitude
    float mag = sqrtf(Gx * Gx + Gy * Gy);
    edges[y * width + x] = (mag > threshold) ? 255 : 0;
}

// Apply region of interest mask
__global__ void apply_mask(unsigned char *edges, unsigned char *mask, unsigned char *filtered, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;
    int idx = y * width + x;

    // Retain edge pixel only if it falls within the ROI mask
    filtered[idx] = (mask[idx] > 0) ? edges[idx] : 0;
}

// Detect lines using Hough transform
__global__ void hough_transform(unsigned char *d_edges, int *d_accumulator,
                                int width, int height,
                                float rho_step, float theta_step,
                                int num_rho, int num_theta, float max_dist)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    // Cast votes in the accumulator for all lines passing through an edge pixel
    if (d_edges[y * width + x] > 0)
    {
        for (int t = 0; t < num_theta; ++t)
        {
            float theta = t * theta_step;
            float rho = x * cosf(theta) + y * sinf(theta);

            int r_idx = roundf((rho + max_dist) / rho_step);

            if (r_idx >= 0 && r_idx < num_rho)
            {
                atomicAdd(&d_accumulator[t * num_rho + r_idx], 1);
            }
        }
    }
}

// Main execution entry point
int main()
{
    // Open input video stream
    VideoCapture cap("input.mp4");
    if (!cap.isOpened())
    {
        std::cout << "Error opening video stream.\n";
        return -1;
    }

    int width = cap.get(CAP_PROP_FRAME_WIDTH);
    int height = cap.get(CAP_PROP_FRAME_HEIGHT);
    int fps = cap.get(CAP_PROP_FPS);

    VideoWriter out("output_debug.mp4", VideoWriter::fourcc('M', 'P', '4', 'V'), fps, Size(width * 2, height), true);

    // Define region of interest (ROI) mask
    Mat h_mask = Mat::zeros(height, width, CV_8UC1);
    Point pts[1][4];
    pts[0][0] = Point(width * 0.1, height * 0.80);
    pts[0][1] = Point(width * 0.45, height * 0.55);
    pts[0][2] = Point(width * 0.55, height * 0.55);
    pts[0][3] = Point(width * 0.9, height * 0.80);
    const Point *ppt[1] = {pts[0]};
    int npt[] = {4};
    fillPoly(h_mask, ppt, npt, 1, Scalar(255), LINE_8);

    // Allocate device memory for images and intermediate results
    unsigned char *d_input, *d_gray, *d_blurred, *d_edges, *d_mask, *d_filtered;
    cudaMalloc(&d_input, width * height * 3);
    cudaMalloc(&d_gray, width * height);
    cudaMalloc(&d_blurred, width * height);
    cudaMalloc(&d_edges, width * height);
    cudaMalloc(&d_mask, width * height);
    cudaMalloc(&d_filtered, width * height);

    cudaMemcpy(d_mask, h_mask.data, width * height, cudaMemcpyHostToDevice);

    float theta_step = CV_PI / 180.0f;
    float rho_step = 1.0f;
    int num_theta = 180;
    float max_dist = ceil(sqrt(width * width + height * height));
    int num_rho = 2 * (int)max_dist;

    int acc_size = num_theta * num_rho * sizeof(int);
    int *d_accumulator;
    cudaMalloc(&d_accumulator, acc_size);
    int *h_accumulator = (int *)malloc(acc_size);

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    Mat frame, final_out, edges_out(height, width, CV_8UC1), edges_bgr, combined_out;
    int frame_count = 0;

    // Process video frames
    while (cap.read(frame))
    {
        frame_count++;

        cudaMemcpy(d_input, frame.data, width * height * 3, cudaMemcpyHostToDevice);
        cudaMemset(d_accumulator, 0, acc_size);

        // Execute CUDA kernels for image processing pipeline
        rgb2gray<<<grid, block>>>(d_input, d_gray, width, height);
        gaussian_blur<<<grid, block>>>(d_gray, d_blurred, width, height);
        sobel_edges<<<grid, block>>>(d_blurred, d_edges, width, height, 80.0f);
        apply_mask<<<grid, block>>>(d_edges, d_mask, d_filtered, width, height);

        hough_transform<<<grid, block>>>(d_filtered, d_accumulator, width, height,
                                         rho_step, theta_step, num_rho, num_theta, max_dist);

        cudaMemcpy(h_accumulator, d_accumulator, acc_size, cudaMemcpyDeviceToHost);

        cudaMemcpy(edges_out.data, d_edges, width * height, cudaMemcpyDeviceToHost);

        cvtColor(edges_out, edges_bgr, COLOR_GRAY2BGR);
        final_out = frame.clone();

        // Extract left and right lane candidates from accumulator
        int left_max_votes = 0, right_max_votes = 0;
        int left_best_t = -1, left_best_r = -1;
        int right_best_t = -1, right_best_r = -1;

        for (int t = 0; t < num_theta; t++)
        {
            if (t > 70 && t < 110)
                continue;
            if (t < 20 || t > 160)
                continue;

            for (int r = 0; r < num_rho; r++)
            {
                int votes = h_accumulator[t * num_rho + r];
                if (votes >= 80)
                {
                    if (t < 90)
                    {
                        if (votes > left_max_votes)
                        {
                            left_max_votes = votes;
                            left_best_t = t;
                            left_best_r = r;
                        }
                    }
                    else
                    {
                        if (votes > right_max_votes)
                        {
                            right_max_votes = votes;
                            right_best_t = t;
                            right_best_r = r;
                        }
                    }
                }
            }
        }

        auto draw_lane = [&](int t, int r)
        {
            if (t == -1)
                return;

            float theta = t * theta_step;
            float rho = r * rho_step - max_dist;

            float y_bottom = height * 0.80f;
            float y_top = height * 0.55f;

            float cos_t = cosf(theta);
            float sin_t = sinf(theta);

            if (fabs(cos_t) < 1e-6)
                return;

            int x_bottom = cvRound((rho - y_bottom * sin_t) / cos_t);
            int x_top = cvRound((rho - y_top * sin_t) / cos_t);

            line(final_out, Point(x_bottom, y_bottom), Point(x_top, y_top), Scalar(0, 0, 255), 4, LINE_AA);

            line(edges_bgr, Point(x_bottom, y_bottom), Point(x_top, y_top), Scalar(0, 0, 255), 4, LINE_AA);
        };

        draw_lane(left_best_t, left_best_r);
        draw_lane(right_best_t, right_best_r);

        hconcat(final_out, edges_bgr, combined_out);
        out.write(combined_out);

        if (frame_count % 30 == 0)
            std::cout << "Processed " << frame_count << " frames...\n";
    }

    // Free allocated memory
    cudaFree(d_input);
    cudaFree(d_gray);
    cudaFree(d_blurred);
    cudaFree(d_edges);
    cudaFree(d_mask);
    cudaFree(d_filtered);
    cudaFree(d_accumulator);
    free(h_accumulator);

    cap.release();
    out.release();
    std::cout << "Processing Complete. Check 'output_debug.mp4'.\n";
    return 0;
}
