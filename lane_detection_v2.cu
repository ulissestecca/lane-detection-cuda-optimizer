#define __STRICT_ANSI__
#include <math.h>
#undef cospi
#undef sinpi
#undef cospif
#undef sinpif

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

using namespace cv;

#define CUDA_CHECK(call)                                              \
    do                                                                \
    {                                                                 \
        cudaError_t _e = (call);                                      \
        if (_e != cudaSuccess)                                        \
        {                                                             \
            std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__ \
                      << " — " << cudaGetErrorString(_e) << "\n";     \
            std::exit(EXIT_FAILURE);                                  \
        }                                                             \
    } while (0)

__constant__ int c_sobelX[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
__constant__ int c_sobelY[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
__constant__ int c_gauss[9] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
__constant__ float c_cos[180];
__constant__ float c_sin[180];

// Convert input image to grayscale
__global__ void rgb2gray(const uchar3 *__restrict__ input,
                         unsigned char *__restrict__ gray,
                         int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    // Compute grayscale pixel using aligned uchar3 read and standard RGB weights
    uchar3 px = input[y * width + x];
    gray[y * width + x] = (unsigned char)(0.114f * px.x + 0.587f * px.y + 0.299f * px.z);
}

#define TILE_W 16
#define TILE_H 16

// Helper function to compute blur from global memory for boundary pixels
__device__ inline float compute_blur_from_global(const unsigned char *__restrict__ gray,
                                                 int width, int height,
                                                 int gx, int gy, int dx, int dy)
{
    int blur_sum = 0, k_idx = 0;
    for (int ky = -1; ky <= 1; ky++)
    {
        for (int kx = -1; kx <= 1; kx++)
        {
            int cx = min(max(gx + dx + kx, 0), width - 1);
            int cy = min(max(gy + dy + ky, 0), height - 1);
            blur_sum += gray[cy * width + cx] * c_gauss[k_idx++];
        }
    }
    return blur_sum / 16.0f;
}

// Apply blur, Sobel filter and mask
__global__ void blur_sobel_mask_fused(const unsigned char *__restrict__ gray,
                                      const unsigned char *__restrict__ mask,
                                      unsigned char *__restrict__ filtered,
                                      int width, int height, float threshold)
{
    // Load data into shared memory, including halo region for the blur
    __shared__ unsigned char smem[TILE_H + 2][TILE_W + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int gx = blockIdx.x * TILE_W + tx;
    int gy = blockIdx.y * TILE_H + ty;

    // Load the core 16x16 pixels and the surrounding 1-pixel halo into shared memory
    int cx = min(max(gx, 0), width - 1);
    int cy = min(max(gy, 0), height - 1);
    smem[ty + 1][tx + 1] = gray[cy * width + cx];

    if (tx == 0)
        smem[ty + 1][0] = gray[cy * width + min(max(gx - 1, 0), width - 1)];
    if (tx == TILE_W - 1)
        smem[ty + 1][TILE_W + 1] = gray[cy * width + min(max(gx + 1, 0), width - 1)];
    if (ty == 0)
        smem[0][tx + 1] = gray[min(max(gy - 1, 0), height - 1) * width + cx];
    if (ty == TILE_H - 1)
        smem[TILE_H + 1][tx + 1] = gray[min(max(gy + 1, 0), height - 1) * width + cx];

    if (tx == 0 && ty == 0)
        smem[0][0] = gray[min(max(gy - 1, 0), height - 1) * width + min(max(gx - 1, 0), width - 1)];
    if (tx == TILE_W - 1 && ty == 0)
        smem[0][TILE_W + 1] = gray[min(max(gy - 1, 0), height - 1) * width + min(max(gx + 1, 0), width - 1)];
    if (tx == 0 && ty == TILE_H - 1)
        smem[TILE_H + 1][0] = gray[min(max(gy + 1, 0), height - 1) * width + min(max(gx - 1, 0), width - 1)];
    if (tx == TILE_W - 1 && ty == TILE_H - 1)
        smem[TILE_H + 1][TILE_W + 1] = gray[min(max(gy + 1, 0), height - 1) * width + min(max(gx + 1, 0), width - 1)];

    __syncthreads();

    __shared__ float s_blurred[TILE_H + 2][TILE_W + 2];

    s_blurred[ty + 1][tx + 1] = 0.0f;
    // Compute the 1-pixel blurred boundary halo directly from global memory to prevent edge artifacts
    if (tx == 0)
        s_blurred[ty + 1][0] = compute_blur_from_global(gray, width, height, gx, gy, -1, 0);
    if (tx == TILE_W - 1)
        s_blurred[ty + 1][TILE_W + 1] = compute_blur_from_global(gray, width, height, gx, gy, 1, 0);
    if (ty == 0)
        s_blurred[0][tx + 1] = compute_blur_from_global(gray, width, height, gx, gy, 0, -1);
    if (ty == TILE_H - 1)
        s_blurred[TILE_H + 1][tx + 1] = compute_blur_from_global(gray, width, height, gx, gy, 0, 1);
    if (tx == 0 && ty == 0)
        s_blurred[0][0] = compute_blur_from_global(gray, width, height, gx, gy, -1, -1);
    if (tx == TILE_W - 1 && ty == 0)
        s_blurred[0][TILE_W + 1] = compute_blur_from_global(gray, width, height, gx, gy, 1, -1);
    if (tx == 0 && ty == TILE_H - 1)
        s_blurred[TILE_H + 1][0] = compute_blur_from_global(gray, width, height, gx, gy, -1, 1);
    if (tx == TILE_W - 1 && ty == TILE_H - 1)
        s_blurred[TILE_H + 1][TILE_W + 1] = compute_blur_from_global(gray, width, height, gx, gy, 1, 1);

    // Compute the core convolution for the inner pixels using the fast shared memory
    if (gx < width && gy < height)
    {
        int blur_sum = 0, k = 0;
        for (int ky = 0; ky <= 2; ky++)
            for (int kx = 0; kx <= 2; kx++)
                blur_sum += smem[ty + ky][tx + kx] * c_gauss[k++];
        s_blurred[ty + 1][tx + 1] = blur_sum / 16.0f;
    }

    __syncthreads();

    if (gx >= width || gy >= height)
        return;

    int idx = gy * width + gx;

    if (gx < 1 || gy < 1 || gx >= width - 1 || gy >= height - 1)
    {
        filtered[idx] = 0;
        return;
    }

    // Compute gradient magnitude using Sobel operator on the blurred results
    float Gx = 0.0f, Gy = 0.0f;
    int k = 0;
    for (int ky = 0; ky <= 2; ky++)
        for (int kx = 0; kx <= 2; kx++)
        {
            float p = s_blurred[ty + ky][tx + kx];
            Gx += p * c_sobelX[k];
            Gy += p * c_sobelY[k];
            k++;
        }

    float mag = sqrtf(Gx * Gx + Gy * Gy);

    // Retain pixel only if magnitude exceeds threshold and it is within the ROI mask
    filtered[idx] = (mag > threshold && mask[idx] > 0) ? 255 : 0;
}

// Detect lines using Hough transform
__global__ void hough_transform(const unsigned char *__restrict__ d_edges,
                                int *d_accumulator,
                                int width, int height,
                                float rho_step, int num_rho,
                                int num_theta, float max_dist)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;
    if (d_edges[y * width + x] == 0)
        return;

    // Cast votes in accumulator using precomputed trigonometric values
    for (int t = 0; t < num_theta; ++t)
    {
        float rho = x * c_cos[t] + y * c_sin[t];
        int r_idx = __float2int_rn((rho + max_dist) / rho_step);
        if (r_idx >= 0 && r_idx < num_rho)
            atomicAdd(&d_accumulator[t * num_rho + r_idx], 1);
    }
}

// Main execution entry point
int main()
{
    // Open input video stream
    VideoCapture cap("input.mp4");
    if (!cap.isOpened())
    {
        std::cerr << "Error opening video.\n";
        return -1;
    }

    int width = (int)cap.get(CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
    int fps = (int)cap.get(CAP_PROP_FPS);

    VideoWriter out("output.mp4", VideoWriter::fourcc('M', 'P', '4', 'V'),
                    fps, Size(width, height), true);

    // Precompute sine and cosine for Hough transform
    float h_cos[180], h_sin[180];
    const float theta_step = (float)CV_PI / 180.0f;
    for (int t = 0; t < 180; t++)
    {
        h_cos[t] = cosf(t * theta_step);
        h_sin[t] = sinf(t * theta_step);
    }
    CUDA_CHECK(cudaMemcpyToSymbol(c_cos, h_cos, 180 * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_sin, h_sin, 180 * sizeof(float)));

    // Define region of interest (ROI) mask
    Mat h_mask = Mat::zeros(height, width, CV_8UC1);
    {
        Point pts[4] = {
            Point((int)(width * 0.10f), (int)(height * 0.80f)),
            Point((int)(width * 0.45f), (int)(height * 0.55f)),
            Point((int)(width * 0.55f), (int)(height * 0.55f)),
            Point((int)(width * 0.90f), (int)(height * 0.80f))};
        const Point *ppt[1] = {pts};
        int npt[] = {4};
        fillPoly(h_mask, ppt, npt, 1, Scalar(255), LINE_8);
    }

    uchar3 *d_input;
    unsigned char *d_gray, *d_filtered, *d_mask;
    int *d_accumulator;

    const float max_dist = ceilf(sqrtf((float)(width * width + height * height)));
    const int num_rho = 2 * (int)max_dist;
    const int num_theta = 180;
    const int acc_size = num_theta * num_rho * (int)sizeof(int);

    // Allocate device memory for images and intermediate results
    CUDA_CHECK(cudaMalloc(&d_input, width * height * sizeof(uchar3)));
    CUDA_CHECK(cudaMalloc(&d_gray, width * height));
    CUDA_CHECK(cudaMalloc(&d_filtered, width * height));
    CUDA_CHECK(cudaMalloc(&d_mask, width * height));
    CUDA_CHECK(cudaMalloc(&d_accumulator, acc_size));
    CUDA_CHECK(cudaMemcpy(d_mask, h_mask.data, width * height,
                          cudaMemcpyHostToDevice));

    int *h_accumulator = (int *)malloc(acc_size);

    dim3 block(TILE_W, TILE_H);
    dim3 grid((width + TILE_W - 1) / TILE_W, (height + TILE_H - 1) / TILE_H);

    Mat frame, final_out;
    int frame_count = 0;

    // Process video frames
    while (cap.read(frame))
    {
        frame_count++;
        if (!frame.isContinuous())
            frame = frame.clone();

        CUDA_CHECK(cudaMemcpy(d_input, frame.data,
                              width * height * sizeof(uchar3),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_accumulator, 0, acc_size));

        // Execute CUDA kernels for image processing pipeline
        rgb2gray<<<grid, block>>>(d_input, d_gray, width, height);
        CUDA_CHECK(cudaDeviceSynchronize());

        blur_sobel_mask_fused<<<grid, block>>>(d_gray, d_mask, d_filtered,
                                               width, height, 80.0f);
        CUDA_CHECK(cudaDeviceSynchronize());

        hough_transform<<<grid, block>>>(d_filtered, d_accumulator,
                                         width, height,
                                         1.0f, num_rho, num_theta, max_dist);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_accumulator, d_accumulator,
                              acc_size, cudaMemcpyDeviceToHost));

        final_out = frame.clone();
        // Extract left and right lane candidates from accumulator
        int left_max = 0, right_max = 0;
        int left_t = -1, left_r = -1, right_t = -1, right_r = -1;

        for (int t = 0; t < num_theta; t++)
        {
            if (t < 20 || t > 160)
                continue;
            if (t > 70 && t < 110)
                continue;
            for (int r = 0; r < num_rho; r++)
            {
                int v = h_accumulator[t * num_rho + r];
                if (v < 80)
                    continue;
                if (t < 90)
                {
                    if (v > left_max)
                    {
                        left_max = v;
                        left_t = t;
                        left_r = r;
                    }
                }
                else
                {
                    if (v > right_max)
                    {
                        right_max = v;
                        right_t = t;
                        right_r = r;
                    }
                }
            }
        }

        auto draw_lane = [&](int t, int r)
        {
            if (t == -1)
                return;
            float theta = t * theta_step, rho = r - max_dist;
            float cos_t = cosf(theta), sin_t = sinf(theta);
            if (fabsf(cos_t) < 1e-6f)
                return;
            float y_bot = height * 0.80f, y_top = height * 0.55f;
            line(final_out,
                 Point(cvRound((rho - y_bot * sin_t) / cos_t), (int)y_bot),
                 Point(cvRound((rho - y_top * sin_t) / cos_t), (int)y_top),
                 Scalar(0, 0, 255), 4, LINE_AA);
        };

        draw_lane(left_t, left_r);
        draw_lane(right_t, right_r);

        out.write(final_out);
        if (frame_count % 30 == 0)
            std::cout << "Processed " << frame_count << " frames...\n";
    }

    // Free allocated memory
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_gray));
    CUDA_CHECK(cudaFree(d_filtered));
    CUDA_CHECK(cudaFree(d_mask));
    CUDA_CHECK(cudaFree(d_accumulator));
    free(h_accumulator);
    cap.release();
    out.release();
    std::cout << "Processing Complete.\n";
    return 0;
}
