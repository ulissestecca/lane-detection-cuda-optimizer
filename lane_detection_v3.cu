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
#include <cstring>

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
__global__ void rgb2gray_vec4(const uchar3 *__restrict__ input,
                              unsigned char *__restrict__ gray,
                              int width, int height)
{
    // Calculate global thread coordinates for vectorized 4-pixel access
    int x4 = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int x = x4 * 4;
    // Handle image boundaries where 4-pixel chunks don't perfectly fit
    if (x + 3 >= width)
    {

        for (int i = 0; i < 4 && x + i < width; i++)
        {
            uchar3 px = input[y * width + x + i];
            gray[y * width + x + i] = (unsigned char)(0.114f * px.x + 0.587f * px.y + 0.299f * px.z);
        }
        return;
    }

    // Load 4 contiguous pixels using aligned memory access
    uchar3 p0 = input[y * width + x + 0];
    uchar3 p1 = input[y * width + x + 1];
    uchar3 p2 = input[y * width + x + 2];
    uchar3 p3 = input[y * width + x + 3];

    // Compute grayscale values and store them as a single uchar4 vector
    uchar4 out;
    out.x = (unsigned char)(0.114f * p0.x + 0.587f * p0.y + 0.299f * p0.z);
    out.y = (unsigned char)(0.114f * p1.x + 0.587f * p1.y + 0.299f * p1.z);
    out.z = (unsigned char)(0.114f * p2.x + 0.587f * p2.y + 0.299f * p2.z);
    out.w = (unsigned char)(0.114f * p3.x + 0.587f * p3.y + 0.299f * p3.z);

    *reinterpret_cast<uchar4 *>(&gray[y * width + x]) = out;
}

#define TILE_W 16
#define TILE_H 16

// Apply blur, Sobel filter and mask
__global__ void blur_sobel_mask_tex(cudaTextureObject_t tex,
                                    const unsigned char *__restrict__ mask,
                                    unsigned char *__restrict__ filtered,
                                    int width, int height, float threshold)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx >= width || gy >= height)
        return;

    int idx = gy * width + gx;

    if (gx < 2 || gy < 2 || gx >= width - 2 || gy >= height - 2)
    {
        filtered[idx] = 0;
        return;
    }

    float Gx = 0.0f, Gy = 0.0f;
    int k_sobel = 0;

    // Apply Sobel filter over a 3x3 window. For each position, compute 
    // the Gaussian blur on-the-fly over a nested 3x3 window.
    for (int ky = -1; ky <= 1; ky++)
    {
        for (int kx = -1; kx <= 1; kx++)
        {

            int blur_sum = 0;
            int k_gauss = 0;
            for (int by = -1; by <= 1; by++)
            {
                for (int bx = -1; bx <= 1; bx++)
                {

                    // Fetch pixels from texture object to benefit from 2D cache
                    int px = tex2D<unsigned char>(tex, gx + kx + bx, gy + ky + by);
                    blur_sum += px * c_gauss[k_gauss++];
                }
            }
            float blurred_pixel = blur_sum / 16.0f;

            Gx += blurred_pixel * c_sobelX[k_sobel];
            Gy += blurred_pixel * c_sobelY[k_sobel];
            k_sobel++;
        }
    }

    // Compute gradient magnitude from Sobel responses
    float mag = sqrtf(Gx * Gx + Gy * Gy);

    // Apply threshold and ROI mask to the final magnitude
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

    // Cast votes in accumulator array using trigonometric constants
    for (int t = 0; t < num_theta; ++t)
    {
        float rho = x * c_cos[t] + y * c_sin[t];
        int r_idx = __float2int_rn((rho + max_dist) / rho_step);
        if (r_idx >= 0 && r_idx < num_rho)
            atomicAdd(&d_accumulator[t * num_rho + r_idx], 1);
    }
}

// Helper function to create a texture object
static cudaTextureObject_t create_texture(unsigned char *d_ptr,
                                          int width, int height)
{
    cudaResourceDesc res_desc = {};
    res_desc.resType = cudaResourceTypePitch2D;
    res_desc.res.pitch2D.devPtr = d_ptr;
    res_desc.res.pitch2D.desc = cudaCreateChannelDesc<unsigned char>();
    res_desc.res.pitch2D.width = width;
    res_desc.res.pitch2D.height = height;
    res_desc.res.pitch2D.pitchInBytes = width * sizeof(unsigned char);

    cudaTextureDesc tex_desc = {};
    tex_desc.addressMode[0] = cudaAddressModeClamp;
    tex_desc.addressMode[1] = cudaAddressModeClamp;
    tex_desc.filterMode = cudaFilterModePoint;
    tex_desc.readMode = cudaReadModeElementType;
    tex_desc.normalizedCoords = 0;

    cudaTextureObject_t tex = 0;
    CUDA_CHECK(cudaCreateTextureObject(&tex, &res_desc, &tex_desc, nullptr));
    return tex;
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

    const float rho_step = 1.0f;
    const int num_theta = 180;
    const float max_dist = ceilf(sqrtf((float)(width * width + height * height)));
    const int num_rho = 2 * (int)max_dist;
    const int acc_size = num_theta * num_rho * (int)sizeof(int);

    // Allocate host and device memory
    unsigned char *h_frame_pinned;
    int *h_accumulator;
    CUDA_CHECK(cudaMallocHost(&h_frame_pinned, width * height * 3));
    CUDA_CHECK(cudaMallocHost(&h_accumulator, acc_size));

    uchar3 *d_input[2];
    unsigned char *d_gray[2], *d_filtered[2];
    int *d_accumulator_gpu[2];
    unsigned char *d_mask;

    for (int i = 0; i < 2; i++)
    {
        CUDA_CHECK(cudaMalloc(&d_input[i], width * height * sizeof(uchar3)));
        CUDA_CHECK(cudaMalloc(&d_gray[i], width * height));
        CUDA_CHECK(cudaMalloc(&d_filtered[i], width * height));
        CUDA_CHECK(cudaMalloc(&d_accumulator_gpu[i], acc_size));
    }
    CUDA_CHECK(cudaMalloc(&d_mask, width * height));
    CUDA_CHECK(cudaMemcpy(d_mask, h_mask.data, width * height,
                          cudaMemcpyHostToDevice));

    cudaTextureObject_t tex_gray[2];
    for (int i = 0; i < 2; i++)
        tex_gray[i] = create_texture(d_gray[i], width, height);

    // Initialize CUDA streams
    cudaStream_t stream[2];
    CUDA_CHECK(cudaStreamCreate(&stream[0]));
    CUDA_CHECK(cudaStreamCreate(&stream[1]));

    const int VEC = 4;
    dim3 block_gray(32, 8);
    dim3 grid_gray((width / VEC + 31) / 32,
                   (height + 7) / 8);

    dim3 block_std(TILE_W, TILE_H);
    dim3 grid_std((width + TILE_W - 1) / TILE_W,
                  (height + TILE_H - 1) / TILE_H);

    dim3 block_hough(32, 8);
    dim3 grid_hough((width + 31) / 32,
                    (height + 7) / 8);

    Mat frame, final_out;
    int frame_count = 0;
    int cur = 0;

    if (!cap.read(frame))
    {
        std::cerr << "Empty video.\n";
        return -1;
    }

    // Process video frames in a loop
    while (true)
    {
        frame_count++;
        int buf = cur & 1;

        std::memcpy(h_frame_pinned, frame.data, width * height * 3);
        CUDA_CHECK(cudaMemcpyAsync(d_input[buf], h_frame_pinned,
                                   width * height * sizeof(uchar3),
                                   cudaMemcpyHostToDevice, stream[buf]));
        CUDA_CHECK(cudaMemsetAsync(d_accumulator_gpu[buf], 0,
                                   acc_size, stream[buf]));

        // Execute CUDA kernels asynchronously
        rgb2gray_vec4<<<grid_gray, block_gray, 0, stream[buf]>>>(
            d_input[buf], d_gray[buf], width, height);

        blur_sobel_mask_tex<<<grid_std, block_std, 0, stream[buf]>>>(
            tex_gray[buf], d_mask, d_filtered[buf],
            width, height, 80.0f);

        hough_transform<<<grid_hough, block_hough, 0, stream[buf]>>>(
            d_filtered[buf], d_accumulator_gpu[buf],
            width, height, rho_step, num_rho, num_theta, max_dist);

        CUDA_CHECK(cudaMemcpyAsync(h_accumulator, d_accumulator_gpu[buf],
                                   acc_size, cudaMemcpyDeviceToHost,
                                   stream[buf]));

        Mat next_frame;
        bool has_next = cap.read(next_frame);

        CUDA_CHECK(cudaStreamSynchronize(stream[buf]));

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
            float theta = t * theta_step, rho = (float)r - max_dist;
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

        if (!has_next)
            break;
        if (!next_frame.isContinuous())
            next_frame = next_frame.clone();
        frame = next_frame;
        cur ^= 1;
    }

    // Free allocated memory and destroy resources
    for (int i = 0; i < 2; i++)
        CUDA_CHECK(cudaDestroyTextureObject(tex_gray[i]));

    CUDA_CHECK(cudaStreamDestroy(stream[0]));
    CUDA_CHECK(cudaStreamDestroy(stream[1]));
    for (int i = 0; i < 2; i++)
    {
        CUDA_CHECK(cudaFree(d_input[i]));
        CUDA_CHECK(cudaFree(d_gray[i]));
        CUDA_CHECK(cudaFree(d_filtered[i]));
        CUDA_CHECK(cudaFree(d_accumulator_gpu[i]));
    }
    CUDA_CHECK(cudaFree(d_mask));
    CUDA_CHECK(cudaFreeHost(h_frame_pinned));
    CUDA_CHECK(cudaFreeHost(h_accumulator));
    cap.release();
    out.release();
    std::cout << "Processing Complete.\n";
    return 0;
}
