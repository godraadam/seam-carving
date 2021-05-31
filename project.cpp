#include "stdafx.h"
#include "common.h"
#include <iostream>
#include <numeric>
#include <limits>
#include <cmath>


struct Timer {

	std::chrono::time_point<std::chrono::steady_clock> startTime, endTime;
	std::chrono::duration<float> duration;

	//save time point when instantiated
	Timer() : duration(0.0f) {
		startTime = std::chrono::high_resolution_clock::now();
	}


	//when variable goes out of scope, destructor is called implicitly, print ellapsed time since constructor was called
	~Timer() {
		endTime = std::chrono::high_resolution_clock::now();
		duration = endTime - startTime;
		float ms = duration.count() * 1000.0f;
		std::cout << ms << "ms" << std::endl;
	}
};

using GrayScalePixel = uint8_t;
using GrayScaleImage = cv::Mat_<GrayScalePixel>;
using LabeledImage = cv::Mat_<float>;
using RGBPixel = cv::Point3_<uint8_t>;
using RGBImage = cv::Mat_<RGBPixel>;

using Kernel = cv::Mat_<float>;

struct KernelPair {
	const Kernel horizontal;
	const Kernel vertical;
};

constexpr size_t HISTOGRAM_SIZE = 256;
constexpr int WEAK_EDGE = 128, STRONG_EDGE = 255, NON_EDGE = 0;


auto saturate(int c) -> uint8_t {
	if (c > 255) return 255;
	if (c < 0) return 0;
	return c;
}

//Helper method to check if given coordinates are within the bounds of an image
auto isInside(const GrayScaleImage& img, int row, int col) {
	if (row < 0 || row >= img.rows) return false;
	if (col < 0 || col >= img.cols) return false;
	return true;
}

auto isInside(const cv::Mat_<int>& img, int row, int col) {
	if (row < 0 || row >= img.rows) return false;
	if (col < 0 || col >= img.cols) return false;
	return true;
}

std::vector<int> getHistogram(GrayScaleImage& img, size_t bucket_count) {
	std::vector<int> v;
	v.assign(bucket_count, 0);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			int bucket = int(img(i, j) * bucket_count / 256.0f);
			v[bucket]++;
		}
	}
	return v;
}


std::vector<int> getHistogram(GrayScaleImage& img) {
	return getHistogram(img, HISTOGRAM_SIZE);
}


enum NeighbourhoodType { N4, N8, NP };

auto getNeighbours(GrayScaleImage& img, int x, int y, NeighbourhoodType nType) -> std::vector<std::pair<size_t, size_t>> /*list of (col, row) pairs*/ {
	std::vector<int> x_offsets, y_offsets;
	std::vector<std::pair<size_t, size_t>> neighbours;
	switch (nType) {
	case NeighbourhoodType::N4:
		x_offsets = { -1, 0, 1, 0 };
		y_offsets = { 0, 1, 0, -1 };
		break;
	case NeighbourhoodType::N8:
		x_offsets = { -1, 0, 1, -1, 1, -1, 0, 1 };
		y_offsets = { 1, 1, 1, 0, 0, -1, -1, -1 };
		break;
	case NeighbourhoodType::NP:
		x_offsets = { -1, 0, 1, -1 };
		y_offsets = { -1, -1, -1, 0 };
		break;
	}
	for (int i = 0; i < y_offsets.size(); i++) {
		for (int j = 0; j < x_offsets.size(); j++) {
			int neighbour_x = x + x_offsets[j];
			int neighbour_y = y + y_offsets[i];
			if (isInside(img, neighbour_y, neighbour_x)) {
				neighbours.push_back(std::make_pair(neighbour_x, neighbour_y));
			}
		}
	}
	return neighbours;
}


// The gaussian blur kernel is separable => O(n) runtime instead of O(n^2)
auto getGaussianBlurKernel(size_t radius = 1, float std_dev = 0.0f) -> KernelPair {
	int size = 2 * radius + 1;
	if (std_dev == 0) std_dev = radius / 6.0;
	Kernel horizontal(size, 1), vertical(1, size);
	for (int i = 0; i < size; i++) {
		float val = (1.0f / (sqrt(2 * PI) * std_dev)) * exp2f(-((i - size / 2.0 + 0.5) * (i - size / 2.0 + 0.5)) / 2.0 * std_dev * std_dev);
		horizontal(i, 0) = val;
		vertical(0, i) = val;
	}
	return KernelPair{ horizontal, vertical };
};


// The simple blur kernel is separable => O(n) runtime instead of O(n^2)
auto getSimpleBlurKernel(size_t radius = 1) -> KernelPair {
	uint size = 2 * radius + 1;
	Kernel horizontal(size, 1), vertical(1, size);
	for (int i = 0; i < size; i++) {
		horizontal(i, 0) = 1.0f;
		vertical(0, i) = 1.0f;
	}
	return KernelPair{ horizontal, vertical };
};

auto getSobelKernel() -> KernelPair {
	Kernel kernely(3, 3);
	kernely(0, 0) = -1.0;
	kernely(0, 1) = 0.0f;
	kernely(0, 2) = 1.0f;
	kernely(1, 0) = -2.0f;
	kernely(1, 1) = 0.0f;
	kernely(1, 2) = 2.0f;
	kernely(2, 0) = -1.0f;
	kernely(2, 1) = 0.0f;
	kernely(2, 2) = 1.0f;

	Kernel kernelx(3, 3);
	kernelx(0, 0) = 1.0;
	kernelx(0, 1) = 2.0f;
	kernelx(0, 2) = 1.0f;
	kernelx(1, 0) = 0.0f;
	kernelx(1, 1) = 0.0f;
	kernelx(1, 2) = 0.0f;
	kernelx(2, 0) = -1.0f;
	kernelx(2, 1) = -2.0f;
	kernelx(2, 2) = -1.0f;
	return { kernelx, kernely };
};

auto normalizeFloatImage(const cv::Mat_<float> img) -> GrayScaleImage {
	//normalize magnitude image (bring it to 0-255)
	GrayScaleImage normalizedImage(img.rows, img.cols);
	float max = *(std::max_element(img.begin(), img.end()));
	float min = *(std::min_element(img.begin(), img.end()));
	float diff = max - min;
	float ratio = 255 / diff;
	normalizedImage.forEach([img, ratio, min](GrayScalePixel& p, const int* pos) -> void {
		p = (img(pos[0], pos[1]) - min) * ratio;
		});
	return normalizedImage;
}


auto normalizeImage(const cv::Mat_<float>& img, const Kernel& kernel) -> GrayScaleImage {
	GrayScaleImage result(img.rows, img.cols);

	//if at least one negative value, i.e high pass filter
	if (std::any_of(kernel.begin(), kernel.end(), [](auto it) -> bool {return it < 0; })) {

		float sumOfPositives = 0.0f;
		float sumOfNegatives = 0.0f;

		for (int i = 0; i < kernel.rows; i++) {
			for (int j = 0; j < kernel.cols; j++) {
				if (kernel(i, j) < 0) sumOfNegatives += -kernel(i, j);
				else sumOfPositives += kernel(i, j);
			}
		}

		float S = 1 / (2 * max(sumOfPositives, sumOfNegatives));
		result.forEach([S, img](GrayScalePixel& p, const int* pos)-> void {
			p = int(S * img(pos[0], pos[1]) + 255 / 2);
			});
		return result;
	}
	else {
		const float sum = std::accumulate(kernel.begin(), kernel.end(), 0.0f);
		if (sum != 0) {
			result.forEach([sum, img](GrayScalePixel& p, const int* pos)-> void {
				p = saturate(int(img(pos[0], pos[1]) / sum));
				});
		}
		else {
			result.forEach([sum, img](GrayScalePixel& p, const int* pos)-> void {
				p = int(img(pos[0], pos[1]));
				});
		}
		return result;
	}
}

auto runKernelOnPixel(const GrayScaleImage& img, const Kernel& kernel, int x, int y) -> float {
	float p = 0.0f;
	for (int i = 0; i < kernel.rows; i++) {
		for (int j = 0; j < kernel.cols; j++) {
			int dx = kernel.rows / 2;
			int dy = kernel.cols / 2;
			int row = x + i - dx >= 0 ? (x + i - dx < img.rows ? x + i - dx : x) : x;
			int col = y + j - dy >= 0 ? (y + j - dy < img.cols ? y + j - dy : y) : y;
			p += img(row, col) * (kernel(i, j));
		}
	}
	return p;
}

auto runKernelOnImage(const GrayScaleImage& img, const Kernel& kernel, bool normalize = true) -> cv::Mat_<float> {
	cv::Mat_<float> result(img.rows, img.cols);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			result(i, j) = runKernelOnPixel(img, kernel, i, j);
		}
	}
	if (normalize) return normalizeImage(result, kernel);
	return result;
}

auto runSeparableKernelOnImage(const GrayScaleImage& img, const KernelPair& kernels, bool normalize = true) -> GrayScaleImage {
	return runKernelOnImage(runKernelOnImage(img, kernels.vertical, normalize), kernels.horizontal, normalize);
}

auto gaussianBlur(const GrayScaleImage& img, size_t radius, float std_dev) -> GrayScaleImage {
	KernelPair gaussian = getGaussianBlurKernel(radius, std_dev);
	GrayScaleImage result(img.rows, img.cols);
	return runSeparableKernelOnImage(img, gaussian);
}

auto medianBlur(const GrayScaleImage& img, size_t radius) -> GrayScaleImage {
	KernelPair meanFilter = getSimpleBlurKernel(radius);
	GrayScaleImage result(img.rows, img.cols);
	return runSeparableKernelOnImage(img, meanFilter);
}


auto edgeDetectCanny(const GrayScaleImage& img, float p, float k) -> GrayScaleImage {
	//blur to reduce noise
	GrayScaleImage blurredImage = gaussianBlur(img, 2, 0.5);

	//run the horizontal and vertical kernels on the image
	KernelPair sobelKernel = getSobelKernel();
	auto horizontalGradient = runKernelOnImage(blurredImage, sobelKernel.horizontal, false);
	auto verticalGradient = runKernelOnImage(blurredImage, sobelKernel.vertical, false);

	//calculate the magnitude and direction of the gradient
	cv::Mat_<float> magnitudeImage(img.rows, img.cols);
	cv::Mat_<float> directionImage(img.rows, img.cols);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			magnitudeImage(i, j) = sqrtf(horizontalGradient(i, j) * horizontalGradient(i, j) + verticalGradient(i, j) * verticalGradient(i, j));
			directionImage(i, j) = atan2f(horizontalGradient(i, j), verticalGradient(i, j));
		}
	}

	// non maxima surpression
	std::pair<int, int> offsets[] = { {1, 0}, {1, -1}, {0, -1}, {-1, -1} };

	GrayScaleImage normalizedMagnitudeImage = normalizeFloatImage(magnitudeImage);
	GrayScaleImage intermediate = normalizedMagnitudeImage.clone();
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			//map from angle of gradient to an octant 0...3
			double alpha = directionImage(i, j);
			alpha = alpha < 0 ? alpha + 2 * PI : alpha;
			int octant = (int)floor(4 * alpha / PI + 0.5) % 4;
			auto& [dx, dy] = offsets[octant];
			if (isInside(normalizedMagnitudeImage, i + dy, j + dx)) {
				if (normalizedMagnitudeImage(i, j) <= normalizedMagnitudeImage(i + dy, j + dx)) {
					intermediate(i, j) = 0x00;
				}
			}
			if (isInside(normalizedMagnitudeImage, i - dy, j - dx)) {
				if (normalizedMagnitudeImage(i, j) <= normalizedMagnitudeImage(i - dy, j - dx)) {
					intermediate(i, j) = 0x00;
				}
			}
		}
	}

	////adaptive tresholding
	//auto histogram = getHistogram(intermediate);
	//int noNonEdgePixels = (1.0f - p) * (intermediate.rows * intermediate.cols - histogram[0]);

	//int sum = 0, treshold = 1;
	//while (sum < noNonEdgePixels && treshold < histogram.size()) sum += histogram[treshold++];

	////edge extension
	//{
	//	int low_treshold = k * treshold;

	//	intermediate.forEach([treshold, low_treshold](GrayScalePixel& p, const int* pos) -> void {
	//		p = p > treshold ? STRONG_EDGE : (p < low_treshold ? NON_EDGE : WEAK_EDGE);
	//		});
	//}
	//{
	//	for (int i = 0; i < intermediate.rows; i++) {
	//		for (int j = 0; j < intermediate.cols; j++) {
	//			if (intermediate(i, j) == STRONG_EDGE) {
	//				std::queue<std::pair<int, int>> q;
	//				q.push(std::make_pair(j, i));
	//				while (!q.empty()) {
	//					auto& [x, y] = q.front();
	//					q.pop();
	//					auto neighbours = getNeighbours(intermediate, x, y, N8);
	//					for (auto& [nx, ny] : neighbours) {
	//						if (intermediate(ny, nx) == WEAK_EDGE) {
	//							q.push(std::make_pair(nx, ny));
	//							intermediate(ny, nx) = STRONG_EDGE;
	//						}
	//					}
	//				}
	//			}
	//		}
	//	}
	//	intermediate.forEach([](GrayScalePixel& p, const int* pos) -> void {
	//		p = p == WEAK_EDGE ? NON_EDGE : p;
	//		});
	//}

	return intermediate;

}

auto getMinimumEnergyPath(const GrayScaleImage& img) -> std::vector<int> {
	cv::Mat_<int> accumulativePaths(img.rows, img.cols);
	cv::Mat_<int> directions(img.rows, img.cols);

	//copy last row
	for (int i = 0; i < img.cols; i++) {
		accumulativePaths(img.rows - 1, i) = (int)img(img.rows - 1, i);
	}
	for (int i = img.rows - 2; i >= 0; i--) {
		for (int j = 0; j < img.cols; j++) {
			int min = INT_MAX;
			int dir = 0;
			for (int dx = -1; dx <= 1; dx++) {
				if (j + dx < 0 || j + dx >= img.cols)
					continue;
				if (accumulativePaths(i + 1, j + dx) < min) {
					min = accumulativePaths(i + 1, j + dx);
					dir = dx;
				}
			}
			accumulativePaths(i, j) = (int)img(i, j) + min;
			directions(i, j) = dir;
		}
	}
	/*GrayScaleImage normalized;
	cv::normalize(accumulativePaths, normalized, 0, 1, cv::NORM_MINMAX);*/
	//find best path from first row of pixels
	std::vector<int> path;
	int min = INT_MAX;
	int index = 0;
	for (int i = 0; i < img.cols; i++) {
		if (accumulativePaths(0, i) < min) {
			min = accumulativePaths(0, i);
			index = i;
			
		}
	}

	//rebuild path
	path.push_back(index);
	for (int i = 0; i < img.rows; i++) {
		int current_x = path.back();
		int current_y = i;
		int dir = directions(current_y, current_x);
		int new_x = current_x + dir;
		path.push_back(new_x);
	}
	return path;
}

auto removeSeam(const RGBImage& img, std::vector<int>& seam) -> RGBImage {
	RGBImage result(img.rows, img.cols - 1);
	for (int i = 0; i < img.rows; i++) {
		int shift = 0;
		for (int j = 0; j < img.cols; j++) {
			if (j == seam[i]) {
				shift = -1;
				continue;
			}
			result(i, j + shift) = img(i, j);
		}
	}
	return result;
}

auto removeSeam(const GrayScaleImage& img, std::vector<int>& seam) -> GrayScaleImage {
	GrayScaleImage result(img.rows, img.cols - 1);
	for (int i = 0; i < img.rows; i++) {
		int shift = 0;
		for (int j = 0; j < img.cols; j++) {
			if (j == seam[i]) {
				shift = -1;
				continue;
			}
			result(i, j + shift) = img(i, j);
		}
	}
	return result;
}


auto paintCarve(const RGBImage& img, const std::vector<int>& seam) -> RGBImage {
	RGBImage result = img.clone();
	for (int i = 0; i < img.rows; i++) {
		result(i, seam[i]) = { 0xff, 0x00, 0x00 };
	}
	return result;
}

auto demo(RGBImage& img, size_t nrOfPixelsToRemove) {
	assert(nrOfPixelsToRemove < img.cols);
	GrayScaleImage grayScale(img.rows, img.cols);
	cv::cvtColor(img, grayScale, cv::COLOR_BGR2GRAY);
	GrayScaleImage edges;
	{
		Timer t;
		edges = edgeDetectCanny(grayScale, 0.1, 0.4);
		std::cout << "Found edges in ";
	}
	for (int i = 0; i < nrOfPixelsToRemove; i++) {
		std::vector<int> seam;
		{
			Timer t;
			seam = getMinimumEnergyPath(edges);
			std::cout << "Found lowest energy path in ";
		}

		edges = removeSeam(edges, seam);
		img = removeSeam(img, seam);
		
		std::cout << std::endl;
		imshow("edges", edges);
		imshow("resized", paintCarve(img, seam));
		cv::waitKey(1);
	}
}



//test code
auto main() -> int {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		RGBImage src = cv::imread(fname);
		std::cout << "enter resize ratio!\n";
		float ratio;
		std::cin >> ratio;
		assert(ratio < 1.0f && ratio >= 0.0f);
		imshow("original", src);
		demo(src, src.cols * ratio);
	}
	return 0;
}