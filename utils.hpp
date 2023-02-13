/**
 * CS61064 - High Perfomance Parallel Programming
 * OpenMP/MPI - Assignment 1
 * Parallel Image morphing by affine transformations
 * 
 * Author: Utkarsh Patel (18EC35034)
 *
 * Interface for utilities
 */

#ifndef ATX_UTILS_H
#define ATX_UTILS_H

#include <vector>
#include <chrono>
#include <random>
#include <unordered_map>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace atx {

namespace utils {

static std::mt19937_64 random_number_generator(std::chrono::high_resolution_clock::now().time_since_epoch().count());

template <typename T>
inline T random_number(T low, T high) 
{ 
  std::uniform_real_distribution<T> distribution(low, high); 
  return distribution(random_number_generator); 
}

/**
 * @brief Drop some indices based on dropout probability
 * 
 * @param N Total range
 * @param dropout Dropout probability
 * @return Retained indices
 */
inline std::vector<std::size_t> random_choice(std::size_t N, double dropout)
{
    std::vector<std::size_t> choice;
    for (std::size_t i = 0; i < N; i++) {
        double cur = atx::utils::random_number<double>(0, 1);
        if (cur < dropout) {
            choice.push_back(i);   
        }
    }
    return choice;
}

/**
 * @brief Custom hash function for fast look-up
 * 
 */
struct custom_hash {
    static uint64_t splitmix64(uint64_t x) {
        x += 0x9e3779b97f4a7c15;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
        x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
        return x ^ (x >> 31);
    }
 
    size_t operator()(uint64_t x) const {
        static const uint64_t FIXED_RANDOM = std::chrono::steady_clock::now().time_since_epoch().count();
        return splitmix64(x + FIXED_RANDOM);
    }

    template<class L, class R>
    size_t operator()(std::pair<L, R> const& Y) const{
        static const uint64_t FIXED_RANDOM = std::chrono::steady_clock::now().time_since_epoch().count();
        return splitmix64(Y.first * 31ull + Y.second + FIXED_RANDOM);
    }
};

using lookup_table = std::unordered_map<std::pair<float, float>, std::size_t, atx::utils::custom_hash>;

void read_image(char *, char *, cv::Mat&, std::vector<cv::Point2f>&);

void create_lookup_table(std::vector<cv::Point2f>&, lookup_table&);

/**
 * @brief Draw a point on the image
 * 
 * @param img input image
 * @param p point to be drawn
 * @param color color
 */
inline void draw_point(cv::Mat& img, const cv::Point2f& p, 
                       const cv::Scalar& color)
{
    cv::circle(img, p, 2, color, cv::FILLED, cv::LINE_AA, 0);
}

void draw_triangulation(cv::Mat&, cv::Subdiv2D&, const cv::Scalar&);

cv::Subdiv2D get_triangulation(cv::Mat&, std::vector<cv::Point2f>&);

} /* utils */

} /* atx */

#endif /* ATX_UTILS_H */
