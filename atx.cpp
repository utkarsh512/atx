/**
 * CS61064 - High Perfomance Parallel Programming
 * OpenMP/MPI - Assignment 1
 * Parallel Image morphing by affine transformations
 * 
 * Author: Utkarsh Patel (18EC35034)
 *
 * Implementation for affine transformation
 */

#include <vector>
#include <cassert>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "atx.hpp"

constexpr double EPS = 1e-6;


/**
 * @brief Multiply two matrices
 * 
 * @details This function is not parallelized as it is only used for 
 * multiplying 3x3 matrices and 3x1 vectors.
 */
std::vector<std::vector<double>> 
atx::core::matrix_multiply(const std::vector<std::vector<double>>& matrix_1,
                           const std::vector<std::vector<double>>& matrix_2)
{
    std::size_t n = matrix_1.size();
    std::size_t k = matrix_1[0].size();
    std::size_t k_ = matrix_2.size();
    std::size_t m = matrix_2[0].size();
    assert(k == k_);

    std::vector<std::vector<double>> res(n, std::vector<double>(m));
    for (std::size_t i = 0; i < n; i++) {
        for (std::size_t j = 0; j < m; j++) {
            double sum = 0;
            for (std::size_t p = 0; p < k; p++) {
                sum += matrix_1[i][p] * matrix_2[p][j];
            }
            res[i][j] = sum;
        }
    }    
    
    return res;
}


/**
 * @brief Get inverse of a non-singular matrix
 */
std::vector<std::vector<double>> 
atx::core::get_matrix_inverse(const std::vector<std::vector<double>>& matrix) 
{
    std::size_t n = matrix.size();    assert(n == 3);
    std::size_t m = matrix[0].size(); assert(m == 3);

    double determinant = 0;
    for (int i = 0; i < 3; i++) {
        determinant += matrix[0][i] * (matrix[1][(i+1)%3] * matrix[2][(i+2)%3] \
            - matrix[1][(i+2)%3] * matrix[2][(i+1)%3]);
    }
    assert(determinant != 0);

    std::vector<std::vector<double>> inverse(n, std::vector<double>(m));
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            inverse[i][j] = ((matrix[(j+1)%3][(i+1)%3] * matrix[(j+2)%3][(i+2)%3]) \
                - (matrix[(j+1)%3][(i+2)%3] * matrix[(j+2)%3][(i+1)%3])) \
                / determinant;
        }
    }
    
    return inverse;
}


/**
 * @brief Get the affine transform for the triangle pair
 * 
 * @param source_triangle Source triangle
 * @param dest_triangle Destination triangle
 * @return std::vector<std::vector<double>> 
 */
std::vector<std::vector<double>>
atx::core::get_affine_transform(const cv::Vec6f& source_triangle, 
                                const cv::Vec6f& dest_triangle)
{
    std::vector<std::vector<double>> V(3, std::vector<double>(3, 1)), 
                                     W(3, std::vector<double>(3, 1));
    for (int i = 0; i < 3; i++) {
        V[0][i] = source_triangle[i * 2];
        V[1][i] = source_triangle[i * 2 + 1];
        W[0][i] = dest_triangle[i * 2];
        W[1][i] = dest_triangle[i * 2 + 1];
    }
    
    auto V_inv = get_matrix_inverse(V);
    return matrix_multiply(W, V_inv);
}


/**
 * @brief Get ((x_min, x_max), (y_min, y_max)) for given triangle
 */
std::pair<std::pair<int, int>, std::pair<int, int>>
atx::core::get_triangle_bounds(const cv::Vec6f& triangle)
{
    int x_max = INT_MIN, x_min = INT_MAX, y_max = INT_MIN, y_min = INT_MAX;
    for (int i = 0; i < 3; i++) {
        x_max = std::max(x_max, static_cast<int>(triangle[i * 2]));
        x_min = std::min(x_min, static_cast<int>(triangle[i * 2]));
        y_max = std::max(y_max, static_cast<int>(triangle[i * 2 + 1]));
        y_min = std::min(y_min, static_cast<int>(triangle[i * 2 + 1]));
    }
    return std::make_pair(std::make_pair(x_min, x_max), std::make_pair(y_min, y_max));
}


double atx::core::triangle_area(const cv::Vec6f& triangle) {
    double signed_area = triangle[0] * (triangle[3] - triangle[5]) \
                       + triangle[2] * (triangle[5] - triangle[1]) \
                       + triangle[4] * (triangle[1] - triangle[3]);
    return std::abs(signed_area);
}


/**
 * @brief Check whether a point (x, y) is inside a triangle
 * 
 * @param x Point's x-coordinate
 * @param y Point's y-coordinate
 * @param triangle triangle in the check
 */
bool atx::core::is_inside_triangle(int x, int y, const cv::Vec6f& triangle)
{
    auto triangle1 = cv::Vec6f(triangle);
    auto triangle2 = cv::Vec6f(triangle);
    auto triangle3 = cv::Vec6f(triangle);

    triangle1[0] = static_cast<float>(x);
    triangle1[1] = static_cast<float>(y);
    triangle2[2] = static_cast<float>(x);
    triangle2[3] = static_cast<float>(y);
    triangle3[4] = static_cast<float>(x);
    triangle3[5] = static_cast<float>(y);

    auto area = triangle_area(triangle);
    auto area1 = triangle_area(triangle1);
    auto area2 = triangle_area(triangle2);
    auto area3 = triangle_area(triangle3);
    auto delta = std::abs(area1 + area2 + area3 - area);
    return delta < EPS;
}