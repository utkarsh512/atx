/**
 * CS61064 - High Perfomance Parallel Programming
 * OpenMP/MPI - Assignment 1
 * Parallel Image morphing by affine transformations
 * 
 * Author: Utkarsh Patel (18EC35034)
 *
 * Interface for affine transformation
 */

#ifndef ATX_CORE_H
#define ATX_CORE_H

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace atx {

namespace core {

std::vector<std::vector<double>> 
matrix_multiply(const std::vector<std::vector<double>>&,
                const std::vector<std::vector<double>>&);

std::vector<std::vector<double>> 
get_matrix_inverse(const std::vector<std::vector<double>>&);

std::vector<std::vector<double>> 
get_affine_transform(const cv::Vec6f&, const cv::Vec6f&);

std::pair<std::pair<int, int>, std::pair<int, int>>
get_triangle_bounds(const cv::Vec6f&);

double triangle_area(const cv::Vec6f&);

bool is_inside_triangle(int, int, const cv::Vec6f&);

} /* core */

} /* atx */

#endif /* ATX_CORE_H */