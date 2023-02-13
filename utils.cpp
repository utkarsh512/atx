/**
 * CS61064 - High Perfomance Parallel Programming
 * OpenMP/MPI - Assignment 1
 * Parallel Image morphing by affine transformations
 * 
 * Author: Utkarsh Patel (18EC35034)
 *
 * Implementation for utilities
 */

#include <vector>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "utils.hpp"


/**
 * @brief Read an image along with a set of points for creating triangles
 * 
 * @param path_to_image path to image file
 * @param path_to_points path to file containing points
 * @param img input image
 * @param points vector of points
 */
void atx::utils::read_image(char *path_to_image, char *path_to_points,
                            cv::Mat& img, std::vector<cv::Point2f>& points)
{
    img = cv::imread(path_to_image);
    std::ifstream ifs(path_to_points);
    double x, y;
    while (ifs >> x >> y) {
        points.emplace_back(x, y);
    }
}


/**
 * @brief Create lookup table for fast searching and indexing
 * 
 * @param points vector of points
 * @param lookup loopup table
 */
void atx::utils::create_lookup_table(std::vector<cv::Point2f>& points, 
                                     lookup_table& lookup)
{
    for (std::size_t i = 0; i < points.size(); i++) {
        lookup[std::make_pair(points[i].x, points[i].y)] = i;
    }
}


/**
 * @brief Draw triangles over the image
 * 
 * @param img image
 * @param subdiv Subdiv2D object containing the triangles
 * @param delaunay_color pen color
 */
void atx::utils::draw_triangulation(cv::Mat& img, cv::Subdiv2D& subdiv, 
                                    const cv::Scalar& delaunay_color)
{
    std::vector<cv::Vec6f> triangle_list;
    subdiv.getTriangleList(triangle_list);
    std::vector<cv::Point> pt(3);
    cv::Size size = img.size();
    cv::Rect rect(0, 0, size.width, size.height);

    for (auto&& t : triangle_list) {
        pt[0] = cv::Point(cvRound(t[0]), cvRound(t[1]));
        pt[1] = cv::Point(cvRound(t[2]), cvRound(t[3]));
        pt[2] = cv::Point(cvRound(t[4]), cvRound(t[5]));

        if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2])) {
            cv::line(img, pt[0], pt[1], delaunay_color, 1, cv::LINE_AA, 0);
            cv::line(img, pt[1], pt[2], delaunay_color, 1, cv::LINE_AA, 0);
            cv::line(img, pt[2], pt[0], delaunay_color, 1, cv::LINE_AA, 0);
        }
    }
}


/**
 * @brief Get the triangulation using Delaunay's Algorithm
 * 
 * @param img input image
 * @param points points to consider for Delaunay's triangulation
 * @return cv::Subdiv2D 
 */
cv::Subdiv2D atx::utils::get_triangulation(cv::Mat& img, 
                                           std::vector<cv::Point2f>& points)
{
    cv::Size size = img.size();
    cv::Rect rect(0, 0, size.width, size.height);
    cv::Subdiv2D subdiv(rect);
    for (const auto& point : points) {
        subdiv.insert(point);
    }
    return subdiv;
}