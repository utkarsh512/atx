/**
 * CS61064 - High Perfomance Parallel Programming
 * OpenMP/MPI - Assignment 1
 * Parallel Image Morphing by Affine Transformations
 * 
 * Author: Utkarsh Patel (18EC35034)
 *
 * Main driver code
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <unordered_map>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <omp.h>

#include "atx.hpp"
#include "utils.hpp"


int main(int argc, char *argv[]) 
{
    if (argc != 6) {
        std::cout << "Use following args:\n" \
                     "- arg1: path to image\n" \
                     "- arg2: path to point list 1\n" \
                     "- arg3: path to point list 2\n" \
                     "- arg4: path to output image\n" \
                     "- arg5: dropout probability";
        return 0;
    }

    /* Read image and list of points */
    cv::Mat image1, image2;
    std::vector<cv::Point2f> points1_all, points2_all;
    atx::utils::lookup_table lookup;
    atx::utils::read_image(argv[1], argv[2], image1, points1_all);
    atx::utils::read_image(argv[1], argv[3], image2, points2_all);

    /* Drop some points as per dropout probability */
    double dropout = std::stod(argv[5]);
    auto retained_indices = atx::utils::random_choice(points1_all.size(), dropout);
    std::vector<cv::Point2f> points1(retained_indices.size()), \
                             points2(retained_indices.size());
    for (std::size_t i = 0; i < retained_indices.size(); i++) {
        auto idx = retained_indices[i];
        points1[i] = points1_all[idx];
        points2[i] = points2_all[idx];
    }
    atx::utils::create_lookup_table(points1, lookup);

    /* Perform triangulation */
    auto subdiv = atx::utils::get_triangulation(image1, points1);

    /* Get triangles */
    std::vector<cv::Vec6f> triangles1, triangles2;
    subdiv.getTriangleList(triangles1);
    std::size_t n_triangles = triangles1.size();
    triangles2.resize(n_triangles);

    /* For mapping triangles from 1st image to triangles of 2nd image */
    std::vector<std::vector<int>> index(n_triangles, std::vector<int>(3));

    /* Affine transform matrices for each triangle pair */
    std::vector<std::vector<std::vector<double>>> affine_transforms(n_triangles);
    std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>> bounds(n_triangles);

#if defined(_OPENMP)
    double start_time = omp_get_wtime();
#else
    auto start_time = std::chrono::steady_clock::now();
#endif 

    #pragma omp parallel default(shared)
    {
        std::size_t i, j;
        int y, x;

        /* Create index to find corresponding triangle pairs 
         * No collapse clause used to increase cache hits per thread
         */
        #pragma omp for schedule(static)
        for (i = 0; i < n_triangles; i++) {
            for (j = 0; j < 3; j++) {
                index[i][j] = lookup[std::make_pair(triangles1[i][j << 1], triangles1[i][j << 1 | 1])];
            }
        }

        /* Reorder triangles so that triangles1[i] and triangles2[i] belong to
         * the same pair
         */
        #pragma omp for schedule(static)
        for (i = 0; i < index.size(); i++) {
            triangles1[i] = cv::Vec6f({points1[index[i][0]].x, points1[index[i][0]].y,
                                       points1[index[i][1]].x, points1[index[i][1]].y,
                                       points1[index[i][2]].x, points1[index[i][2]].y});
            triangles2[i] = cv::Vec6f({points2[index[i][0]].x, points2[index[i][0]].y,
                                       points2[index[i][1]].x, points2[index[i][1]].y,
                                       points2[index[i][2]].x, points2[index[i][2]].y});
        }

        /* Compute affine transformation matrix for each triangle pair */
        #pragma omp for schedule(static)
        for (i = 0; i < n_triangles; i++) {
            affine_transforms[i] = atx::core::get_affine_transform(triangles2[i], triangles1[i]);
            bounds[i] = atx::core::get_triangle_bounds(triangles2[i]);
        }

        /* Here, scheduling is made dynamic as workload in each iteration 
         * is not same. Chunk size is set to 1.
         */
        #pragma omp for schedule(dynamic)
        for (i = 0; i < n_triangles; i++) {
            /* Get bounds for each triangle */
            int ymin = bounds[i].second.first, ymax = bounds[i].second.second;
            int xmin = bounds[i].first.first, xmax = bounds[i].first.second;

            for (y = ymin; y <= ymax; y++) {
                for (x = xmin; x <= xmax; x++) {
                    if (!atx::core::is_inside_triangle(x, y, triangles2[i])) {
                        /* Point lies outside the triangle */
                        continue;
                    }

                    if (x < 0 || x >= image2.cols || y < 0 || y >= image2.rows) {
                        /* Point cannot be accessed in the image */
                        continue;
                    }

                    /* Create a vector V = [Vx, Vy, 1] */
                    std::vector<std::vector<double>> V(3, std::vector<double>(1, 1));
                    V[0][0] = static_cast<double>(x);
                    V[1][0] = static_cast<double>(y);

                    /* Get transformed vector W = MV = [Wx, Wy, 1] */
                    auto W = atx::core::matrix_multiply(affine_transforms[i], V);
                    int tx = static_cast<int>(W[0][0] + 0.5);
                    int ty = static_cast<int>(W[1][0] + 0.5);

                    if (!atx::core::is_inside_triangle(tx, ty, triangles1[i])) {
                        /* Point lies outside the triangle */
                        continue;
                    }

                    if (tx < 0 || tx >= image1.cols || ty < 0 || ty >= image1.rows) {
                        /* Point cannot be accessed in the image */
                        continue;
                    }

                    image2.at<cv::Vec3b>(y, x) = image1.at<cv::Vec3b>(ty, tx);
                }
            }
        }
    }

    double elapsed_time;

#if defined(_OPENMP)
    double stop_time = omp_get_wtime();
    elapsed_time = (stop_time - start_time) * 1e3;
#else
    auto stop_time = std::chrono::steady_clock::now();
    elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>> \
        (stop_time - start_time).count() * 1e3;
#endif

    std::cout << std::fixed << std::setprecision(2) << "Finished parallel "\
        "region in " << elapsed_time << "ms [#points: " << points1.size() \
        << "] [#triangles: " << n_triangles << "]" << std::endl;

    /* Write image to output file*/
    cv::imwrite(argv[4], image2);
    return 0;
}