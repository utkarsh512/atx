# CS61064 - High Perfomance Parallel Programming
# OpenMP/MPI - Assignment 1
# Parallel Image morphing by affine transformations
# 
# Author: Utkarsh Patel (18EC35034)
#
# Generate morphed image from given two set of points
#
# To be run as:
# $ bash benchmark.sh dropout
# 
# where, dropout is dropout probability

./main ./data/man1.jpg ./data/points1.txt ./data/points2.txt ./result/man1-points1-points2.jpg ${1}
./main ./data/man1.jpg ./data/points2.txt ./data/points1.txt ./result/man1-points2-points1.jpg ${1}
./main ./data/man2.jpg ./data/points1.txt ./data/points2.txt ./result/man2-points1-points2.jpg ${1}
./main ./data/man2.jpg ./data/points2.txt ./data/points1.txt ./result/man2-points2-points1.jpg ${1}
./main ./data/villa.jpg ./data/points1.txt ./data/points2.txt ./result/villa-points1-points2.jpg ${1}
./main ./data/villa.jpg ./data/points2.txt ./data/points1.txt ./result/villa-points2-points1.jpg ${1}
./main ./data/game_screen.jpg ./data/points1.txt ./data/points2.txt ./result/game_screen-points1-points2.jpg ${1}
./main ./data/game_screen.jpg ./data/points2.txt ./data/points1.txt ./result/game_screen-points2-points1.jpg ${1}
./main ./data/cover.jpg ./data/points1.txt ./data/points2.txt ./result/cover-points1-points2.jpg ${1}
./main ./data/cover.jpg ./data/points2.txt ./data/points1.txt ./result/cover-points2-points1.jpg ${1}