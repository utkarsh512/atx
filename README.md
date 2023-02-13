# Parallel Image Morphing by Affine Transformations

### CS61064 - High Perfomance Parallel Programming - OpenMP/MPI Assignment 1

**Utkarsh Patel (18EC35034)**

## Benchmarking Performance

The program is run for different number of points considered in triangulation step 
(hence for different number of triangles). We benchmark the time elapsed while 
executing the parallelizable section (the SECTION here onwards.)

![number of points vs. elapsed time](./result/points-vs-elapsedtime.png)
![number of triangles vs. elapsed time](./result/triangles-vs-elapsedtime.png)

We observe that using OpenMP, performance is increased by nearly **3x**.

## Results

| Original image | Morphed image #1 | Morphed image #2 |
| -------------- | ---------------- | ---------------- |
| ![./data/man1.jpg](./data/man1.jpg) | ![./result/man1-points1-points2.jpg](./result/man1-points1-points2.jpg) | ![./result/man1-points2-points1.jpg](./result/man1-points2-points1.jpg) |
| ![./data/man2.jpg](./data/man2.jpg) | ![./result/man2-points1-points2.jpg](./result/man2-points1-points2.jpg) | ![./result/man2-points2-points1.jpg](./result/man2-points2-points1.jpg) |
| ![./data/villa.jpg](./data/villa.jpg) | ![./result/villa-points1-points2.jpg](./result/villa-points1-points2.jpg) | ![./result/villa-points2-points1.jpg](./result/villa-points2-points1.jpg) |
| ![./data/cover.jpg](./data/cover.jpg) | ![./result/cover-points1-points2.jpg](./result/cover-points1-points2.jpg) | ![./result/cover-points2-points1.jpg](./result/cover-points2-points1.jpg) |
! ![./data/game_screen.jpg](./data/game_screen.jpg) | ![./result/game_screen-points1-points2.jpg](./result/game_screen-points1-points2.jpg) | ![](./result/game_screen-points2-points1.jpg)

## Build

To build this repository
```shell
$ make
```

## Generate results

To view results from sample images
```shell
$ bash benchmark.sh DROPOUT_PROB
```
where `DROPOUT_PROB` is the probability of dropping some of the points.