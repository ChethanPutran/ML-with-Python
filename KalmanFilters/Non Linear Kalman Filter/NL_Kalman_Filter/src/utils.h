#ifndef NL_KALMAN_UTILS_H
#define NL_KALMAN_UTILS_H
#include <vector>
#include "display.h"

double wrap_angle(double angle);
double calculate_mean(std::vector<double> &dataset);
double calculate_rmse(std::vector<double> &dataset);
std::vector<Vector2> generate_ellipse(double x, double y, double sigma_xx, double sigma_yy, double sigma_xy, int num_points = 50);
std::vector<Vector2> generate_circle(double x, double y, double radius, int num_points = 50);

#define NL_KALMAN_UTILS_H
#endif //