
#include "utils.h"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>
#include <eigen3/Eigen/Core>

double wrap_angle(double angle)
{
    angle = fmod(angle, (2.0 * M_PI));

    if (angle <= -M_PI)
    {
        angle += (2.0 * M_PI);
    }
    else if (angle > M_PI)
    {
        angle -= (2.0 * M_PI);
    }
    return angle;
}

// Function to calculate mean
double calculate_mean(std::vector<double> &dataset)
{
    if (dataset.empty())
    {
        return NAN;
    }
    double sum = std::accumulate(std::begin(dataset), std::end(dataset), 0.0);
    double mean = sum / dataset.size();

    return mean;
}

// Function to calculate Root Mean Squared Errors
double calculate_rmse(std::vector<double> &dataset)
{
    double sum_errors = 0.0;

    if (dataset.empty())
    {
        return 0.0;
    }
    std::for_each(std::begin(dataset), std::end(dataset), [&](const double d)
                  { sum_errors += (d * d); });
    double rmse = sqrt(sum_errors / dataset.size());

    return rmse;
}

// Function to generate ellipse points
std::vector<Vector2> generate_ellipse(double x, double y, double sigma_xx, double sigma_yy, double sigma_xy, int num_points = 50)
{
    // Creating covariance matrix
    Eigen::Matrix2d positional_covariance;
    positional_covariance << sigma_xx, sigma_xy, sigma_xy, sigma_yy;

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(positional_covariance, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::Matrix2d D = svd.matrixU() * Eigen::VectorXd(3.0 * svd.singularValues().array().sqrt()).asDiagonal();

    // Creating 50 points between [0,2*pi]
    auto theta = Eigen::ArrayXd::LinSpaced(num_points, 0, 2 * M_PI);

    // Creating a array of size 2*50
    Eigen::ArrayXd theta_array(2, num_points);

    // Storing cosines of theta in 0th row
    theta_array.row(0) = theta.cos();

    // Storing cosines of theta in 1st row
    theta_array.row(1) = theta.sin();

    // Multiplying D with theta array
    Eigen::MatrixXd A = D * Eigen::MatrixXd(theta_array);
    std::vector<Vector2> shape_body;

    // Creating vector of Vector2 with x  as A[0,i] and y as A[1,i]
    for (int i = 0; i < A.cols(); ++i)
    {
        shape_body.push_back(Vector2(A(0, i), A(1, i)));
    }
    std::vector<Vector2> ellipse = offset_points(shape_body, Vector2(x, y));

    return ellipse;
}

// Function generate circle points
std::vector<Vector2> generate_circle(double x, double y, double radius, int num_points = 50)
{

    // Generating points between [0,2pi]
    auto theta = Eigen::ArrayXd::LinSpaced(num_points, 0, 2 * M_PI);

    // Vector to hold generated x and y coordinates
    std::vector<Vector2> circle;

    for (int i = 0; i < theta.size(); ++i)
    {
        // Generating x = x + rcos(theta(i))
        //  y = y + rsin(theta(i))
        circle.push_back(Vector2(x + radius * cos(theta(i)), y + radius * sin(theta(i))));
    }

    return circle;
}
