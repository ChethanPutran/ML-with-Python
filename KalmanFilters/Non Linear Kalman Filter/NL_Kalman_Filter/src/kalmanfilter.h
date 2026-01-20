#ifndef INCLUDE_NL_KALMAN_H
#define INCLUDE_NL_KALMAN_H
#include <vector>
#include <eigen3/Eigen/Dense>
#include "car.h"
#include "sensors.h"
#include "beacons.h"

class KalmanFilter
{
public:
    KalmanFilter();
    Eigen::VectorXd get_state() const;
    Eigen::MatrixXd get_covariance() const;
    void set_state(const Eigen::VectorXd &state);
    void set_covariance(const Eigen::MatrixXd &covariance);
    Vehicle_State get_vehicle_state();
    Eigen::Matrix2d get_vehicle_state_position_covariance();
    void prediction_step(double dt);
    void prediction_step(Gyro_Measurement gyro, double dt);
    void handle_lidar_measurements(const std::vector<Lidar_Measurement> &lidar_measurement, const Beacon_Map &map);
    void handle_lidar_measurements(Lidar_Measurement measurement, const Beacon_Map &map);
    void handle_gps_measurements(GPS_Measurement measurement);

private:
    bool is_initialized;
    Eigen::VectorXd m_state;
    Eigen::MatrixXd m_covariance;
};

#endif // INCLUDE_NL_KALMAN_H