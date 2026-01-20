#include "kalmanfilter.h"
#include "utils.h"

const bool INITIALIZE_ON_FIRST_PREDICTION = 0.0;
const double INITIAL_POSITION_STD = 0.0;
const double INITIAL_VELOCITY_STD = 15.0;
const double ACCELERATION_STD = 0.1;
const double GPS_POSITION_STD = 3.0;

Eigen::VectorXd KalmanFilter::get_state() const
{
    return this->m_state;
}
Eigen::MatrixXd KalmanFilter::get_covariance() const
{
    return this->m_covariance;
}
void KalmanFilter::set_state(const Eigen::VectorXd &state)
{
    this->m_state = state;
    this->is_initialized = true;
}
void KalmanFilter::set_covariance(const Eigen::MatrixXd &covariance)
{
    this->m_state = covariance;
}
Vehicle_State KalmanFilter::get_vehicle_state()
{
    if (this->is_initialized)
    {
        Eigen::VectorXd state = this->get_state();
        double psi = std::atan2(state[3], state[2]);
        double velocity = std::sqrt(state[2] * state[2] + state[3] * state[3]);
        return Vehicle_State(state[0], state[1], psi, velocity);
    }
    return Vehicle_State();
}
Eigen::Matrix2d KalmanFilter::get_vehicle_state_position_covariance()
{
    Eigen::Matrix2d positional_cov = Eigen::Matrix2d::Zero();
    Eigen::MatrixXd covariance = get_covariance();
    if (this->is_initialized && covariance.size() != 0)
    {
        positional_cov << covariance(0, 0), covariance(0, 1), covariance(1, 0), covariance(1, 1);
    }
    return positional_cov;
}
void KalmanFilter::prediction_step(double dt)
{
    if (!this->is_initialized && INITIALIZE_ON_FIRST_PREDICTION)
    {
        Eigen::VectorXd state = Eigen::Vector4d::Zero();
        Eigen::MatrixXd covariance = Eigen::Matrix4d::Zero();

        // Assume the initial position is (X,Y) = (0,0) m
        // Assume the initial velocity is 5 m/s at 45 degrees (VX,VY) = (5*cos(45deg),5*sin(45deg)) m/s
        state << 0, 0, 5.0 * cos(M_PI / 4), 5.0 * sin(M_PI / 4);
        this->set_state(state);
        this->set_covariance(covariance);
    }
    if (this->is_initialized)
    {
        Eigen::VectorXd state = this->get_state();
        Eigen::MatrixXd covariance = this->get_covariance();

        this->set_state(state);
        this->set_covariance(covariance);
    }
}
void KalmanFilter::prediction_step(Gyro_Measurement gyro, double dt)
{
    this->prediction_step(dt);
}
void KalmanFilter::handle_lidar_measurements(const std::vector<Lidar_Measurement> &lidar_measurement, const Beacon_Map &map) {}
void KalmanFilter::handle_lidar_measurements(Lidar_Measurement measurement, const Beacon_Map &map) {}
void KalmanFilter::handle_gps_measurements(GPS_Measurement measurement) {}
