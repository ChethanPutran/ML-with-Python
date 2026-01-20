#ifndef INCLUDE_NL_KALAMAN_CAR_H
#define INCLUDE_NL_KALAMAN_CAR_H

#include <queue>
#include <cmath>
#include "display.h"
#include "utils.h"

struct Vehicle_State
{
    double x, y, psi, velocity, yaw_rate, steering;
    Vehicle_State()
    {
        this->x = 0.0;
        this->y = 0.0;
        this->psi = 0.0;
        this->velocity = 0.0;
        this->yaw_rate = 0.0;
        this->steering = 0.0;
    }
    Vehicle_State(double x, double y, double psi, double velocity)
    {
        this->x = x;
        this->y = y;
        this->psi = psi;
        this->velocity = velocity;
        this->yaw_rate = 0.0;
        this->steering = 0.0;
    }
    Vehicle_State(double x, double y, double psi, double velocity, double yaw_rate, double steering)
    {
        this->x = x;
        this->y = y;
        this->psi = psi;
        this->velocity = velocity;
        this->yaw_rate = yaw_rate;
        this->steering = steering;
    }
};

class Motion
{
public:
    Motion()
    {
        this->m_velocity_b = 0.0;
        this->m_steering = 0.0;
    }
    virtual void start(double time, Vehicle_State state)
    {
        this->m_start_time = time;
        this->m_start_state = state;
    }
    virtual void end(double time, double dt, Vehicle_State state)
    {
    }
    virtual bool update(double time, double dt, Vehicle_State state) {}
    virtual double get_velocity()
    {
        return this->m_velocity_b;
    }
    virtual double get_steering()
    {
        return this->m_steering;
    }

protected:
    Vehicle_State m_start_state;
    double m_steering;
    double m_velocity_b;
    double m_start_time;
};
class Motion_Straight : public Motion
{
public:
    Motion_Straight(double time, double velocity)
    {
        this->m_time = time;
        this->m_velocity = velocity;
    }

    bool update(double time, double dt, Vehicle_State state)
    {
        this->m_velocity_b = this->m_velocity;
        this->m_steering = 0.0;
        return time > (this->m_start_time + this->m_time);
    }

private:
    double m_velocity;
    double m_time;
};

class Motion_Turn_To : public Motion
{
public:
    Motion_Turn_To(double heading, double velocity)
    {
        this->m_heading = heading;
        this->m_velocity = velocity;
    }

    bool update(double time, double dt, Vehicle_State state)
    {
        this->m_velocity_b = this->m_velocity;
        double angle_error = wrap_angle(this->m_heading - state.psi);
        this->m_steering = angle_error * (std::signbit(state.velocity) ? -1.0 : 1.0);
        return std::fabs(angle_error) < 0.001;
    }

private:
    double m_heading;
    double m_velocity;
};

class Motion_Move_To : public Motion
{
public:
    Motion_Move_To(double x, double y, double velocity)
    {
        this->m_x = x;
        this->m_y = y;
        this->m_velocity = velocity;
    }

    bool update(double time, double dt, Vehicle_State state)
    {
        this->m_velocity_b = this->m_velocity;
        double delta_x = this->m_x - state.x;
        double delta_y = this->m_y - state.y;
        double range = sqrt(delta_x * delta_x + delta_y * delta_y);
        double angle = atan2(delta_y, delta_x);
        double psi = wrap_angle(state.psi - (std::signbit(state.velocity) ? M_PI : 0.0));
        double angle_error = wrap_angle(angle - psi);
        this->m_steering = angle_error * (std::signbit(state.velocity) ? -1.0 : 1.0);
        return (range < 5.0);
    }

private:
    double m_x;
    double m_y;
    double m_velocity;
};

class Bicycle_Motion
{
public:
    // Starting point is the origin
    Bicycle_Motion()
    {
        this->m_initial_state = Vehicle_State(0, 0, 0, 0);
        this->m_wheel_base = 4.0;
        this->m_max_velocity = 28.0;
        this->m_max_acceleration = 2.0;
        this->m_max_steering = 0.8;
        this->reset();
    }
    // Starting point is the user given point
    Bicycle_Motion(double x0, double y0, double psi0, double v0)
    {
        this->m_initial_state = Vehicle_State(x0, y0, psi0, v0);
        this->m_wheel_base = 4.0;
        this->m_max_velocity = 28.0;
        this->m_max_acceleration = 2.0;
        this->m_max_steering = 0.8;
        this->reset();
    }
    void reset()
    {
        this->m_current_state = this->m_initial_state;
        this->m_steering = this->m_initial_state.steering;
        this->m_velocity = this->m_initial_state.velocity;
    }
    void reset(Vehicle_State state)
    {
        this->m_initial_state = state;
        this->reset();
    }
    void set_steering(double steer)
    {
        this->m_steering = steer;
    }
    void set_velocity(double velocity)
    {
        this->m_velocity = velocity;
    }
    Vehicle_State get_vehicle_state() const
    {
        return this->m_current_state;
    }

    void update(double dt)
    {
        double cos_psi = cos(this->m_current_state.psi);
        double sin_psi = sin(this->m_current_state.psi);

        double x = this->m_current_state.x + this->m_current_state.velocity * cos_psi * dt;
        double y = this->m_current_state.y + this->m_current_state.velocity * sin_psi * dt;

        double acceleration = this->m_velocity - this->m_current_state.velocity;
        // Max acceleration
        if (acceleration > this->m_max_acceleration)
        {
            acceleration = m_max_acceleration;
        }
        // Max brake
        if (acceleration < -this->m_max_acceleration)
        {
            acceleration = -this->m_max_acceleration;
        }
        double steer = this->m_steering;

        // Max right turn
        if (steer > this->m_max_steering)
        {
            steer = this->m_max_steering;
        }

        // Max left turn
        if (steer < -this->m_max_steering)
        {
            steer = -this->m_max_steering;
        }

        double velocity = this->m_current_state.velocity + acceleration * dt;

        if (velocity > this->m_max_velocity)
        {
            velocity = this->m_max_velocity;
        }
        if (velocity < -this->m_max_velocity)
        {
            velocity = -this->m_max_velocity;
        }

        double psi_dot = this->m_current_state.velocity * steer / this->m_wheel_base;
        double psi = wrap_angle(this->m_current_state.psi + psi_dot * dt);

        // Updating state
        this->m_current_state = Vehicle_State(x, y, psi, velocity, psi_dot, steer);
    }

private:
    Vehicle_State m_initial_state;
    Vehicle_State m_current_state;
    double m_wheel_base;
    double m_velocity;
    double m_steering;
    double m_max_acceleration;
    double m_max_velocity;
    double m_max_steering;
};

class Car
{
public:
    Car()
    {
        this->m_vehicle_model;
        this->m_current_motion = nullptr;
        // Car body
        this->m_car_lines_body = {{2, -1}, {2, 1}, {-2, 1}, {-2, -1}, {2, -1}};

        // Car Sensor Unit
        this->m_marker_lines = {{{0.5, 0.5}, {-0.5, -0.5}}, {{0.5, -0.5}, {-0.5, 0.5}}, {{0, 0}, {3.5, 0}}};

        // Wheel
        this->m_wheel_lines = {{-0.6, 0.3}, {0.6, 0.3}, {0.6, -0.3}, {-0.6, -0.3}, {-0.6, 0.3}};

        // Wheel position
        this->m_wheel_front_left_offset = Vector2(2, -1.6);
        this->m_wheel_front_right_offset = Vector2(2, 1.6);
        this->m_wheel_rear_left_offset = Vector2(-2, 1.6);
        this->m_wheel_rear_right_offset = Vector2(-2, -1.6);
    }

    void reset(double x0, double y0, double psi0, double v0)
    {
        this->m_vehicle_model.reset(Vehicle_State(x0, y0, psi0, v0));
        while (!this->m_vehicle_motions.empty())
        {
            this->m_vehicle_motions.pop();
        }
        this->m_current_motion = nullptr;
    }

    void add_vehicle_motion(Motion *motion)
    {
        if (motion != nullptr)
        {
            this->m_vehicle_motions.push(motion);
        }
    }

    bool update(double time, double dt)
    {
        if (this->m_current_motion == nullptr && !this->m_vehicle_motions.empty())
        {
            this->m_current_motion = this->m_vehicle_motions.front();
            this->m_vehicle_motions.pop();
            this->m_current_motion->start(time, this->m_vehicle_model.get_vehicle_state());
        }
        // Executing current motion
        if (this->m_current_motion != nullptr)
        {
            bool motion_complete = this->m_current_motion->update(time, dt, this->m_vehicle_model.get_vehicle_state());
            this->m_vehicle_model.set_steering(this->m_current_motion->get_steering());
            this->m_vehicle_model.set_velocity(this->m_current_motion->get_velocity());

            // Ending current motion
            if (motion_complete)
            {
                this->m_current_motion = nullptr;
            }
        }
        else
        {
            this->m_vehicle_model.set_steering(0.0);
            this->m_vehicle_model.set_velocity(0.0);
        }
    }

private:
    Bicycle_Motion m_vehicle_model;
    Motion *m_current_motion;
    std::queue<Motion *> m_vehicle_motions;
    std::vector<Vector2> m_car_lines_body;
    std::vector<Vector2> m_wheel_lines;
    std::vector<std::vector<Vector2>> m_marker_lines;
    Vector2 m_wheel_front_left_offset,
        m_wheel_front_right_offset,
        m_wheel_rear_left_offset,
        m_wheel_rear_right_offset;
};

#endif // INCLUDE_NL_KALAMAN_CAR_H