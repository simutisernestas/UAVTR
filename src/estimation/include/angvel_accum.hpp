#pragma once

#include <Eigen/Dense>
#include <mutex>

struct AngVelAccumulator {
    AngVelAccumulator() : x(0), y(0), z(0),
                          ang_vel_count(0) {}

    void add(const float x_add, const float y_add, const float z_add) {
        std::scoped_lock lock(mtx);
        x += x_add;
        y += y_add;
        z += z_add;
        ang_vel_count++;
    }

    void reset() {
        x = y = z = 0;
        ang_vel_count = 0;
    }

    [[nodiscard]] Eigen::Vector3f get_ang_vel() {
        std::scoped_lock lock(mtx);

        if (ang_vel_count == 0)
            return {0, 0, 0};

        Eigen::Vector3f ang_vel{x / (float) ang_vel_count,
                                y / (float) ang_vel_count,
                                z / (float) ang_vel_count};
        reset();

        return ang_vel;
    }

    float x, y, z;
    uint16_t ang_vel_count;
    std::mutex mtx;
};