#ifndef ESTIMATOR_STAMPED_BUFF_H
#define ESTIMATOR_STAMPED_BUFF_H

#include <queue>
#include <Eigen/Dense>

struct IMUStampedBuffer {
    std::queue<double> time_;
    std::queue<Eigen::Vector3f> data_;

    void push(double t, const Eigen::Vector3f &d) {
        time_.push(t);
        data_.push(d);
    }

    // time is whenever camera measurement is received
    // I wish to integrate all the measurements until this time
    // and only then fuse the measurement
    bool get(double cam_t, Eigen::Vector3f &d, double &time) {
        if (time_.empty())
            return false;

        auto acc_time = time_.front();
        if (acc_time - cam_t > (30 * 1e-3))
            return false;

        d = data_.front();
        time = acc_time;
        pop();
        return true;
    }

    void pop() {
        time_.pop();
        data_.pop();
    }
};

#endif //ESTIMATOR_STAMPED_BUFF_H
