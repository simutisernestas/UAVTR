#include <stdio.h>
#include <iostream>
#include <kalman/ExtendedKalmanFilter.hpp>
#include <kalman/LinearizedSystemModel.hpp>

namespace robot
{
    // constant N states
    static constexpr size_t N = 10;

    template <typename T>
    class State : public Kalman::Vector<T, N>
    {
    public:
        KALMAN_VECTOR(State, T, N)
    };

    template <typename T>
    class Control : public Kalman::Vector<T, 0>
    {
    public:
        KALMAN_VECTOR(Control, T, 0)
    };

    template <typename T, template <class> class CovarianceBase = Kalman::StandardBase>
    class SystemModel : public Kalman::LinearizedSystemModel<State<T>, Control<T>, CovarianceBase>
    {
    public:
        //! State type shortcut definition
        typedef State<T> S;

        //! Control type shortcut definition
        typedef Control<T> C;

        SystemModel()
        {
            this->W.setIdentity();
            this->W(N - 1, N - 1) = 0.0;
        }

        //! Predicted state vector after transition
        S f(const S &x, const C &u) const
        {
            (void)u;

            S x_;
            x_.setZero();
            x_ = x;
            T lambda = x[N - 1];
            x_.head(3) += dt_ * (1 / lambda) * Eigen::Vector<T, 3>::Ones();
            x_.head(3) += dt_ * dt_ * 0.5 * (1 / lambda) * Eigen::Vector<T, 3>::Ones();
            x_.segment(3, 3) += dt_ * Eigen::Vector<T, 3>::Ones();
            return x_;
        }

        inline void set_dt(const T dt)
        {
            dt_ = dt;
        }

        void print_jacobians() const
        {
            std::cout << "F: " << '\n'
                      << this->F << '\n';
            std::cout << "W: " << '\n'
                      << this->W << '\n';
        }

    protected:
        void updateJacobians(const S &x, const C &u)
        {
            (void)u;
            // F = df/dx (Jacobian of state transition w.r.t. the state)
            // F is 10x10 matrix, start with identity matrix
            this->F.setIdentity();
            T lambda = x[N - 1];

            // block (1:3, 3:6) identity * dt/lambda
            this->F.block(0, 3, 3, 3) = dt_ / lambda * Eigen::Matrix<T, 3, 3>::Identity();
            // block (1:3, 6:9) identity * dt^2/2lambda
            this->F.block(0, 6, 3, 3) = dt_ * dt_ / (2 * lambda) * Eigen::Matrix<T, 3, 3>::Identity();
            // block (4:6, 6:9) identity * dt
            this->F.block(3, 6, 3, 3) = dt_ * Eigen::Matrix<T, 3, 3>::Identity();
            // block (1:3, 9:10) - 1 * dt/lambda^2 - 1 * dt^2/2lambda^2
            this->F.block(0, 9, 3, 1) = -dt_ / (lambda * lambda) * Eigen::Vector<T, 3>::Ones() - dt_ * dt_ / (2 * lambda * lambda) * Eigen::Vector<T, 3>::Ones();

            // W = df/dw (Jacobian of state transition w.r.t. the noise)
            this->W.setIdentity();
            // TODO: more sophisticated noise modelling
            //       i.e. The noise affects the the direction in which we move as
            //       well as the velocity (i.e. the distance we move)
        }

    private:
        T dt_{1 / 200.0};
    };

    template <typename T>
    class PositionMeasurement : public Kalman::Vector<T, 3>
    {
    public:
        KALMAN_VECTOR(PositionMeasurement, T, 3)
    };

    template <typename T, template <class> class CovarianceBase = Kalman::StandardBase>
    class PositionMeasurementModel : public Kalman::LinearizedMeasurementModel<State<T>, PositionMeasurement<T>, CovarianceBase>
    {
    public:
        //! State type shortcut definition
        typedef State<T> S;

        //! Measurement type shortcut definition
        typedef PositionMeasurement<T> M;

        PositionMeasurementModel()
        {
            // Setup noise jacobian. As this one is static, we can define it once
            // and do not need to update it dynamically
            this->V.setIdentity();
            this->V *= 0.1;
        }

        M h(const S &x) const
        {
            M measurement = x.head(3);
            return measurement;
        }

    protected:
        void updateJacobians(const S &x)
        {
            (void)x;
            // H = dh/dx (Jacobian of measurement function w.r.t. the state)
            this->H.setZero();
            this->H.block(0, 0, 3, 3) = Eigen::Matrix<T, 3, 3>::Identity();
        }
    };

    template <typename T>
    class AccelerationMeasurement : public Kalman::Vector<T, 3>
    {
    public:
        KALMAN_VECTOR(AccelerationMeasurement, T, 3)
    };

    template <typename T, template <class> class CovarianceBase = Kalman::StandardBase>
    class AccelerationMeasurementModel : public Kalman::LinearizedMeasurementModel<State<T>, AccelerationMeasurement<T>, CovarianceBase>
    {
    public:
        //! State type shortcut definition
        typedef State<T> S;

        //! Measurement type shortcut definition
        typedef AccelerationMeasurement<T> M;

        AccelerationMeasurementModel()
        {
            // Setup noise jacobian. As this one is static, we can define it once
            // and do not need to update it dynamically
            this->V.setIdentity();
            this->V *= 0.1;
        }

        M h(const S &x) const
        {
            M measurement = x.segment(6, 3);
            return measurement;
        }

    protected:
        void updateJacobians(const S &x)
        {
            (void)x;
            // H = dh/dx (Jacobian of measurement function w.r.t. the state)
            this->H.setZero();
            this->H.block(0, 6, 3, 3) = Eigen::Matrix<T, 3, 3>::Identity();
        }
    };
}
