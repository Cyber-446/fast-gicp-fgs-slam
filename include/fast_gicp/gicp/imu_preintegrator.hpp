#pragma once

#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/PreintegrationParams.h>
#include <gtsam/geometry/Rot3.h>

#include <Eigen/Core>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <memory>
#include <stdexcept>

class IMUPreintegrator {
public:
    using Matrix3d = Eigen::Matrix3d;
    using Matrix6d = Eigen::Matrix<double, 6, 6>;
    using Matrix9d = Eigen::Matrix<double, 9, 9>;
    using Vector3d = Eigen::Vector3d;

    struct Result {
        Matrix3d delta_R;
        Vector3d delta_p;
        Vector3d delta_p_pose;
        Vector3d delta_v;
        Matrix9d covariance;
        
        Matrix3d dR_d_bg;
        Matrix3d dP_d_ba;
        Matrix3d dP_d_bg;
        Matrix3d dV_d_ba;
        Matrix3d dV_d_bg;
        
        double dt = 0.0;
        Vector3d v_i_world;
    };

    IMUPreintegrator(
        double gravity,
        const Matrix3d& accel_cov,
        const Matrix3d& gyro_cov,
        const Matrix3d& integration_cov,
        const Vector3d& accel_bias = Vector3d::Zero(),
        const Vector3d& gyro_bias  = Vector3d::Zero())
    {

        params_ = gtsam::PreintegrationParams::MakeSharedD(gravity);

        params_->accelerometerCovariance = accel_cov;
        params_->gyroscopeCovariance = gyro_cov;
        params_->integrationCovariance = integration_cov;

        bias_ = gtsam::imuBias::ConstantBias(accel_bias, gyro_bias);
        
        pim_ = std::make_unique<gtsam::PreintegratedImuMeasurements>(params_, bias_);
    }

    void setState(const Matrix3d& R_i_world, const Vector3d& v_i_world) {
        R_i_ = R_i_world;
        v_i_world_ = v_i_world;
        has_state_ = true;
    }

    void reset() {
        pim_->resetIntegration();
    }

    void integrate(const Vector3d& accel, const Vector3d& gyro, double dt) {
        gtsam::Vector3 a(accel.x(), accel.y(), accel.z());
        gtsam::Vector3 w(gyro.x(), gyro.y(), gyro.z());
        pim_->integrateMeasurement(a, w, dt);
    }

    struct ImuDeltaData {
        Vector3d linear_acc;
        Matrix3d angular_vel;
        uint64_t timestamp_ns;
    };

    Result getResult(const ImuDeltaData* imu_delta = nullptr) const {
        Result out;
        if (imu_delta == nullptr){
            if (!has_state_) {
                throw std::runtime_error("IMUPreintegrator: state was not initialized");
            }

            const gtsam::Vector3 dP = pim_->deltaPij();
            const gtsam::Vector3 dV = pim_->deltaVij();

            out.delta_R = pim_->deltaRij().matrix();
            out.delta_p = Vector3d(dP.x(), dP.y(), dP.z());
            out.delta_v = Vector3d(dV.x(), dV.y(), dV.z());
            out.dt = pim_->deltaTij();

            const Vector3d v_i = R_i_.transpose() * v_i_world_;
            const Vector3d gravity_world = params_->n_gravity;
            const Vector3d g_i = R_i_.transpose() * gravity_world;

            out.delta_p_pose = out.delta_p + v_i * out.dt + 0.5 * g_i * out.dt * out.dt;
            out.v_i_world = v_i_world_;

            out.covariance = pim_->preintMeasCov();

            out.dR_d_bg = Matrix3d::Zero();
            out.dP_d_ba = Matrix3d::Zero();
            out.dP_d_bg = Matrix3d::Zero();
            out.dV_d_ba = Matrix3d::Zero();
            out.dV_d_bg = Matrix3d::Zero();
        } else {
            out.delta_p_pose = imu_delta->linear_acc;
            out.delta_R = imu_delta->angular_vel;
            out.covariance.setZero();
        }

        return out;
    }

    void finishInterval() {
        if (!has_state_) {
            throw std::runtime_error("IMUPreintegrator: state was not initialized");
        }

        const double dt = pim_->deltaTij();
        const gtsam::Vector3 dV = pim_->deltaVij();
        const Vector3d delta_v(dV.x(), dV.y(), dV.z());
        const Vector3d gravity_world = params_->n_gravity;

        v_i_world_ = v_i_world_ + gravity_world * dt + R_i_ * delta_v;
        R_i_ = R_i_ * pim_->deltaRij().matrix();
        pim_->resetIntegration();
    }

    const Vector3d& velocityWorld() const { return v_i_world_; }
    const Matrix3d& rotationWorld() const { return R_i_; }
    const gtsam::imuBias::ConstantBias& bias() const { return bias_; }

private:
    boost::shared_ptr<gtsam::PreintegrationParams> params_;
    
    gtsam::imuBias::ConstantBias bias_;
    std::unique_ptr<gtsam::PreintegratedImuMeasurements> pim_;

    Matrix3d R_i_ = Matrix3d::Identity();
    Vector3d v_i_world_ = Vector3d::Zero();
    bool has_state_ = false;

};