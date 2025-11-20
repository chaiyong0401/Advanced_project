#include "controller.h"

#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/frames.hpp>

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/geometry.hpp>


cRoboticsController::cRoboticsController(const std::string& urdf_path, 
                                         const std::string& manipulator_control_mode, 
                                         const double& dt)
: manipulator_control_mode_(manipulator_control_mode),
  dt_(dt)
{
    loadPath("/home/dyros-recruit/fr3_control_mujoco_template_ad_pro/planned_path.txt");
    pinocchio::urdf::buildModel(urdf_path, model_);
    data_ = pinocchio::Data(model_);

    // Initialize joint space state
    q_ = VectorXd::Zero(model_.nq);
    qdot_ = VectorXd::Zero(model_.nq);
    tau_ = VectorXd::Zero(model_.nq);
    q_desired_ = VectorXd::Zero(model_.nq);
    tau_desired_ = VectorXd::Zero(model_.nq);
    q_init_ = VectorXd::Zero(model_.nq);
    qdot_init_ = VectorXd::Zero(model_.nq);

    // Initialize task space state
    x_ = Matrix4d::Identity();
    xdot_ = VectorXd::Zero(6);
    J_ = MatrixXd::Zero(6, model_.nv);
    x2_ = Matrix4d::Identity();
    x2dot_ = VectorXd::Zero(6);
    J2_ = MatrixXd::Zero(6, model_.nv);

    // Initialize joint space dynamics
    M_ = MatrixXd::Zero(model_.nq, model_.nv);
    g_ = VectorXd::Zero(model_.nq);       
    c_ = VectorXd::Zero(model_.nq);       

    logging_file_.open("logging.txt");
}

void cRoboticsController::keyMapping(const int &key)
{
    switch (key)
    {
    // Implement with user input
    case '1':
       setMode(joint_ctrl_init);
        break;
    case '2':
       setMode(joint_ctrl_home);
        break;
    
    // --------------------------------------------------------------------------------------
    // TODO 3: Add your keyboard mapping here (using the control modes from TODO 1)
    // Example:
    case '3':
       setMode(hw2_1);
        break;
    // --------------------------------------------------------------------------------------
    case '4':
       setMode(maze_init);
        break;
    case '5':
       setMode(maze_tracking);
        break;
    case '6':
       setMode(torque_ctrl_dynamic);
        break;

    default:
        break;
    }
}

void cRoboticsController::compute(const double& play_time)
{
    play_time_ = play_time;
    if(is_mode_changed_)
    {
        is_mode_changed_ = false;
        control_start_time_ = play_time_;
        q_init_ = q_;
        qdot_init_ = qdot_;
        x_init_ = x_;
        x2_init_ = x2_;
    }

    switch (control_mode_)
    {
        case joint_ctrl_init:
        {
            Vector7d target_position;
            target_position << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, M_PI / 4.;
            if(manipulator_control_mode_ == "position") moveJointPosition(target_position, 2.0);
            else moveJointPositionTorque(target_position, 2.0);
            break;
        }
        case joint_ctrl_home:
        {
            Vector7d target_position;
            target_position << 0.0, 0.0, 0.0, -M_PI/2., 0.0, M_PI/2., M_PI / 4.;
            if(manipulator_control_mode_ == "position") moveJointPosition(target_position, 2.0);
            else moveJointPositionTorque(target_position, 2.0);
            break;
        }
        // ----------------------------------------------------------------------------
        // TODO 4: Add your control logic here (using the control functions from TODO 5)
        // Example:
        case hw2_1:
        {
            HW2_1();
            break;
        }
        case maze_init:
        {
            HW_Maze_Tracking_Init();
            break;
        }

        case maze_tracking:
        {
            HW_Maze_Tracking();
            break;
        }
        case torque_ctrl_dynamic:
        {
            torqueCtrlDynamic();
            break;
        }
        
        // ----------------------------------------------------------------------------
        
        default:
            if(manipulator_control_mode_ == "torque") tau_desired_ = g_;
            break;
    }
}

// =============================================================================
// =============================== User functions ==============================
// =============================================================================
void cRoboticsController::moveJointPosition(const VectorXd& target_position, double duration)
{
	Vector7d zero_vector;
	zero_vector.setZero();
	q_desired_ = DyrosMath::cubicVector<7>(play_time_,
		                                   control_start_time_,
		                                   control_start_time_ + duration, 
                                           q_init_, 
                                           target_position, 
                                           zero_vector, 
                                           zero_vector);
}

void cRoboticsController::moveJointPositionTorque(const VectorXd& target_position, double duration)
{
	Matrix7d kp, kv;
	Vector7d q_cubic, qd_cubic;
	
	kp = Matrix7d::Identity() * 500.0;
	kv = Matrix7d::Identity() * 20.0;

	for (int i = 0; i < 7; i++)
	{
		qd_cubic(i) = DyrosMath::cubicDot(play_time_, 
                                          control_start_time_,
			                              control_start_time_ + duration, q_init_(i), 
                                          target_position(i), 
                                          0, 
                                          0);
		q_cubic(i) = DyrosMath::cubic(play_time_,
                                      control_start_time_,
			                          control_start_time_ + duration, 
                                      q_init_(i), 
                                      target_position(i), 
                                      0, 
                                      0);
	}


	tau_desired_ = M_ * (kp*(q_cubic - q_) + kv*(qd_cubic - qdot_)) + g_;
}

// src/controller.cpp

void cRoboticsController::loadPath(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open path file: " << filename << std::endl;
        return;
    }

    maze_path_.clear();
    double x, y, z, qx, qy, qz, qw;
    while (file >> x >> y >> z >> qw >> qx >> qy >> qz) {
        Vector7d waypoint;
        waypoint << x, y, z, qw, qx, qy, qz;
        maze_path_.push_back(waypoint);
    }
    std::cout << "Loaded " << maze_path_.size() << " waypoints from " << filename << std::endl;
    file.close();
}

// -------------------------------------
// TODO 5: Add your control functions here
// Example:
void cRoboticsController::HW2_1()
{
    // logging_file_ << ... << std::endl;
    // q_desired_ = 
}

void cRoboticsController::torqueCtrlDynamic()
{
    // logging_file_ << ... << std::endl;
    // torque_desired_ = 
}

// src/controller.cpp (적절한 위치에 추가)

void cRoboticsController::HW_Maze_Tracking_Init()
{
    if (maze_path_.empty()) {
        std::cerr << "Path is empty! Cannot perform initialization move." << std::endl;
        setMode(DEFAULT);
        return;
    }
    
    // --- 설정 변수 ---
    double t_phase_1 = 2.0;
    double init_duration = 3.0; // 초기 위치로 이동하는 데 걸리는 시간 (3초)
    double safe_height_offset = 0.2;
    double current_time = play_time_ - control_start_time_;

    Vector7d p_start = maze_path_[0];
    Vector3d x_target = p_start.head(3);

    Vector3d x_hover = x_target;
    x_hover(2) += safe_height_offset; // 안전 높이 추가

    Eigen::Quaterniond Q_target(p_start(3), p_start(4), p_start(5), p_start(6)); 
    Vector3d x_0 = x_init_.block<3,1>(0,3);
    Matrix3d R_0 = x_init_.block<3,3>(0,0);
    Eigen::Quaterniond Q_0(R_0);

    // std::cout << "p_start" << p_start(0) << p_start(1) << p_start(2) << p_start(3) << p_start(4) << p_start(5) << p_start(6) << std::endl;
    // std::cout << "x_hover: " << x_hover.transpose() << std::endl;

    Vector3d x_des, xdot_des;
    Matrix3d R_des;
    Vector3d omega_des = Vector3d::Zero();
    

    if (current_time < t_phase_1)
    {
        for (int i = 0; i < 3; i++) {
            x_des(i) = DyrosMath::cubic(current_time, 0, t_phase_1, x_0(i), x_hover(i), 0, 0);
            xdot_des(i) = DyrosMath::cubicDot(current_time, 0, t_phase_1, x_0(i), x_hover(i), 0, 0);
        }
    }
    else
    {   
        for (int i = 0; i < 3; i++) {
            x_des(i) = DyrosMath::cubic(current_time, t_phase_1, init_duration, x_hover(i), x_target(i), 0, 0);
            xdot_des(i) = DyrosMath::cubicDot(current_time, t_phase_1, init_duration, x_hover(i), x_target(i), 0, 0);
        }
    }

    
    R_des = R_0; // 초기 자세 유지

    Matrix4d x_curr_mat = getEEPose(q_desired_);
    Vector3d x_curr = x_curr_mat.block<3,1>(0,3);
    Matrix3d R_curr = x_curr_mat.block<3,3>(0,0);

    Vector3d x_err = x_des - x_curr;
    Vector3d phi_err = DyrosMath::getPhi(R_des, R_curr);
    
    Vector6d task_err;
    task_err << x_err, phi_err;

    MatrixXd J = getEEJac(q_desired_);
    MatrixXd JJt = J * J.transpose();
    double damping = 1e-4; 
    MatrixXd J_pinv = J.transpose() * (JJt + damping * MatrixXd::Identity(6, 6)).inverse();

    Vector6d xdot_ref;
    xdot_ref << xdot_des, omega_des; 

    Matrix6d Kp = Matrix6d::Identity() * 500.0; 

    VectorXd qdot_d = J_pinv * (xdot_ref + Kp * task_err);
    q_desired_ += qdot_d * dt_;

    logging_file_ << play_time_ << " " 
                  << x_des.transpose() << " " 
                  << x_curr.transpose() << " "
                  << "PHASE1_INIT_MOVE" << std::endl;
}


void cRoboticsController::HW_Maze_Tracking()
{
    if (maze_path_.empty()) {
        std::cerr << "Path is empty!" << std::endl;
        return;
    }

    // --- 설정 변수 ---
    double time_per_segment = 0.02; // 경로 점 사이 시간 (속도 조절)
    double stick_length = 0.03; // 막대 길이
    Vector3d omega_des = Vector3d::Zero();
    Matrix3d R_0 = x_init_.block<3,3>(0,0);
    Matrix3d R_des;
    
    double path_time = play_time_ - control_start_time_;
    double exact_index = path_time / time_per_segment;
    int idx = static_cast<int>(exact_index);
    double alpha = exact_index - idx;


    // if (idx >= maze_path_.size() - 1) {
    //     std::cout << "MAZE_TRACKING complete! Holding final pose." << std::endl;
    //     idx = maze_path_.size() - 2;
    //     alpha = 1.0; 
    // }

    Vector7d p1 = maze_path_[idx];
    Vector7d p2 = maze_path_[idx + 1];


    Vector3d x_des = (1.0 - alpha) * p1.head(3) + alpha * p2.head(3);
    

    Vector3d xdot_des = (p2.head(3) - p1.head(3)) / time_per_segment;
    if (idx >= maze_path_.size() - 2 && alpha == 1.0) {
        xdot_des.setZero(); // 정지
    }

    Eigen::Quaterniond Q1(p1(6), p1(3), p1(4), p1(5)); // w, x, y, z
    Eigen::Quaterniond Q2(p2(6), p2(3), p2(4), p2(5));
    
    if (Q1.dot(Q2) < 0.0) {
        Q2.coeffs() = -Q2.coeffs(); // 최단 경로 보정
    }
    
    Eigen::Quaterniond Q_des = Q1.slerp(alpha, Q2);
    R_des = Q_des.toRotationMatrix();

    // R_des = R_0; // 초기 자세 유지

    // x_des(2) += stick_length; // 막대 길이만큼 z축 올리기
    Vector3d stick_direction = R_des.col(2); // EE의 Z축 방향 벡터
    x_des -= stick_direction * stick_length;


    Matrix4d x_curr_mat = getEEPose(q_desired_);
    Vector3d x_curr = x_curr_mat.block<3,1>(0,3);
    Matrix3d R_curr = x_curr_mat.block<3,3>(0,0);

    Vector3d x_err = x_des - x_curr;
    Vector3d phi_err = DyrosMath::getPhi(R_des, R_curr);
    
    Vector6d task_err;
    task_err << x_err, phi_err;

    MatrixXd J = getEEJac(q_desired_);
    MatrixXd JJt = J * J.transpose();
    double damping = 1e-4; 
    MatrixXd J_pinv = J.transpose() * (JJt + damping * MatrixXd::Identity(6, 6)).inverse();

    Vector6d xdot_ref;
    xdot_ref << xdot_des, omega_des; 

    Matrix6d Kp = Matrix6d::Identity() * 500.0; 

    VectorXd qdot_d = J_pinv * (xdot_ref + Kp * task_err);
    q_desired_ += qdot_d * dt_;

    // Logging
    logging_file_ << play_time_ << " " 
                  << x_des.transpose() << " " 
                  << x_curr.transpose() << " "
                  << "PHASE2_TRACKING" << std::endl;
}


// -------------------------------------
// =============================================================================

void cRoboticsController::setMode(const CTRL_MODE& control_mode)
{
    is_mode_changed_ = true;
    control_mode_ = control_mode;
    std::cout << "Control mode change: " << control_mode_ << std::endl;
}

void cRoboticsController::printState()
{
    // TODO 6: Extend or modify this for debugging your controller
    std::cout << "\n\n------------------------------------------------------------------" << std::endl;
    std::cout << "time     : " << std::fixed << std::setprecision(3) << play_time_ << std::endl;
    std::cout << "q now    :\t";
    std::cout << std::fixed << std::setprecision(3) << q_.transpose() << std::endl;
    std::cout << "q desired:\t";
    std::cout << std::fixed << std::setprecision(3) << q_desired_.transpose() << std::endl;
    std::cout << "x        :\n";
    std::cout << std::fixed << std::setprecision(3) << x_ << std::endl;
    std::cout << "x dot    :\t";
    std::cout << std::fixed << std::setprecision(3) << xdot_.transpose() << std::endl;
    std::cout << "J        :\n";
    std::cout << std::fixed << std::setprecision(3) << J_ << std::endl;
}

Matrix4d cRoboticsController::getLinkPose(const VectorXd& q, const std::string& link_name)
{
    if(q.size() != model_.nq)
    {
        std::cerr << "getEEPose Error: size of q " << q.size() << " is not equal to model.nq size: " << model_.nq << std::endl;
        return Matrix4d::Identity();
    }
    pinocchio::FrameIndex link_index = model_.getFrameId(link_name);
    if (link_index == static_cast<pinocchio::FrameIndex>(-1))  
    {
        std::cerr << "Error: Link name " << link_name << " not found in URDF." << std::endl;
        return Matrix4d::Identity();
    }

    pinocchio::Data data_tmp(model_);
    pinocchio::framesForwardKinematics(model_, data_tmp, q);
    return data_tmp.oMf[link_index].toHomogeneousMatrix();
}

MatrixXd cRoboticsController::getLinkJac(const VectorXd& q, const std::string& link_name)
{
    if(q.size() != model_.nq)
    {
        std::cerr << "getEEJac Error: size of q " << q.size() << " is not equal to model.nq size: " << model_.nq << std::endl;
        return MatrixXd::Zero(6, model_.nv);
    }
    pinocchio::FrameIndex link_index = model_.getFrameId(link_name);
    if (link_index == static_cast<pinocchio::FrameIndex>(-1))  
    {
        std::cerr << "Error: Link name " << link_name << " not found in URDF." << std::endl;
        return MatrixXd::Zero(6, model_.nv);
    }

    MatrixXd J;
    J.setZero(6, model_.nv);
    pinocchio::Data data_tmp(model_);
    pinocchio::computeJointJacobians(model_, data_tmp, q);
    pinocchio::getFrameJacobian(model_, data_tmp, link_index, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J);

    return J;
}

Matrix4d cRoboticsController::getEEPose(const VectorXd& q)
{
    if(q.size() != model_.nq)
    {
        std::cerr << "getEEPose Error: size of q " << q.size() << " is not equal to model.nq size: " << model_.nq << std::endl;
        return Matrix4d::Identity();
    }
    pinocchio::FrameIndex ee_index = model_.getFrameId(ee_name_);
    if (ee_index == static_cast<pinocchio::FrameIndex>(-1))  
    {
        std::cerr << "Error: Link name " << ee_name_ << " not found in URDF." << std::endl;
        return Matrix4d::Identity();
    }

    pinocchio::Data data_tmp(model_);
    pinocchio::framesForwardKinematics(model_, data_tmp, q);
    return data_tmp.oMf[ee_index].toHomogeneousMatrix();
}

MatrixXd cRoboticsController::getEEJac(const VectorXd& q)
{
    if(q.size() != model_.nq)
    {
        std::cerr << "getEEJac Error: size of q " << q.size() << " is not equal to model.nq size: " << model_.nq << std::endl;
        return MatrixXd::Zero(6, model_.nv);
    }
    pinocchio::FrameIndex ee_index = model_.getFrameId(ee_name_);
    if (ee_index == static_cast<pinocchio::FrameIndex>(-1))  
    {
        std::cerr << "Error: Link name " << ee_name_ << " not found in URDF." << std::endl;
        return MatrixXd::Zero(6, model_.nv);
    }

    MatrixXd J;
    J.setZero(6, model_.nv);
    pinocchio::Data data_tmp(model_);
    pinocchio::computeJointJacobians(model_, data_tmp, q);
    pinocchio::getFrameJacobian(model_, data_tmp, ee_index, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J);

    return J;
}

MatrixXd cRoboticsController::getMassMatrix(const VectorXd& q)
{
    if(q.size() != model_.nq)
    {
        std::cerr << "getMassMatrix Error: size of q " << q.size() << " is not equal to model.nq size: " << model_.nq << std::endl;
        return MatrixXd::Zero(model_.nq, model_.nq);
    }
    pinocchio::Data data_tmp(model_);
    pinocchio::crba(model_, data_tmp, q);

    return data_tmp.M.selfadjointView<Upper>();  // Only upper triangular part of M_ is computed by pinocchio::crba
}

VectorXd cRoboticsController::getGravityVector(const VectorXd& q)
{
    if(q.size() != model_.nq)
    {
        std::cerr << "getGravityVector Error: size of q " << q.size() << " is not equal to model.nq size: " << model_.nq << std::endl;
        return VectorXd::Zero(model_.nq);
    }
    pinocchio::Data data_tmp(model_);
    pinocchio::computeGeneralizedGravity(model_, data_tmp, q);

    return data_tmp.g;
}

bool cRoboticsController::updateModel(const VectorXd& q, const VectorXd& qdot, const VectorXd& tau)
{
    q_ = q;
    qdot_ = qdot;
    tau_ = tau;
    if(!updateKinematics(q_, qdot_)) return false;
    if(!updateDynamics(q_, qdot_)) return false;

    return true;
}

bool cRoboticsController::updateKinematics(const VectorXd& q, const VectorXd& qdot)
{
    if(q.size() != model_.nq)
    {
        std::cerr << "updateKinematics Error: size of q " << q.size() << " is not equal to model.nq size: " << model_.nq << std::endl;
        return false;
    }
    if(qdot.size() != model_.nv)
    {
        std::cerr << "updateKinematics Error: size of qdot " << qdot.size() << " is not equal to model.nv size: " << model_.nv << std::endl;
        return false;
    }
    pinocchio::FrameIndex ee_index = model_.getFrameId(ee_name_);
    if (ee_index == static_cast<pinocchio::FrameIndex>(-1))  
    {
        std::cerr << "Error: Link name " << ee_name_ << " not found in URDF." << std::endl;
        return false;
    }

    pinocchio::FrameIndex link4_index = model_.getFrameId(link_names_[4]);
    if (link4_index == static_cast<pinocchio::FrameIndex>(-1))  
    {
        std::cerr << "Error: Link name " << link_names_[4] << " not found in URDF." << std::endl;
        return false;
    }

    pinocchio::computeJointJacobians(model_, data_, q);

    pinocchio::getFrameJacobian(model_, data_, ee_index, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J_);
    x_ = data_.oMf[ee_index].toHomogeneousMatrix();
    xdot_ = J_ * qdot;

    pinocchio::getFrameJacobian(model_, data_, link4_index, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J2_);
    x2_ = data_.oMf[link4_index].toHomogeneousMatrix();
    x2dot_ = J2_ * qdot;

    return true;
}

bool cRoboticsController::updateDynamics(const VectorXd& q, const VectorXd& qdot)
{
    if(q.size() != model_.nq)
    {
        std::cerr << "updateDynamics Error: size of q " << q.size() << " is not equal to model.nq size: " << model_.nq << std::endl;
        return false;
    }
    if(qdot.size() != model_.nv)
    {
        std::cerr << "updateDynamics Error: size of qdot " << qdot.size() << " is not equal to model.nv size: " << model_.nv << std::endl;
        return false;
    }
    pinocchio::crba(model_, data_, q);
    pinocchio::computeGeneralizedGravity(model_, data_, q);
    pinocchio::computeCoriolisMatrix(model_, data_, q, qdot);

    // update joint space dynamics
    M_ = data_.M;
    M_ = M_.selfadjointView<Upper>();  // Only upper triangular part of M_ is computed by pinocchio::crba
    g_ = data_.g;
    c_ = data_.C * qdot_;

    return true;
}

VectorXd cRoboticsController::getCtrlInput()
{
    if(manipulator_control_mode_ == "position")
    {
        return q_desired_;
    }
    else
    {
        return tau_desired_;
    }
}