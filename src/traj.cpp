#include <iostream>
#include <rclcpp/rclcpp.hpp>
#include <cmath>
#include <px4_msgs/msg/vehicle_local_position.hpp>
#include <px4_msgs/msg/vehicle_attitude.hpp>
#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/vehicle_command.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <offboard_rl/utils.h>

#include <vector>
#include <sstream>
#include <thread>
#include <mutex>

using namespace std::chrono_literals;
using namespace px4_msgs::msg;
using Eigen::Vector3d;
using Eigen::Vector4d;

class Traj : public rclcpp::Node
{
public:
    Traj() : Node("go_to_point_multi_wp")
    {
        rmw_qos_profile_t qos_profile = rmw_qos_profile_sensor_data;
        auto qos = rclcpp::QoS(rclcpp::QoSInitialization(qos_profile.history, 5), qos_profile);

        local_position_subscription_ = this->create_subscription<px4_msgs::msg::VehicleLocalPosition>(
            "/fmu/out/vehicle_local_position", qos,
            std::bind(&Traj::vehicle_local_position_callback, this, std::placeholders::_1));
        attitude_subscription_ = this->create_subscription<px4_msgs::msg::VehicleAttitude>(
            "/fmu/out/vehicle_attitude", qos,
            std::bind(&Traj::vehicle_attitude_callback, this, std::placeholders::_1));
        offboard_control_mode_publisher_ = this->create_publisher<px4_msgs::msg::OffboardControlMode>("/fmu/in/offboard_control_mode", 10);
        trajectory_setpoint_publisher_ = this->create_publisher<px4_msgs::msg::TrajectorySetpoint>("/fmu/in/trajectory_setpoint", 10);
        vehicle_command_publisher_ = this->create_publisher<px4_msgs::msg::VehicleCommand>("/fmu/in/vehicle_command", 10);

        timer_offboard_ = this->create_wall_timer(100ms, std::bind(&Traj::activate_offboard, this));
        timer_trajectory_publish_ = this->create_wall_timer(20ms, std::bind(&Traj::publish_trajectory_setpoint, this));

        keyboard_thread = std::thread(&Traj::keyboard_listener, this);
    }

    ~Traj() {
        if (keyboard_thread.joinable()) keyboard_thread.join();
    }

private:
    struct Waypoint {
        Eigen::Vector4d xyzyaw; // x,y,z,yaw
    };

    struct QuinticCoeffs {
        // coefficients a0 + a1 t + a2 t^2 + a3 t^3 + a4 t^4 + a5 t^5
        Eigen::Matrix<double,6,1> a;
        double duration;
    };

    rclcpp::Subscription<px4_msgs::msg::VehicleLocalPosition>::SharedPtr local_position_subscription_;
    rclcpp::Subscription<px4_msgs::msg::VehicleAttitude>::SharedPtr attitude_subscription_;
    rclcpp::Publisher<px4_msgs::msg::OffboardControlMode>::SharedPtr offboard_control_mode_publisher_;
    rclcpp::Publisher<px4_msgs::msg::VehicleCommand>::SharedPtr vehicle_command_publisher_;
    rclcpp::Publisher<px4_msgs::msg::TrajectorySetpoint>::SharedPtr trajectory_setpoint_publisher_;

    rclcpp::TimerBase::SharedPtr timer_offboard_;
    rclcpp::TimerBase::SharedPtr timer_trajectory_publish_;

    std::thread keyboard_thread;
    std::mutex mtx_;

    bool set_point_received{false};
    bool offboard_active{false};
    bool trajectory_computed{false};

    VehicleLocalPosition current_position_{};
    VehicleAttitude current_attitude_{};

    double offboard_counter{0};

    std::vector<Waypoint> waypoints_;
    std::vector<std::array<QuinticCoeffs,4>> segment_coeffs_; // for each segment: coeffs for x,y,z,yaw
    std::vector<double> segment_times_; // duration each segment
    double total_T{0.0};
    double global_time{0.0}; // time since trajectory start
    size_t current_segment_idx{0};

    // callbacks
    void vehicle_local_position_callback(const px4_msgs::msg::VehicleLocalPosition::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lk(mtx_);
        current_position_ = *msg;
    }

    void vehicle_attitude_callback(const px4_msgs::msg::VehicleAttitude::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lk(mtx_);
        current_attitude_ = *msg;
    }

    // compute quintic coefficients given p0,v0,a0 and pf,vf,af and duration T
    Eigen::Matrix<double,6,1> compute_quintic_coeffs(double p0, double v0, double a0, double pf, double vf, double af, double T)
    {
        Eigen::Matrix<double,6,6> M;
        Eigen::Matrix<double,6,1> b;
        // a0 + a1 t + a2 t^2 + a3 t^3 + a4 t^4 + a5 t^5
        // constraints at t=0: p(0)=p0, v(0)=v0, a(0)=a0
        // at t=T: p(T)=pf, v(T)=vf, a(T)=af
        M.setZero();
        // p(0) = a0
        M(0,0) = 1.0;
        b(0) = p0;
        // v(0) = a1
        M(1,1) = 1.0;
        b(1) = v0;
        // a(0) = 2 a2
        M(2,2) = 2.0;
        b(2) = a0;

        // p(T)
        M(3,0) = 1.0;
        M(3,1) = T;
        M(3,2) = T*T;
        M(3,3) = std::pow(T,3);
        M(3,4) = std::pow(T,4);
        M(3,5) = std::pow(T,5);
        b(3) = pf;
        // v(T)
        M(4,1) = 1.0;
        M(4,2) = 2.0*T;
        M(4,3) = 3.0*std::pow(T,2);
        M(4,4) = 4.0*std::pow(T,3);
        M(4,5) = 5.0*std::pow(T,4);
        b(4) = vf;
        // a(T)
        M(5,2) = 2.0;
        M(5,3) = 6.0*T;
        M(5,4) = 12.0*std::pow(T,2);
        M(5,5) = 20.0*std::pow(T,3);
        b(5) = af;

        Eigen::Matrix<double,6,1> a = M.inverse() * b;
        return a;
    }

    // evaluate quintic and derivatives
    void eval_quintic(const Eigen::Matrix<double,6,1>& a, double t, double &p, double &v, double &acc)
    {
        double t2 = t*t;
        double t3 = t2*t;
        double t4 = t3*t;
        double t5 = t4*t;
        p = a(0) + a(1)*t + a(2)*t2 + a(3)*t3 + a(4)*t4 + a(5)*t5;
        v = a(1) + 2.0*a(2)*t + 3.0*a(3)*t2 + 4.0*a(4)*t3 + 5.0*a(5)*t4;
        acc = 2.0*a(2) + 6.0*a(3)*t + 12.0*a(4)*t2 + 20.0*a(5)*t3;
    }

    // keyboard: read N waypoints (>=7) and total time (seconds)
    void keyboard_listener()
    {
        std::cout << "Inserisci il numero di waypoint (>=7), poi ogni waypoint su una riga come: x y z yaw (yaw in radianti)\n";
        std::cout << "Alla fine inserisci il tempo totale T (secondi) per completare la traiettoria.\n";
        std::cout << "Esempio:\n7\n0 0 -1 0\n1 0 -1 0.2\n1 1 -1 0.4\n0 1 -1 0.6\n-1 1 -1 0.8\n-1 0 -1 1.0\n0 0 -1 0\n30\n";
        std::cout << "Se l'input è invalido verrà usata una traiettoria di default con 7 waypoint.\n";

        std::string line;
        std::getline(std::cin, line);
        std::istringstream iss(line);
        int N;
        if (!(iss >> N) || N < 7) {
            std::cout << "Numero waypoint non valido. Uso default 7 waypoint.\n";
            populate_default_waypoints();
            total_T = 30.0;
            prepare_trajectory();
            set_point_received = true;
            return;
        }

        std::vector<Waypoint> wps;
        for (int i=0;i<N;i++) {
            std::string wp_line;
            if (!std::getline(std::cin, wp_line)) break;
            if (wp_line.empty()) { i--; continue; }
            std::istringstream ws(wp_line);
            double x,y,z,yaw;
            if (!(ws >> x >> y >> z >> yaw)) {
                std::cout << "Waypoint invalido (linea " << i+1 << "). Uso default.\n";
                populate_default_waypoints();
                total_T = 30.0;
                prepare_trajectory();
                set_point_received = true;
                return;
            }
            Waypoint wp;
            wp.xyzyaw << x, y, -z, yaw; // converti in NED 
            wps.push_back(wp);
        }
        // read T
        std::string tline;
        if (!std::getline(std::cin, tline) || tline.empty()) {
            std::cout << "Tempo totale non inserito. Uso default 30s.\n";
            total_T = 30.0;
        } else {
            std::istringstream ts(tline);
            if (!(ts >> total_T) || total_T <= 0.0) {
                std::cout << "Valore T invalido. Uso default 30s.\n";
                total_T = 30.0;
            }
        }

        {
            std::lock_guard<std::mutex> lk(mtx_);
            waypoints_ = std::move(wps);
            set_point_received = true;
        }
        std::cout << "Waypoints ricevuti: " << waypoints_.size() << " ; tempo totale T = " << total_T << " s\n";
        prepare_trajectory();
    }

    void populate_default_waypoints()
    {
        waypoints_.clear();
        // default: 7 waypoints (x,y,z,yaw)
        std::vector<std::array<double,4>> def = {
            {0.0, 0.0, 10.0, 0.0},
            {10.0, -10.0, 10.0, 0.1},
            {25.0, -15.0, 10.0, 0.2},
            {30.0, -5.0, 10.0, 0.3},
            {25.0, 0.0, 10.0, 0.7},
            {10.0, 0.0, 10.0, 0.3},
            {0.0, 0.0, 10.0, 0.0}
        };
        for (auto &d : def) {
            Waypoint w;
            w.xyzyaw << d[0], d[1], d[2], d[3];
            waypoints_.push_back(w);
        }
    }

    // compute velocities at intermediate waypoints (non-zero), then build quintic per segment
    void prepare_trajectory()
    {
        std::lock_guard<std::mutex> lk(mtx_);

        if (waypoints_.size() < 2) {
            RCLCPP_ERROR(this->get_logger(), "Pochi waypoint per preparare la traiettoria");
            return;
        }

        size_t N = waypoints_.size();
        // distances between consecutive waypoints (in pos+yaw metric for time allocation)
        std::vector<double> dists;
        dists.reserve(N-1);
        double total_dist = 0.0;
        for (size_t i=0;i<N-1;i++) {
            Eigen::Vector3d p0 = waypoints_[i].xyzyaw.head<3>();
            Eigen::Vector3d p1 = waypoints_[i+1].xyzyaw.head<3>();
            double d = (p1 - p0).norm();
            // include yaw difference scaled (small weight)
            double yaw_diff = utilities::angleError(waypoints_[i+1].xyzyaw(3), waypoints_[i].xyzyaw(3));
            d += 0.2 * std::abs(yaw_diff);
            dists.push_back(d);
            total_dist += d;
        }
        if (total_dist <= 0.0) total_dist = 1.0;

        // time allocation proportional to distances
        segment_times_.clear();
        for (size_t i=0;i<dists.size();i++) {
            double segT = (dists[i] / total_dist) * total_T;
            if (segT < 0.5) segT = 0.5; // min duration per segment
            segment_times_.push_back(segT);
        }

        // compute desired velocities at waypoints:
        // At start: velocity tangent toward next waypoint (non-zero)
        // At intermediate: average of incoming and outgoing unit tangents times a scalar speed (we choose cruise speed proportional to local geometry)
        // At final: zero
        std::vector<Eigen::Vector4d> vels(N, Eigen::Vector4d::Zero()); // vx, vy, vz, vyaw
        double base_speed = 0.6; // m/s baseline
        for (size_t i=0;i<N;i++) {
            if (i == N-1) {
                vels[i].setZero();
                continue;
            }
            if (i == 0) {
                Eigen::Vector3d dir = (waypoints_[1].xyzyaw.head<3>() - waypoints_[0].xyzyaw.head<3>());
                double norm = dir.norm();
                if (norm < 1e-6) dir.setZero();
                else dir /= norm;
                vels[0].head<3>() = dir * base_speed;
                double yaw_diff = utilities::angleError(waypoints_[1].xyzyaw(3), waypoints_[0].xyzyaw(3));
                //vels = yaw_diff / segment_times_[0]; // yaw rate
                Eigen::Vector4d vel_vec;
		vel_vec << 0.0, 0.0, 0.0, yaw_diff / segment_times_[0];

		// il segmento corrente è segment i
		vels[i] = vel_vec;
            } else {
                if (i == N-1) continue;
                Eigen::Vector3d dir_in = (waypoints_[i].xyzyaw.head<3>() - waypoints_[i-1].xyzyaw.head<3>());
                Eigen::Vector3d dir_out = (waypoints_[i+1].xyzyaw.head<3>() - waypoints_[i].xyzyaw.head<3>());
                double n_in = dir_in.norm();
                double n_out = dir_out.norm();
                if (n_in > 1e-6) dir_in /= n_in; else dir_in.setZero();
                if (n_out > 1e-6) dir_out /= n_out; else dir_out.setZero();
                Eigen::Vector3d dir_avg = dir_in + dir_out;
                double navg = dir_avg.norm();
                if (navg > 1e-6) {
                    dir_avg /= navg;
                    double prev_speed =  (waypoints_[i].xyzyaw.head<3>() - waypoints_[i-1].xyzyaw.head<3>()).norm()/segment_times_[i-1];
                    double post_speed =  (waypoints_[i+1].xyzyaw.head<3>() - waypoints_[i].xyzyaw.head<3>()).norm()/segment_times_[i];
                    double avg_speed = 0.5 * (prev_speed + post_speed);
                    vels[i].head<3>() = dir_avg * avg_speed;
                } else {
                    vels[i].head<3>() = dir_out * (base_speed * 0.5);          
                }
                // yaw rate: average of incoming/outgoing yaw rates
                double yaw_in = utilities::angleError(waypoints_[i].xyzyaw(3), waypoints_[i-1].xyzyaw(3)) / segment_times_[i-1];
                double yaw_out = utilities::angleError(waypoints_[i+1].xyzyaw(3), waypoints_[i].xyzyaw(3)) / segment_times_[i];
                //vels = 0.5*(yaw_in + yaw_out);
                Eigen::Vector4d vel_vec;
		vel_vec << vels[i].x(), vels[i].y(), vels[i].z(), 0.5*(yaw_in + yaw_out);
		vels[i] = vel_vec;

            }
        }
        // ensure last vel zero
        vels.back().setZero();

        // accelerations set to zero at boundaries
        std::vector<Eigen::Vector4d> accs(N, Eigen::Vector4d::Zero());

        // build quintic for each segment and axis
        segment_coeffs_.clear();
        segment_coeffs_.resize(segment_times_.size());
        for (size_t seg=0; seg<segment_times_.size(); ++seg) {
            double Tseg = segment_times_[seg];
            // start waypoint index = seg, end = seg+1
            Eigen::Vector4d p0 = waypoints_[seg].xyzyaw;
            Eigen::Vector4d pf = waypoints_[seg+1].xyzyaw;
            Eigen::Vector4d v0 = vels[seg];
            Eigen::Vector4d vf = vels[seg+1];
            Eigen::Vector4d a0 = accs[seg];
            Eigen::Vector4d af = accs[seg+1];

            std::array<QuinticCoeffs,4> coeffs_xyzyaw;

            for (int axis=0; axis<4; ++axis) {
                double p0_axis = p0(axis);
                double pf_axis = pf(axis);
                double v0_axis = v0(axis);
                double vf_axis = vf(axis);
                double a0_axis = a0(axis);
                double af_axis = af(axis);
                if (axis == 3) { // yaw: use angleError to get shortest difference
                    double yaw0 = p0(3);
                    double yawf = p0(3) + utilities::angleError(pf(3), p0(3));
                    // treat pf_axis as yawf (unwrapped)
                    pf_axis = yawf;
                    
                }
                QuinticCoeffs qc;
                qc.a = compute_quintic_coeffs(p0_axis, v0_axis, a0_axis, pf_axis, vf_axis, af_axis, Tseg);
                qc.duration = Tseg;
                coeffs_xyzyaw[axis] = qc;
            }
            segment_coeffs_[seg] = coeffs_xyzyaw;
        }

        // reset execution counters
        global_time = 0.0;
        current_segment_idx = 0;
        trajectory_computed = true;

        std::cout << "Traiettoria preparata: " << segment_times_.size() << " segmenti, tempo totale ~ " << total_T << " s\n";
    }

    void activate_offboard()
    {
        if (set_point_received)
        {
            if(offboard_counter == 10) {
                
                VehicleCommand msg{};
                msg.param1 = 1;
                msg.param2 = 6;
                msg.command = VehicleCommand::VEHICLE_CMD_DO_SET_MODE;
                msg.target_system = 1;
                msg.target_component = 1;
                msg.source_system = 1;
                msg.source_component = 1;
                msg.from_external = true;
                msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
                vehicle_command_publisher_->publish(msg);

                // Arm the vehicle
                msg.param1 = 1.0;
                msg.param2 = 0.0;
                msg.command = VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM;
                msg.target_system = 1;
                msg.target_component = 1;
                msg.source_system = 1;
                msg.source_component = 1;
                msg.from_external = true;
                msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
                vehicle_command_publisher_->publish(msg);

              
                offboard_active = true;
            }

            OffboardControlMode msg{};
            msg.position = true;
            msg.velocity = false;
            msg.acceleration = false;
            msg.attitude = false;
            msg.body_rate = false;
            msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
            offboard_control_mode_publisher_->publish(msg);

            if (offboard_counter < 11) offboard_counter++;
        }
    }

    void publish_trajectory_setpoint()
    {
        if (!set_point_received || !offboard_active || !trajectory_computed) {
            return;
        }

        double dt = 1.0/50.0; // 20 ms
    
        double tlocal = global_time;
        size_t seg = 0;
        double accum = 0.0;
        for (size_t i=0;i<segment_times_.size();i++) {
            if (tlocal <= accum + segment_times_[i]) {
                seg = i;
                break;
            }
            accum += segment_times_[i];
            seg = i+1;
        }
        if (seg >= segment_times_.size()) {
       
            TrajectorySetpoint msg{};
            auto last = waypoints_.back().xyzyaw;
            msg.position = {float(last(0)), float(last(1)), float(last(2))};
            msg.velocity = {0.0f, 0.0f, 0.0f};
            msg.acceleration = {0.0f, 0.0f, 0.0f};
            msg.yaw = float(last(3));
            msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
            trajectory_setpoint_publisher_->publish(msg);
            return;
        }

        double local_t = global_time - accum;
        if (local_t < 0) local_t = 0.0;
        if (local_t > segment_times_[seg]) local_t = segment_times_[seg];

        // evaluate each axis quintic
        double px, vx, ax;
        double py, vy, ay;
        double pz, vz, az;
        double pyaw, vyaw, ayay;

        eval_quintic(segment_coeffs_[seg][0].a, local_t, px, vx, ax);
        eval_quintic(segment_coeffs_[seg][1].a, local_t, py, vy, ay);
        eval_quintic(segment_coeffs_[seg][2].a, local_t, pz, vz, az);
        eval_quintic(segment_coeffs_[seg][3].a, local_t, pyaw, vyaw, ayay);

        TrajectorySetpoint msg{};
        msg.position = {float(px), float(py), float(pz)};
        msg.velocity = {float(vx), float(vy), float(vz)};
        msg.acceleration = {float(ax), float(ay), float(az)};
        // normalize yaw to [-pi,pi]
        double yaw_norm = std::fmod(pyaw + M_PI, 2.0*M_PI);
        if (yaw_norm < 0) yaw_norm += 2.0*M_PI;
        yaw_norm -= M_PI;
        msg.yaw = float(yaw_norm);
 

        msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
        trajectory_setpoint_publisher_->publish(msg);

        global_time += dt;
    }
};

int main(int argc, char *argv[])
{
    std::cout << "Starting go_to_point_multi_wp node..." << std::endl;
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Traj>());
    rclcpp::shutdown();
    return 0;
}

