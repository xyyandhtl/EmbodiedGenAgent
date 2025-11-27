/**********************************************************************
 Copyright (c) 2020-2023, Unitree Robotics.Co.Ltd. All rights reserved.
***********************************************************************/

#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <thread>
#include <algorithm>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include "common/ros2_sport_client.h"
#include "unitree_go/msg/sport_mode_state.hpp"

#define TOPIC_HIGHSTATE "lf/sportmodestate"

using namespace std::chrono_literals;

class Go2SportClientNode : public rclcpp::Node {
public:
    explicit Go2SportClientNode()
        : Node("go2_vel_sport"),
          sport_client_(this)
    {
        vel_[0] = vel_[1] = vel_[2] = 0.0;

        // 初始化最后一次收到速度的时间为当前时间
        last_cmd_time_ = this->now();

        // 订阅 /cmd_vel
        vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/cmd_vel", 10,
            std::bind(&Go2SportClientNode::set_vel, this, std::placeholders::_1));

        // 启动控制线程
        control_thread_ = std::thread([this] {
            std::this_thread::sleep_for(500ms);  // 等待ROS启动
            RobotControl();
        });
    }

    ~Go2SportClientNode() {
        if (control_thread_.joinable()) {
            control_thread_.join();
        }
    }

private:
    void set_vel(const geometry_msgs::msg::Twist::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(vel_mutex_);

        // 限幅
        vel_[0] = std::clamp(msg->linear.x, -1.0, 1.0);
        vel_[1] = std::clamp(msg->linear.y, -1.0, 1.0);
        vel_[2] = std::clamp(msg->angular.z, -0.5, 0.5);

        last_cmd_time_ = this->now();  // 更新最后一次接收速度的时间

        RCLCPP_INFO(this->get_logger(),
                    "Received cmd_vel -> linear.x=%.2f linear.y=%.2f angular.z=%.2f",
                    vel_[0], vel_[1], vel_[2]);
    }

    void RobotControl() {
        rclcpp::WallRate loop_rate(100); // 100 Hz
        const auto timeout = 200ms;      // 超过200ms没有新速度则置零

        while (rclcpp::ok()) {
            double vx, vy, vyaw;
            {
                std::lock_guard<std::mutex> lock(vel_mutex_);
                // 如果超时没有接收到新的速度，则置零
                if ((this->now() - last_cmd_time_) > rclcpp::Duration(timeout)) {
                    vx = vy = vyaw = 0.0;
                } else {
                    vx = vel_[0];
                    vy = vel_[1];
                    vyaw = vel_[2];
                }
            }

            sport_client_.Move(req_, vx, vy, vyaw);
            loop_rate.sleep();
        }
    }

private:
    SportClient sport_client_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr vel_sub_;
    unitree_api::msg::Request req_;  // Unitree Go2 ROS2 request message
    double vel_[3];
    std::mutex vel_mutex_;
    std::thread control_thread_;
    rclcpp::Time last_cmd_time_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);

    auto node = std::make_shared<Go2SportClientNode>();

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();

    rclcpp::shutdown();
    return 0;
}
