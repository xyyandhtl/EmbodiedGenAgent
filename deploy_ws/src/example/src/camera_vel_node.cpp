#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include "common/ros2_sport_client.h"
#include "unitree_go/msg/sport_mode_state.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <thread>
#include <algorithm>

using namespace std::chrono_literals;

class Go2CameraVelNode : public rclcpp::Node
{
public:
    Go2CameraVelNode() : Node("go2_camera_node"), sport_client_(this)
    {
        // Declare parameters
        declare_parameter("enable_speed_control", true);
        declare_parameter("camera_interface", std::string("eth0"));
        declare_parameter("camera_address", std::string("230.1.1.1"));
        declare_parameter("camera_port", 1720);
        declare_parameter("camera_width", 1280);
        declare_parameter("camera_height", 720);
        declare_parameter("publish_compressed", false);
        declare_parameter("camera_image_raw_topic", std::string("/camera/rgb/image_raw"));
        declare_parameter("camera_image_compressed_topic", std::string("/camera/image/compressed"));
        declare_parameter("cmd_vel_topic", std::string("/cmd_vel"));
        declare_parameter("control_frequency", 100);
        declare_parameter("timeout_ms", 200);
        declare_parameter("max_linear_x", 1.0);
        declare_parameter("max_linear_y", 1.0);
        declare_parameter("max_angular_z", 0.5);

        // Get parameter values
        enable_speed_control_ = get_parameter("enable_speed_control").as_bool();
        std::string interface_name = get_parameter("camera_interface").as_string();
        std::string address = get_parameter("camera_address").as_string();
        int port = get_parameter("camera_port").as_int();
        int width = get_parameter("camera_width").as_int();
        int height = get_parameter("camera_height").as_int();
        publish_compressed_ = get_parameter("publish_compressed").as_bool();
        camera_image_raw_topic_ = get_parameter("camera_image_raw_topic").as_string();
        camera_image_compressed_topic_ = get_parameter("camera_image_compressed_topic").as_string();
        cmd_vel_topic_ = get_parameter("cmd_vel_topic").as_string();
        int control_freq = get_parameter("control_frequency").as_int();
        timeout_ms_ = std::chrono::milliseconds(get_parameter("timeout_ms").as_int());
        max_linear_x_ = get_parameter("max_linear_x").as_double();
        max_linear_y_ = get_parameter("max_linear_y").as_double();
        max_angular_z_ = get_parameter("max_angular_z").as_double();

        // Initialize velocity values
        vel_[0] = vel_[1] = vel_[2] = 0.0;
        last_cmd_time_ = this->now();

        // Setup camera if enabled
        setup_camera(interface_name, address, port, width, height);

        // Setup speed control if enabled
        if (enable_speed_control_) {
            setup_speed_control(control_freq);
            RCLCPP_INFO(this->get_logger(), "Speed control is ENABLED");
        } else {
            RCLCPP_INFO(this->get_logger(), "Speed control is DISABLED");
        }
    }

    ~Go2CameraVelNode() {
        if (control_thread_.joinable()) {
            control_thread_.join();
        }
    }

private:
    void setup_camera(const std::string& interface_name, const std::string& address,
                     int port, int width, int height)
    {
        // -----------------------------
        // GStreamer Pipeline
        // -----------------------------
        std::string pipeline =
            "udpsrc address=" + address + " port=" + std::to_string(port) +
            " multicast-iface=" + interface_name + " ! "
            "application/x-rtp, media=video, encoding-name=H264 ! "
            "rtph264depay ! h264parse ! avdec_h264 ! "
            "videoconvert ! video/x-raw,width=" + std::to_string(width) +
            ",height=" + std::to_string(height) + ",format=BGR ! "
            "appsink drop=1";

        cap_.open(pipeline, cv::CAP_GSTREAMER);

        if (!cap_.isOpened()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open GStreamer VideoCapture");
            rclcpp::shutdown();
            return;
        }

        if (publish_compressed_) {
            compressed_pub_ = this->create_publisher<sensor_msgs::msg::CompressedImage>(
                camera_image_compressed_topic_, 15);
            RCLCPP_INFO(this->get_logger(), "Publishing CompressedImage on %s",
                        camera_image_compressed_topic_.c_str());
        } else {
            raw_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
                camera_image_raw_topic_, 15);
            RCLCPP_INFO(this->get_logger(), "Publishing raw Image on %s",
                        camera_image_raw_topic_.c_str());
        }

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(10),
            std::bind(&Go2CameraVelNode::timer_callback, this));
    }

    void setup_speed_control(int control_freq)
    {
        // 订阅 cmd_vel topic (configurable)
        vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            cmd_vel_topic_, 10,
            std::bind(&Go2CameraVelNode::set_vel, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "Subscribing to cmd_vel on %s",
                    cmd_vel_topic_.c_str());

        // 启动控制线程
        control_thread_ = std::thread([this, control_freq] {
            std::this_thread::sleep_for(500ms);  // 等待ROS启动
            RobotControl(control_freq);
        });
    }

    void timer_callback()
    {
        cv::Mat frame;
        if (!cap_.read(frame)) {
            RCLCPP_WARN(this->get_logger(), "Frame grab failed");
            return;
        }

        rclcpp::Time timestamp = this->get_clock()->now();

        if (publish_compressed_) {
            publish_compressed(frame, timestamp);
        } else {
            publish_raw(frame, timestamp);
        }
    }

    void set_vel(const geometry_msgs::msg::Twist::SharedPtr msg)
    {
        if (!enable_speed_control_) return;

        std::lock_guard<std::mutex> lock(vel_mutex_);

        // 限幅 using parameters
        vel_[0] = std::clamp(msg->linear.x, -max_linear_x_, max_linear_x_);
        vel_[1] = std::clamp(msg->linear.y, -max_linear_y_, max_linear_y_);
        vel_[2] = std::clamp(msg->angular.z, -max_angular_z_, max_angular_z_);

        last_cmd_time_ = this->now();  // 更新最后一次接收速度的时间

        RCLCPP_INFO(this->get_logger(),
                    "Received cmd_vel -> linear.x=%.2f linear.y=%.2f angular.z=%.2f",
                    vel_[0], vel_[1], vel_[2]);
    }

    void RobotControl(int frequency) {
        rclcpp::WallRate loop_rate(frequency); // configurable Hz

        while (rclcpp::ok()) {
            double vx, vy, vyaw;
            {
                std::lock_guard<std::mutex> lock(vel_mutex_);
                // 如果超时没有接收到新的速度，则置零
                if ((this->now() - last_cmd_time_) > rclcpp::Duration(timeout_ms_)) {
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

    void publish_raw(const cv::Mat &frame, const rclcpp::Time &ts)
    {
        if (raw_pub_ == nullptr) return;
        
        sensor_msgs::msg::Image msg;
        msg.header.stamp = ts;
        msg.header.frame_id = "camera_link";

        msg.height = frame.rows;
        msg.width = frame.cols;
        msg.encoding = "bgr8";
        msg.is_bigendian = false;
        msg.step = frame.cols * frame.elemSize();

        msg.data.assign(frame.data, frame.data + frame.total() * frame.elemSize());

        raw_pub_->publish(msg);
    }

    void publish_compressed(const cv::Mat &frame, const rclcpp::Time &ts)
    {
        if (compressed_pub_ == nullptr) return;
        
        sensor_msgs::msg::CompressedImage msg;
        msg.header.stamp = ts;
        msg.header.frame_id = "camera_link";
        msg.format = "jpeg";

        std::vector<uchar> buffer;
        cv::imencode(".jpg", frame, buffer, {cv::IMWRITE_JPEG_QUALITY, 80});

        msg.data = buffer;
        compressed_pub_->publish(msg);
    }

private:
    // Camera related variables
    cv::VideoCapture cap_;
    bool publish_compressed_;
    std::string camera_image_raw_topic_;
    std::string camera_image_compressed_topic_;

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr raw_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr compressed_pub_;

    // Speed control related variables
    bool enable_speed_control_;
    std::string cmd_vel_topic_;
    SportClient sport_client_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr vel_sub_;
    unitree_api::msg::Request req_;
    double vel_[3];
    std::mutex vel_mutex_;
    std::thread control_thread_;
    rclcpp::Time last_cmd_time_;
    std::chrono::milliseconds timeout_ms_;
    double max_linear_x_, max_linear_y_, max_angular_z_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Go2CameraVelNode>());
    rclcpp::shutdown();
    return 0;
}