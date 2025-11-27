#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

class UdpCamNode : public rclcpp::Node
{
public:
    UdpCamNode() : Node("udp_cam_node")
    {
        // -----------------------------
        // GStreamer Pipeline
        // -----------------------------
        std::string pipeline =
            "udpsrc address=230.1.1.1 port=1720 multicast-iface=<interface_name> ! "
            "application/x-rtp, media=video, encoding-name=H264 ! "
            "rtph264depay ! h264parse ! avdec_h264 ! "
            "videoconvert ! video/x-raw,width=1280,height=720,format=BGR ! "
            "appsink drop=1";

        cap_.open(pipeline, cv::CAP_GSTREAMER);

        if (!cap_.isOpened()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open GStreamer VideoCapture");
            rclcpp::shutdown();
            return;
        }

        // 判断是否是压缩流（简单规则：包含 H264 就认为是压缩）
        is_compressed_ = pipeline.find("H264") != std::string::npos;

        if (is_compressed_) {
            compressed_pub_ = this->create_publisher<sensor_msgs::msg::CompressedImage>(
                "/udp_cam/image/compressed", 10);
            RCLCPP_INFO(this->get_logger(), "Publishing CompressedImage");
        } else {
            raw_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
                "/udp_cam/image", 15);
            RCLCPP_INFO(this->get_logger(), "Publishing raw Image");
        }

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(10),
            std::bind(&UdpCamNode::timer_callback, this));
    }

private:
    void timer_callback()
    {
        cv::Mat frame;
        if (!cap_.read(frame)) {
            RCLCPP_WARN(this->get_logger(), "Frame grab failed");
            return;
        }

        rclcpp::Time timestamp = this->get_clock()->now();

        if (is_compressed_) {
            publish_compressed(frame, timestamp);
        } else {
            publish_raw(frame, timestamp);
        }
    }

    void publish_raw(const cv::Mat &frame, const rclcpp::Time &ts)
    {
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
    cv::VideoCapture cap_;
    bool is_compressed_;

    rclcpp::TimerBase::SharedPtr timer_;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr raw_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr compressed_pub_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<UdpCamNode>());
    rclcpp::shutdown();
    return 0;
}
