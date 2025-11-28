#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <thread>

int main(void)
{
    std::cout << "=== OpenCV Build Information ===" << std::endl;
    std::cout << cv::getBuildInformation() << std::endl;
    std::cout << "=================================" << std::endl;

    std::string pipeline =
        "udpsrc address=230.1.1.1 port=1720 multicast-iface=eth0 ! "
        "application/x-rtp, media=video, encoding-name=H264 ! "
        "rtph264depay ! "
        "h264parse ! "
        "avdec_h264 ! "
        "videoconvert ! "
        "video/x-raw,width=1280,height=720,format=BGR ! "
        "appsink drop=1";

    std::cout << "[INFO] GStreamer pipeline: " << pipeline << std::endl;
    std::cout << "[INFO] Creating VideoCapture with GStreamer backend..." << std::endl;

    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);

    if (!cap.isOpened()) {
        std::cerr << "[ERROR] VideoCapture not opened" << std::endl;
        std::cerr << "[ERROR] Please check GStreamer pipeline and network configuration" << std::endl;
        std::exit(-1);
    }

    std::cout << "[INFO] VideoCapture opened successfully" << std::endl;

    // 创建保存图片的目录
    std::string save_dir = "received_images";
    system(("mkdir -p " + save_dir).c_str());
    std::cout << "[INFO] Images will be saved to: " << save_dir << std::endl;

    int frame_count = 0;
    auto start_time = std::chrono::steady_clock::now();
    auto last_log_time = start_time;

    std::cout << "[INFO] Starting to receive frames..." << std::endl;

    while (true) {
        cv::Mat frame;
        
        auto read_start = std::chrono::steady_clock::now();
        bool success = cap.read(frame);
        auto read_end = std::chrono::steady_clock::now();
        auto read_duration = std::chrono::duration_cast<std::chrono::milliseconds>(read_end - read_start);

        if (!success) {
            std::cerr << "[ERROR] Failed to read frame at count: " << frame_count << std::endl;
            continue;
        }

        if (frame.empty()) {
            std::cerr << "[WARNING] Received empty frame at count: " << frame_count << std::endl;
            continue;
        }

        frame_count++;

        // 生成带时间戳的文件名
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        
        std::stringstream filename;
        filename << save_dir << "/frame_" 
                 << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S")
                 << "_" << std::setfill('0') << std::setw(3) << ms.count()
                 << "_" << std::setfill('0') << std::setw(6) << frame_count
                 << ".jpg";
        
        // 保存图片
        auto save_start = std::chrono::steady_clock::now();
        bool save_success = cv::imwrite(filename.str(), frame);
        auto save_end = std::chrono::steady_clock::now();
        auto save_duration = std::chrono::duration_cast<std::chrono::milliseconds>(save_end - save_start);

        if (!save_success) {
            std::cerr << "[ERROR] Failed to save image: " << filename.str() << std::endl;
        }

        // 定期输出日志（每30帧或每5秒）
        auto current_time = std::chrono::steady_clock::now();
        auto time_since_last_log = std::chrono::duration_cast<std::chrono::seconds>(current_time - last_log_time);
        
        if (frame_count % 30 == 0 || time_since_last_log.count() >= 5) {
            auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time);
            double fps = (total_duration.count() > 0) ? frame_count / (double)total_duration.count() : 0;
            
            std::cout << "[INFO] Frame: " << frame_count 
                      << " | Size: " << frame.cols << "x" << frame.rows 
                      << " | Read time: " << read_duration.count() << "ms"
                      << " | Save time: " << save_duration.count() << "ms"
                      << " | FPS: " << std::fixed << std::setprecision(2) << fps
                      << " | Latest file: " << filename.str() << std::endl;
            
            last_log_time = current_time;
        }

        // 添加延迟控制，避免保存太快
        std::this_thread::sleep_for(std::chrono::milliseconds(33)); // 约30fps

        // 退出条件：可以设置最大帧数或超时时间
        if (frame_count >= 1000) { // 最多保存1000帧
            std::cout << "[INFO] Reached maximum frame count (1000), exiting..." << std::endl;
            break;
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    double avg_fps = (total_duration.count() > 0) ? frame_count / (double)total_duration.count() : 0;
    
    std::cout << "[INFO] Program finished" << std::endl;
    std::cout << "[INFO] Total frames processed: " << frame_count << std::endl;
    std::cout << "[INFO] Total time: " << total_duration.count() << " seconds" << std::endl;
    std::cout << "[INFO] Average FPS: " << avg_fps << std::endl;
    std::cout << "[INFO] Images saved in: " << save_dir << std::endl;

    return 0;
}