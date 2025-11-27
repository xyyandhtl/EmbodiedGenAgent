#include <opencv2/opencv.hpp>

int main(void)
{
    std::cout << cv::getBuildInformation() << std::endl;

    std::string pipeline =
        "udpsrc address=230.1.1.1 port=1720 multicast-iface=<interface_name> ! "
        "application/x-rtp, media=video, encoding-name=H264 ! "
        "rtph264depay ! "
        "h264parse ! "
        "avdec_h264 ! "
        "videoconvert ! "
        "video/x-raw,width=1280,height=720,format=BGR ! "
        "appsink drop=1";

    cv::VideoCapture cap(
        pipeline,
        cv::CAP_GSTREAMER
    );

    if (!cap.isOpened()) {
        std::cerr << "VideoCapture not opened" << std::endl;
        std::exit(-1);
    }

    while (true) {

        cv::Mat frame;

        cap.read(frame);

        cv::imshow("receiver", frame);

        if (cv::waitKey(1) == 27) {
            break;
        }
    }

    return 0;
}
