#include <iostream>
#include <chrono>
#include <string>

#define CONCAT_IMPL(a, b) a##b
#define CONCAT(a, b) CONCAT_IMPL(a, b)
#define FUNC_TIMER SPTimer CONCAT(timer_, __LINE__)(__func__);

class SPTimer {
public:
    explicit SPTimer(const std::string& name = "")
        : name_(name), running_(true), start_time_(std::chrono::high_resolution_clock::now()) {}

    void stop() {
        if (!running_) return;
        running_ = false;
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end_time - start_time_;
        std::cerr << "[" << name_ << "] use time: " << elapsed.count() << " ms" << std::endl;
    }

    ~SPTimer() {
        stop();
    }

private:
    std::string name_;
    bool running_;
    std::chrono::high_resolution_clock::time_point start_time_;
};

