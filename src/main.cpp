#include "tracker.cpp"
#include <opencv2/videoio.hpp>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <thread>
#include <numeric>
#include <atomic>

int main()
{
    Tracker tracker{};

    std::string imageFilepath{};
    size_t n_images = 300;
    // read all images up front
    std::vector<cv::Mat> images{};
    for (size_t i = 1; i < n_images; i += 1)
    {
        std::string zero_pad = std::string(6 - std::to_string(i).length(), '0');
        imageFilepath = "../boat1/" + zero_pad + std::to_string(i) + ".jpg";
        // std::cout << "Image: " << imageFilepath << std::endl;
        cv::Mat image = cv::imread(imageFilepath, cv::ImreadModes::IMREAD_COLOR);
        if (image.empty())
        {
            std::cout << "Failed to read image" << std::endl;
            return -1;
        }
        images.push_back(image);
    }

    double worst_comp_time{0.0};

    n_images = 300;
    for (size_t i = 0; i < n_images - 1; i += 1)
    {
        cv::Mat image = images[i];
        cv::Rect bbox{};
        auto start = std::chrono::system_clock::now();
        tracker.process(image, bbox);
        auto end = std::chrono::system_clock::now();
        worst_comp_time = std::max(worst_comp_time, std::chrono::duration<double>(end - start).count());
        std::cout << "Comp time: " << std::chrono::duration<double>(end - start).count() * 1000 << " ms" << '\n';

        std::this_thread::sleep_for(std::chrono::milliseconds(33));

        cv::rectangle(image, bbox, cv::Scalar(0, 255, 0), 2, 1);
        cv::imshow("BBOX", image);
        cv::waitKey(1);
    }
    std::cout << "Worst comp time: " << worst_comp_time * 1000 << " ms" << '\n';
    // must go lower than 30, potentially way more

    // double sleep = n_images * (1000 / 30) / 1000;
    // // percentage of time spent computing
    // double compute = (elapsed_seconds.count() - sleep) / sleep;
    // std::cout << "Compute: " << compute << std::endl;

    return 0;

    //     // Measure latency
    //     int numTests{10};
    //     std::chrono::steady_clock::time_point begin =
    //         std::chrono::steady_clock::now();
    //     for (int i = 0; i < numTests; i++)
    //     {
    //         session.Run(Ort::RunOptions{nullptr}, inputNames.data(),
    //                     inputTensors.data(), 1, outputNames.data(),
    //                     outputTensors.data(), 2);
    //     }
    //     std::chrono::steady_clock::time_point end =
    //         std::chrono::steady_clock::now();
    //     std::cout << "Minimum Inference Latency: "
    //               << std::chrono::duration_cast<std::chrono::milliseconds>(end -
    //                                                                        begin)
    //                          .count() /
    //                      static_cast<float>(numTests)
    //               << " ms" << std::endl;

    //     return 0;

    //     // set number of threads to 1
    //     cv::setNumThreads(2);

    //     cv::VideoCapture cap("../video.mp4");
    //     if (!cap.isOpened())
    //     {
    //         std::cout << "Error opening video file " << std::endl;
    //         return -1;
    //     }

    //     int n_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    //     std::cout << "Number of frames: " << n_frames << std::endl;
    //     double fps = cap.get(cv::CAP_PROP_FPS);
    //     std::cout << "FPS: " << fps << std::endl;

    //     Tracker tracker{};
    //     cv::Mat frame;
    //     cv::Rect bbox;
    //     bool ok = cap.read(frame);
    //     if (!ok)
    //     {
    //         std::cout << "Cannot read video file" << std::endl;
    //         return -1;
    //     }
    //     cv::resize(frame, frame, cv::Size(), 0.33, 0.33);

    //     int width = 100;
    //     int height = 100;
    //     int x = 651 - width / 2;
    //     int y = 300 - height / 2;

    //     bbox = {x, y, width, height};
    //     tracker.reinit(frame, bbox); // adds a new one

    //     // create a thread to fuzz the tracker
    //     std::thread t([&tracker, &frame, &bbox]()
    //                   {
    //             std::this_thread::sleep_for(std::chrono::nanoseconds(1000000));
    //         while (true)
    //         {
    // // This fails :) TODO: debug!
    // // terminate called after throwing an instance of 'cv::Exception'
    // //   what():  OpenCV(4.5.4) ./contrib/modules/tracking/src/trackerKCF.cpp:274: error: (-215:Assertion failed) !(roi & image_roi).empty() in function 'init'
    // // ./run.sh: line 1: 67569 Aborted                 (core dumped) nice -20 ./tracker_app
    //             std::this_thread::sleep_for(std::chrono::nanoseconds(100));
    //             tracker.reinit(frame, bbox);
    //         } });

    //     std::chrono::duration<double> elapsed_seconds{0.0};
    //     while (cap.read(frame))
    //     {
    //         cv::resize(frame, frame, cv::Size(), 0.33, 0.33);
    //         std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    //         tracker.process(frame, bbox);
    //         std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    //         elapsed_seconds += end - begin;
    //         cv::rectangle(frame, bbox, cv::Scalar(255, 0, 0), 2, 1);
    //         cv::imshow("Tracking", frame);
    //         cv::waitKey(1);
    //     }
    //     std::cout << "Average time: " << (elapsed_seconds.count() / n_frames) * 1000 << " ms" << std::endl;

    //     t.join();

    return 0;
}