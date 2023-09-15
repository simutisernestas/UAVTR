#include "tracker.cpp"
#include <opencv2/videoio.hpp>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <thread>
#include <numeric>
#include <onnxruntime_cxx_api.h>


int main()
{
    cv::Mat m{};
    std::cout << sizeof(&m) << " " << &m << std::endl;
    return 0;


    // move this inference somewhere else  : )
    std::string modelFilepath{};
    std::string imageFilepath{"../boat1/000046.jpg"};
    std::string instanceName{"image-classification-inference"};

    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                 instanceName.c_str());
    Ort::SessionOptions sessionOptions;
    // sessionOptions.SetIntraOpNumThreads(4);
    sessionOptions.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
    sessionOptions.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);

    Ort::Session session(env, modelFilepath.c_str(), sessionOptions);

    Ort::AllocatorWithDefaultOptions allocator;

    size_t numInputNodes = session.GetInputCount();
    size_t numOutputNodes = session.GetOutputCount();

    auto inputName = session.GetInputNameAllocated(0, allocator);
    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
    std::vector<int64_t> inputDims = inputTensorInfo.GetShape();

    auto outputName0 = session.GetOutputNameAllocated(0, allocator);
    Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();
    std::vector<int64_t> outputDims = outputTensorInfo.GetShape();

    auto outputName1 = session.GetOutputNameAllocated(1, allocator);
    Ort::TypeInfo outputTypeInfo1 = session.GetOutputTypeInfo(1);
    auto outputTensorInfo1 = outputTypeInfo1.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType outputType1 = outputTensorInfo1.GetElementType();
    std::vector<int64_t> outputDims1 = outputTensorInfo1.GetShape();

    std::cout << "Number of Input Nodes: " << numInputNodes << std::endl;
    std::cout << "Number of Output Nodes: " << numOutputNodes << std::endl;
    std::cout << "Input Name: " << inputName.get() << std::endl;
    std::cout << "Input Type: " << inputType << std::endl;
    std::cout << "Input Dimensions: " << inputDims << std::endl;
    std::cout << "Output Name: " << outputName0.get() << std::endl;
    std::cout << "Output Type: " << outputType << std::endl;
    std::cout << "Output Dimensions: " << outputDims << std::endl;
    std::cout << "Output Name: " << outputName1.get() << std::endl;
    std::cout << "Output Type: " << outputType1 << std::endl;
    std::cout << "Output Dimensions: " << outputDims1 << std::endl;

    size_t n_images = 300;
    for (size_t i = 1; i < n_images; i += 10)
    {
        std::string zero_pad = std::string(6 - std::to_string(i).length(), '0');
        imageFilepath = "../boat2/" + zero_pad + std::to_string(i) + ".jpg";
        std::cout << "Image: " << imageFilepath << std::endl;

        cv::Mat image = cv::imread(imageFilepath, cv::ImreadModes::IMREAD_COLOR);
        cv::resize(image, image,
                   cv::Size(inputDims.at(3), inputDims.at(2)),
                   cv::InterpolationFlags::INTER_CUBIC);
        cv::Scalar mean(0.485, 0.456, 0.406); // TODO: this changing of RB in RGB here is weird
        cv::Mat blob = cv::dnn::blobFromImage(image, 1.0 / 255.0, cv::Size(inputDims.at(3), inputDims.at(2)), mean, true);
        cv::Scalar std(0.229, 0.224, 0.225);
        blob /= std;

        size_t inputTensorSize = vectorProduct(inputDims);
        std::vector<float> inputTensorValues(inputTensorSize);
        std::copy(blob.begin<float>(),
                  blob.end<float>(),
                  inputTensorValues.begin());

        size_t outputTensorSize = vectorProduct(outputDims);
        std::vector<float> outputTensorValues(outputTensorSize);
        size_t outputTensorSize1 = vectorProduct(outputDims1);
        std::vector<float> outputTensorValues1(outputTensorSize1);

        std::vector<const char *> inputNames{inputName.get()};
        std::vector<const char *> outputNames{outputName0.get(),
                                              outputName1.get()};
        std::vector<Ort::Value> inputTensors;
        std::vector<Ort::Value> outputTensors;

        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
            OrtDeviceAllocator, OrtMemTypeCPU);
        inputTensors.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo, inputTensorValues.data(), inputTensorSize, inputDims.data(),
            inputDims.size()));
        outputTensors.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo, outputTensorValues.data(), outputTensorSize,
            outputDims.data(), outputDims.size()));
        outputTensors.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo, outputTensorValues1.data(), outputTensorSize1,
            outputDims1.data(), outputDims1.size()));

        session.Run(Ort::RunOptions{nullptr}, inputNames.data(),
                    inputTensors.data(), 1 /*Number of inputs*/, outputNames.data(),
                    outputTensors.data(), 2 /*Number of outputs*/);
        std::cout << "Returned : )" << std::endl;

        std::cout << "Logits size: " << outputTensorValues.size() << std::endl;

        // outputTensorValues.begin(), outputTensorValues.begin() + 92
        // create a verctor from this
        std::vector<float> flog(92);
        std::copy(outputTensorValues.begin(), outputTensorValues.begin() + 92, flog.begin());

        std::cout << flog << std::endl;

        std::vector<size_t> good_bbox_ids;
        for (size_t i = 0; i < 100; i++)
        {
            float rowmax = *std::max_element(outputTensorValues.begin(), outputTensorValues.begin() + 92);
            std::vector<float> y(92);
            float sum = 0.0f;
            auto input = outputTensorValues.begin() + i * 92;
            for (size_t j = 0; j != 92; ++j)
                sum += y[j] = std::exp(input[j] - rowmax);
            size_t max_id = 0;
            float max_confidence = 0.0f;
            for (size_t j = 0; j != 92; ++j)
            {
                input[j] = y[j] / sum;
                if (input[j] > max_confidence)
                {
                    max_confidence = input[j];
                    max_id = j;
                }
            }
            assert(max_confidence <= 1.0f);
            assert(max_confidence >= 0.0f);
            if (max_confidence >= 0.7 && max_id == 9)
            {
                good_bbox_ids.push_back(i);
                std::cout << "Confidence: " << max_confidence << std::endl;
            }
        }

        std::cout << good_bbox_ids << std::endl;
        for (auto &box : good_bbox_ids)
        {
            // fetch first good bbox
            BoundingBox out_bbox;
            out_bbox.x_c = outputTensorValues1[box * 4 + 0];
            out_bbox.y_c = outputTensorValues1[box * 4 + 1];
            out_bbox.w = outputTensorValues1[box * 4 + 2];
            out_bbox.h = outputTensorValues1[box * 4 + 3];
            auto bb = rescale_bboxes(out_bbox, {image.cols, image.rows});
            std::cout << "Bbox: " << bb << std::endl;
            cv::rectangle(image, cv::Point(bb[0], bb[1]),
                          cv::Point(bb[2], bb[3]), cv::Scalar(0, 255, 0), 2, 1);
        }
        cv::imshow("BBOX", image);
        cv::waitKey(0);
    }

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


// bool Tracker::init(const cv::Mat &frame, cv::Rect &bbox /*empty bbox*/, double timeout /*= 5.0 s*/)
// {
//     _frames.clear();
//     _frames.push_back(frame);

//     auto start = std::chrono::steady_clock::now();
//     std::chrono::time_point<std::chrono::steady_clock> now;
//     double running_time;
//     while (1) // wait for detector to initialize the tracker
//     {
//         now = std::chrono::steady_clock::now();
//         running_time = std::chrono::duration_cast<std::chrono::seconds>(now - start).count();
//         if (running_time > timeout)
//             return false;

//         if (_tracker != nullptr)
//         {
//             std::pair<uint64_t, cv::Rect> frame_id_bbox;
//             _detector->get_latest_bbox(frame_id_bbox);
//             bbox = frame_id_bbox.second;
//             return true;
//         }
//     }
// }

// void Tracker::reinit(const cv::Mat &frame, const cv::Rect &bbox)
// {
//     std::lock_guard<std::mutex> lock(_mutex);
//     _tracker.reset();
//     _tracker = cv::TrackerKCF::create();
//     _tracker->init(frame, bbox);
// }