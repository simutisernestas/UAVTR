#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <chrono>

typedef struct Result {
    int x1;
    int x2;
    int y1;
    int y2;
    int obj_id;
    float accuracy;

    Result(int x1_, int x2_, int y1_, int y2_, int obj_id_, float accuracy_) {
        x1 = x1_;
        x2 = x2_;
        y1 = y1_;
        y2 = y2_;
        obj_id = obj_id_;
        accuracy = accuracy_;
    }

} result_t;

int model_input_width;
int model_input_height;

// Class names for YOLOv7
std::array<const std::string, 80> classNames = {
        "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "sofa", "potted plant", "bed", "dining table", "toilet", "tv monitor", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

cv::Mat preprocess(cv::Mat &image) {
    // Channels order: BGR to RGB
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size(model_input_width, model_input_height));

    // Convert image to float32 and normalize
    cv::Mat floatImage;
    resizedImage.convertTo(floatImage, CV_32F, 1.0 / 255.0);

    // Create a 4-dimensional blob from the image
    cv::Mat blobImage = cv::dnn::blobFromImage(floatImage);

    return blobImage;
}

std::vector<Result> postprocess(cv::Size originalImageSize, std::vector<Ort::Value> &outputTensors) {
    auto *rawOutput = outputTensors[0].GetTensorData<float>();
    std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t count = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    std::vector<float> output(rawOutput, rawOutput + count);

    std::vector<Result> resultVector;

    for (int i = 0; i < outputShape[0]; i++) {
        float confidence = output[i * outputShape[1] + 0];
        float x1 = output[i * outputShape[1] + 1];
        float y1 = output[i * outputShape[1] + 2];
        float x2 = output[i * outputShape[1] + 3];
        float y2 = output[i * outputShape[1] + 4];
        int classPrediction = output[i * outputShape[1] + 5];
        float accuracy = output[i * outputShape[1] + 6];

        (void) confidence;

        std::cout << "Class Name: " << classNames.at(classPrediction) << std::endl;
        std::cout << "Coords: Top Left (" << x1 << ", " << y1 << "), Bottom Right (" << x2 << ", " << y2 << ")"
                  << std::endl;
        std::cout << "Accuracy: " << accuracy << std::endl;

        // Coords should be scaled to the original image. The coords from the model are relative to the model's input height and width.
        x1 = (x1 / model_input_width) * originalImageSize.width;
        x2 = (x2 / model_input_width) * originalImageSize.width;
        y1 = (y1 / model_input_height) * originalImageSize.height;
        y2 = (y2 / model_input_height) * originalImageSize.height;

        Result result(x1, x2, y1, y2, classPrediction, accuracy);

        resultVector.push_back(result);

        std::cout << std::endl;
    }

    return resultVector;
}

void drawBoundingBox(cv::Mat &image, std::vector<Result> &resultVector) {

    for (auto result: resultVector) {

        if (result.accuracy > 0.6) { // Threshold, can be made function parameter

            cv::rectangle(image, cv::Point(result.x1, result.y1), cv::Point(result.x2, result.y2),
                          cv::Scalar(0, 255, 0), 2);

            cv::putText(image, classNames.at(result.obj_id),
                        cv::Point(result.x1, result.y1 - 3), cv::FONT_ITALIC,
                        0.8, cv::Scalar(255, 255, 255), 2);

            cv::putText(image, std::to_string(result.accuracy),
                        cv::Point(result.x1, result.y1 + 30), cv::FONT_ITALIC,
                        0.8, cv::Scalar(255, 255, 0), 2);
        }
    }
}

int main() {

    const char *model_path = "../yolov7.onnx";

    Ort::AllocatorWithDefaultOptions allocator;

    Ort::SessionOptions session_options;
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime");
    Ort::Session session(env, model_path, session_options);

    size_t num_input_nodes = session.GetInputCount();
    size_t num_output_nodes = session.GetOutputCount();

    std::vector<Ort::AllocatedStringPtr> input_node_names;
    std::vector<Ort::AllocatedStringPtr> output_node_names;

    std::vector<char *> input_node_names_char;
    std::vector<char *> output_node_names_char;

    for (size_t i = 0; i < num_input_nodes; ++i) {
        input_node_names.push_back(session.GetInputNameAllocated(i, allocator));
        input_node_names_char.push_back(input_node_names[i].get());
    }

    for (size_t i = 0; i < num_output_nodes; ++i) {
        output_node_names.push_back(session.GetOutputNameAllocated(i, allocator));
        output_node_names_char.push_back(output_node_names[i].get());
    }

    for (auto &input_name: input_node_names) {
        std::cout << "input node name   : " << input_name.get() << std::endl;
    }

    for (auto &output_name: output_node_names) {
        std::cout << "output node name  : " << output_name.get() << std::endl;
    }

    std::cout << std::endl;

    cv::Mat image = cv::imread("../boat1/000100.jpg");
    // resize
    cv::resize(image, image, cv::Size(), 0.75, 0.75);

    std::vector<int64_t> inputDims = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

    model_input_height = inputDims.at(3);
    model_input_width = inputDims.at(2);

    cv::Mat inputImage = preprocess(image);

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo,
                                                             inputImage.ptr<float>(),
                                                             inputImage.total() * sizeof(float),
                                                             inputDims.data(),
                                                             inputDims.size());

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<Ort::Value> outputTensors = session.Run(Ort::RunOptions{nullptr},
                                                        input_node_names_char.data(),
                                                        &inputTensor,
                                                        num_input_nodes,
                                                        output_node_names_char.data(),
                                                        num_output_nodes);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << "ms" << std::endl;

    std::vector<Result> resultVector = postprocess(image.size(), outputTensors);

    drawBoundingBox(image, resultVector);

    // Display the image with detections
    cv::imshow("Object Detection", image);
    cv::waitKey(0);

    // for (auto &ptr : input_node_names)
    //     allocator.Free(ptr);
    // for (auto &ptr : output_node_names)
    //     allocator.Free(ptr);

    return 0;
}
