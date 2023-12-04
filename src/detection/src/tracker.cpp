#ifndef TRACKER_H

#include <opencv2/tracking.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <atomic>
#include <thread>
#include <cassert>
#include <numeric>
#include <onnxruntime_cxx_api.h>
#include <string.h>
#include <boost/lockfree/spsc_queue.hpp>

#define MODEL 1 // 0 - detr; 1 - yolo
#define DETR_LOGITS_INDEX 0
#define DETR_BBOX_INDEX 1

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

std::vector<Result> postprocess(const cv::Size &originalImageSize, std::vector<Ort::Value> &outputTensors) {
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

        // Coords should be scaled to the original image. 
        // The coords from the model are relative to the model's input height and width.
        x1 = (x1 / model_input_width) * originalImageSize.width;
        x2 = (x2 / model_input_width) * originalImageSize.width;
        y1 = (y1 / model_input_height) * originalImageSize.height;
        y2 = (y2 / model_input_height) * originalImageSize.height;

        Result result(x1, x2, y1, y2, classPrediction, accuracy);

        resultVector.push_back(result);
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

template<typename T>
T vectorProduct(const std::vector<T> &v) {
    return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

/**
 * @brief Operator overloading for printing vectors
 * @tparam T
 * @param os
 * @param v
 * @return std::ostream&
 */
template<typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &v) {
    os << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        os << v[i];
        if (i != v.size() - 1) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

/**
 * @brief Print ONNX tensor data type
 * https://github.com/microsoft/onnxruntime/blob/rel-1.6.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L93
 * @param os
 * @param type
 * @return std::ostream&
 */
std::ostream &operator<<(std::ostream &os,
                         const ONNXTensorElementDataType &type) {
    switch (type) {
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
            os << "undefined";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            os << "float";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            os << "uint8_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            os << "int8_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            os << "uint16_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            os << "int16_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            os << "int32_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            os << "int64_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
            os << "std::string";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            os << "bool";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            os << "float16";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            os << "double";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
            os << "uint32_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
            os << "uint64_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
            os << "float real + float imaginary";
            break;
        case ONNXTensorElementDataType::
            ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
            os << "double real + float imaginary";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
            os << "bfloat16";
            break;
        default:
            break;
    }

    return os;
}

struct BoundingBox {
    float x_c, y_c, w, h;
};

std::vector<float> box_cxcywh_to_xyxy(const BoundingBox &bb) {
    float x_c = bb.x_c;
    float y_c = bb.y_c;
    float w = bb.w;
    float h = bb.h;
    std::vector<float> b = {(x_c - 0.5f * w), (y_c - 0.5f * h),
                            (x_c + 0.5f * w), (y_c + 0.5f * h)};
    return b;
}

std::vector<float> rescale_bboxes(const BoundingBox &out_bbox, const std::array<int, 2> &size) {
    int img_w = size[0];
    int img_h = size[1];

    std::vector<float> b = box_cxcywh_to_xyxy(out_bbox);

    b[0] *= img_w;
    b[1] *= img_h;
    b[2] *= img_w;
    b[3] *= img_h;

    return b;
}

class ObjDetertor {
public:
    ObjDetertor();

    ~ObjDetertor() {
        // deallocate memory in _input_names and _output_names
        free((void *) _input_names[0]);
        free((void *) _output_names[0]);
        free((void *) _output_names[1]);
    }

    // return true if high confidence detection is made
    bool detect(const cv::Mat &frame);

    inline void get_latest_bbox(cv::Rect &bbox) {
        bbox = _latest_bbox;
    }

    inline void get_points(std::array<cv::Point, 2> &points) {
        points = _latest_bbox_points;
    }

private:
    cv::Rect _latest_bbox{};
    std::array<cv::Point, 2> _latest_bbox_points{};

    Ort::Env _env;
    std::unique_ptr<Ort::Session> _session;
    Ort::AllocatorWithDefaultOptions _allocator;
    std::unique_ptr<Ort::MemoryInfo> _memory_info;

    std::vector<int64_t> _input_dims;
    std::vector<const char *> _input_names;
    std::vector<float> _input_image_values;
    std::vector<Ort::Value> _input_tensors;

    std::vector<std::vector<int64_t>> _output_dims;
    std::vector<const char *> _output_names;
    std::vector<std::vector<float>> _output_values;
    std::vector<Ort::Value> _output_tensors;
};

ObjDetertor::ObjDetertor() {
    auto providers =
            Ort::GetAvailableProviders();
    for (auto &&provider: providers) {(void)provider;}

    // initialize ONNX environment and session
    _env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "inference-engine");
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    sessionOptions.SetIntraOpNumThreads(4);
#if MODEL == 0
    _session = std::make_unique<Ort::Session>(_env, "../weights/detr.onnx", sessionOptions);
#else
    _session = std::make_unique<Ort::Session>(_env, "../weights/yolov7.onnx", sessionOptions);
#endif
    _memory_info = std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault));

    // prepare input
    auto inputName = _session->GetInputNameAllocated(0, _allocator);
    Ort::TypeInfo inputTypeInfo = _session->GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    _input_dims = inputTensorInfo.GetShape();
    model_input_height = _input_dims.at(3);
    model_input_width = _input_dims.at(2);
    if (_input_dims.at(0) == -1)
        _input_dims.at(0) = 1;
    size_t inputTensorSize = vectorProduct(_input_dims);
    _input_image_values.resize(inputTensorSize);
    _input_tensors.push_back(Ort::Value::CreateTensor<float>(
            *_memory_info, _input_image_values.data(), inputTensorSize,
            _input_dims.data(), _input_dims.size()));
    size_t input_name_len = strlen(inputName.get());
    char *input_name = (char *) malloc(sizeof(char) * input_name_len + 1);
    strcpy(input_name, inputName.get());
    _input_names = {input_name};

    // prepare output
    size_t num_output_nodes = _session->GetOutputCount();
    for (size_t i = 0; i < num_output_nodes; ++i) {
        auto outputName = _session->GetOutputNameAllocated(i, _allocator);
        Ort::TypeInfo outputTypeInfo = _session->GetOutputTypeInfo(i);
        auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
        auto output_dims = outputTensorInfo.GetShape();
        if (output_dims.at(0) == -1)
            output_dims.at(0) = 1;
        _output_dims.push_back(output_dims);
        size_t outputTensorSize = vectorProduct(output_dims);
        _output_values.push_back(std::vector<float>(outputTensorSize));
        _output_tensors.push_back(Ort::Value::CreateTensor<float>(
                *_memory_info, _output_values[i].data(), outputTensorSize,
                _output_dims[i].data(), _output_dims[i].size()));
        size_t output_name_len = strlen(outputName.get());
        char *output_name = (char *) malloc(sizeof(char) * output_name_len + 1);
        strcpy(output_name, outputName.get());
        _output_names.push_back(output_name);
    }
}

bool ObjDetertor::detect(const cv::Mat &frame) {
    cv::Rect2f bbox{};

#if MODEL == 0
    // refactor using less local variables
    auto image = frame.clone();
    cv::resize(image, image, cv::Size(_input_dims.at(3), _input_dims.at(2)),
               cv::InterpolationFlags::INTER_CUBIC);
    cv::cvtColor(image, image, cv::ColorConversionCodes::COLOR_BGR2RGB);
    image.convertTo(image, CV_32F, 1.0 / 255);
    cv::Mat channels2[3];
    cv::split(image, channels2);
    channels2[0] = (channels2[0] - 0.485) / 0.229;
    channels2[1] = (channels2[1] - 0.456) / 0.224;
    channels2[2] = (channels2[2] - 0.406) / 0.225;
    cv::merge(channels2, 3, image);
    cv::dnn::blobFromImage(image, image);

    std::copy(image.begin<float>(),
              image.end<float>(),
              _input_image_values.begin());

    _session->Run(Ort::RunOptions{nullptr}, _input_names.data(),
                  _input_tensors.data(), 1 /*Number of inputs*/, _output_names.data(),
                  _output_tensors.data(), 2 /*Number of outputs*/);

    static constexpr size_t BOAT_CLASS_ID = 9;
    static constexpr size_t N_CLASSES = 92;

    float max_confidence = 0.0f;
    size_t good_bbox_id;
    for (size_t i = 0; i < 100; i++)
    {
        float rowmax = *std::max_element(
            _output_values[DETR_LOGITS_INDEX].begin() + N_CLASSES * i,
            _output_values[DETR_LOGITS_INDEX].begin() + N_CLASSES * i + N_CLASSES);
        std::vector<float> y(N_CLASSES);
        y.assign(N_CLASSES, 0.0f);
        float sum = 0.0f;
        auto input = _output_values[DETR_LOGITS_INDEX].begin() + i * N_CLASSES;
        for (size_t j = 0; j < N_CLASSES; ++j)
            sum += y[j] = std::exp(input[j] - rowmax);
        // for all classes
        // for (size_t j = 0; j < N_CLASSES; ++j)
        //     input[j] = y[j] / sum;

        // only boat confidence level : )
        input[BOAT_CLASS_ID] = y[BOAT_CLASS_ID] / sum;
        float confidence = input[BOAT_CLASS_ID];

        assert(confidence <= 1.0f);
        assert(confidence >= 0.0f);
        if (confidence >= max_confidence)
        {
            good_bbox_id = i;
            max_confidence = confidence;
        }
    }

    if (max_confidence < 0.95f)
        return false;

    BoundingBox out_bbox;
    out_bbox.x_c = _output_values[DETR_BBOX_INDEX][good_bbox_id * 4 + 0];
    out_bbox.y_c = _output_values[DETR_BBOX_INDEX][good_bbox_id * 4 + 1];
    out_bbox.w = _output_values[DETR_BBOX_INDEX][good_bbox_id * 4 + 2];
    out_bbox.h = _output_values[DETR_BBOX_INDEX][good_bbox_id * 4 + 3];

    auto bb = rescale_bboxes(out_bbox, {frame.cols, frame.rows});

    bbox.x = bb[0];
    bbox.y = bb[1];
    bbox.width = (bb[2] - bb[0]) * 0.75f;
    bbox.height = (bb[3] - bb[1]) * 0.75f;

    // impose bound on width and height
    bbox.width = std::min(bbox.width, 100.0f);
    bbox.width = std::max(bbox.width, 10.0f);
    bbox.height = std::min(bbox.height, 100.0f);
    bbox.height = std::max(bbox.height, 10.0f);

#else
    auto image = frame.clone();
    cv::Mat blob_image = preprocess(image);

    std::copy(blob_image.begin<float>(),
              blob_image.end<float>(),
              _input_image_values.begin());

    std::vector<Ort::Value> outputTensors = _session->Run(Ort::RunOptions{nullptr},
                                                          _input_names.data(), _input_tensors.data(),
                                                          _input_names.size(),
                                                          _output_names.data(), _output_names.size());

    std::vector<Result> resultVector = postprocess(frame.size(), outputTensors);

    bool found = false;
    float max_accuracy = 0.0f;
    for (const auto &result: resultVector) {
        if (classNames.at(result.obj_id) != "boat")
            continue;
        if (result.accuracy < 0.6f && result.accuracy < max_accuracy)
            continue;
        found = true;
        max_accuracy = result.accuracy;

        cv::Point p1(result.x1, result.y1);
        cv::Point p2(result.x2, result.y2);
        bbox.x = result.x1;
        bbox.y = result.y1;
        bbox.width = (result.x2 - result.x1);
        bbox.height = (result.y2 - result.y1);
    }

    if (!found)
        return false;

#endif

    // save it for use in tracker
    _latest_bbox = bbox;
    return true;
}

class Tracker {

public:
    Tracker();

    ~Tracker() {
        exit = true;
        _obj_detector_thread.join();
    }

    bool process(const cv::Mat &frame, cv::Rect &bbox);

    void catchup_reinit();

    // not safe, use with caution : )
    void hard_reset_bbox(const cv::Rect &bbox);

private:
    cv::Ptr<cv::Tracker> _tracker = nullptr;
    std::unique_ptr<ObjDetertor> _detector = nullptr;
    // this is ~30 MB of data in memory
    boost::lockfree::spsc_queue<cv::Mat, boost::lockfree::capacity<100>> _frames;
    std::thread _obj_detector_thread;
    bool exit = false;
    std::atomic<bool> _allowed_to_swap;
    uint8_t failure_count_ = 0;
};

Tracker::Tracker() {
    std::atomic_init(&_allowed_to_swap, true);
    _detector = std::make_unique<ObjDetertor>();
    _obj_detector_thread = std::thread([this]() {
        while (true) {
            if (exit)
                return;

            if (_frames.empty())
                continue;

            // this will take long time and will need to catchup with new entries
            bool success = _detector->detect(_frames.front());
            if (success) {
                this->catchup_reinit();
                // return;
            } else {
                while (_frames.pop()) {}
            }
        }
    });
}

bool Tracker::process(const cv::Mat &frame, cv::Rect &bbox) {
    _frames.push(frame);

    if (_tracker == nullptr)
        return false;

    _allowed_to_swap = false;
    bool located = _tracker->update(frame, bbox);
    if (!located)
        ++failure_count_;
    if (failure_count_ > 10) {
        _tracker.reset();
        failure_count_ = 0;
    }
    _allowed_to_swap = true;

    return located;
}

// not safe, use with caution : )
void Tracker::hard_reset_bbox(const cv::Rect &bbox) {
    auto params = cv::TrackerKCF::Params();
    params.resize = true;
    params.detect_thresh = 0.7f;
    auto local_tracker = cv::TrackerKCF::create(params);
    local_tracker->init(_frames.front(), bbox);
    while (true) {
        if (_allowed_to_swap) {
            _tracker = local_tracker;
            return;
        }
    }
}

void Tracker::catchup_reinit() {
    cv::Rect bbox;
    _detector->get_latest_bbox(bbox);

    auto params = cv::TrackerKCF::Params();
    params.resize = true;
    params.detect_thresh = 0.7f;
    auto local_tracker = cv::TrackerKCF::create(params);
    local_tracker->init(_frames.front(), bbox);
    _frames.pop();
    while (_frames.read_available() > 1) {
        bool track = local_tracker->update(_frames.front(), bbox);
        if (!track)
            return;

        _frames.pop();
    }
    while (true) {
        if (_allowed_to_swap) {
            _tracker = local_tracker;
            failure_count_ = 0;
            return;
        }
    }
}

#endif // TRACKER_H