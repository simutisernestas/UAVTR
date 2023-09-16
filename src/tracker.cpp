#ifndef TRACKER_H

#include <opencv2/tracking.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <atomic>
#include <thread>
#include <cassert>
#include <numeric>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <string.h>
#include <boost/lockfree/spsc_queue.hpp>

template <typename T>
T vectorProduct(const std::vector<T> &v)
{
    return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

/**
 * @brief Operator overloading for printing vectors
 * @tparam T
 * @param os
 * @param v
 * @return std::ostream&
 */
template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &v)
{
    os << "[";
    for (size_t i = 0; i < v.size(); ++i)
    {
        os << v[i];
        if (i != v.size() - 1)
        {
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
                         const ONNXTensorElementDataType &type)
{
    switch (type)
    {
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

struct BoundingBox
{
    float x_c, y_c, w, h;
};

std::vector<float> box_cxcywh_to_xyxy(const BoundingBox &bb)
{
    float x_c = bb.x_c;
    float y_c = bb.y_c;
    float w = bb.w;
    float h = bb.h;
    std::vector<float> b = {(x_c - 0.5f * w), (y_c - 0.5f * h),
                            (x_c + 0.5f * w), (y_c + 0.5f * h)};
    return b;
}

std::vector<float> rescale_bboxes(const BoundingBox &out_bbox, const std::array<int, 2> &size)
{
    int img_w = size[0];
    int img_h = size[1];

    std::vector<float> b = box_cxcywh_to_xyxy(out_bbox);

    b[0] *= img_w;
    b[1] *= img_h;
    b[2] *= img_w;
    b[3] *= img_h;

    return b;
}

class ObjDetertor
{
public:
    ObjDetertor();
    ~ObjDetertor()
    {
        // deallocate memory in _input_names and _output_names
        free((void *)_input_names[0]);
        free((void *)_output_names[0]);
        free((void *)_output_names[1]);
    }

    // return true if high confidence detection is made
    bool detect(const cv::Mat &frame);

    inline void get_latest_bbox(std::pair<uint64_t, cv::Rect> &frame_id_bbox)
    {
        frame_id_bbox = _frame_id_bbox;
    }

private:
    // TODO: convert this to only bbox
    std::pair<uint64_t, cv::Rect> _frame_id_bbox{};

    Ort::Env _env;
    std::unique_ptr<Ort::Session> _session;
    Ort::AllocatorWithDefaultOptions _allocator;
    std::unique_ptr<Ort::MemoryInfo> _memory_info;
    std::vector<float> _input_image_values;
    std::vector<float> _out_logit_values;
    std::vector<float> _out_bbox_values;
    std::vector<int64_t> _input_dims;
    std::vector<int64_t> _out_logits_dims;
    std::vector<int64_t> _out_bbox_dims;
    std::vector<const char *> _input_names;
    std::vector<const char *> _output_names;
    std::vector<Ort::Value> _input_tensors;
    std::vector<Ort::Value> _output_tensors;
};

ObjDetertor::ObjDetertor()
{
    // initialize ONNX environment and session
    _env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "DETR-inference");
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    sessionOptions.SetIntraOpNumThreads(4);
    _session = std::make_unique<Ort::Session>(_env, "../detr.onnx", sessionOptions);

    // prepare input
    auto inputName = _session->GetInputNameAllocated(0, _allocator);
    Ort::TypeInfo inputTypeInfo = _session->GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    _input_dims = inputTensorInfo.GetShape();
    size_t inputTensorSize = vectorProduct(_input_dims);
    _input_image_values.reserve(inputTensorSize);

    // logits output
    auto outputName0 = _session->GetOutputNameAllocated(0, _allocator);
    Ort::TypeInfo outputTypeInfo = _session->GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
    _out_logits_dims = outputTensorInfo.GetShape();
    size_t logitsTensorSize = vectorProduct(_out_logits_dims);
    _out_logit_values.reserve(logitsTensorSize);

    // bbox output
    auto outputName1 = _session->GetOutputNameAllocated(1, _allocator);
    Ort::TypeInfo outputTypeInfo1 = _session->GetOutputTypeInfo(1);
    auto outputTensorInfo1 = outputTypeInfo1.GetTensorTypeAndShapeInfo();
    _out_bbox_dims = outputTensorInfo1.GetShape();
    size_t bboxTensorSize = vectorProduct(_out_bbox_dims);
    _out_bbox_values.reserve(bboxTensorSize);

    size_t input_name_len = strlen(inputName.get());
    size_t output_name_len0 = strlen(outputName0.get());
    size_t output_name_len1 = strlen(outputName1.get());

    char *input_name = (char *)malloc(sizeof(char) * input_name_len + 1);
    char *output_name0 = (char *)malloc(sizeof(char) * output_name_len0 + 1);
    char *output_name1 = (char *)malloc(sizeof(char) * output_name_len1 + 1);

    // copy from original
    strcpy(input_name, inputName.get());
    strcpy(output_name0, outputName0.get());
    strcpy(output_name1, outputName1.get());

    _input_names = {input_name};
    _output_names = {output_name0, output_name1};

    _memory_info = std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault));

    _input_tensors.push_back(Ort::Value::CreateTensor<float>(
        *_memory_info, _input_image_values.data(), inputTensorSize,
        _input_dims.data(), _input_dims.size()));
    _output_tensors.push_back(Ort::Value::CreateTensor<float>(
        *_memory_info, _out_logit_values.data(), logitsTensorSize,
        _out_logits_dims.data(), _out_logits_dims.size()));
    _output_tensors.push_back(Ort::Value::CreateTensor<float>(
        *_memory_info, _out_bbox_values.data(), bboxTensorSize,
        _out_bbox_dims.data(), _out_bbox_dims.size()));
}

bool ObjDetertor::detect(const cv::Mat &frame)
{
    uint64_t frame_id = reinterpret_cast<uint64_t>(&frame);
    cv::Rect bbox;

    auto image = frame.clone();
    cv::resize(image, image,
               cv::Size(_input_dims.at(3), _input_dims.at(2)),
               cv::InterpolationFlags::INTER_LINEAR);
    cv::Scalar mean(0.485, 0.456, 0.406);
    cv::Mat blob = cv::dnn::blobFromImage(image, 1.0 / 255.0,
                                          cv::Size(_input_dims.at(3), _input_dims.at(2)),
                                          mean, true);
    cv::Scalar std(0.229, 0.224, 0.225);
    blob /= std;

    std::copy(blob.begin<float>(),
              blob.end<float>(),
              _input_image_values.begin());

    _session->Run(Ort::RunOptions{nullptr}, _input_names.data(),
                  _input_tensors.data(), 1 /*Number of inputs*/, _output_names.data(),
                  _output_tensors.data(), 2 /*Number of outputs*/);

    float max_confidence = 0.0f;
    size_t good_bbox_id;
    for (size_t i = 0; i < 100; i++)
    {
        float rowmax = *std::max_element(
            _out_logit_values.begin(), _out_logit_values.begin() + 92);
        std::vector<float> y(92);
        float sum = 0.0f;
        auto input = _out_logit_values.begin() + i * 92;
        for (size_t j = 0; j != 92; ++j)
            sum += y[j] = std::exp(input[j] - rowmax);
        size_t max_id = 0;
        float confidence = 0.0f;
        for (size_t j = 0; j != 92; ++j)
        {
            input[j] = y[j] / sum;
            if (input[j] > confidence)
            {
                confidence = input[j];
                max_id = j;
            }
        }
        assert(confidence <= 1.0f);
        assert(confidence >= 0.0f);
        if (confidence >= max_confidence && max_id == 9)
        {
            good_bbox_id = i;
            max_confidence = confidence;
        }
    }

    if (max_confidence < 0.7f)
        return false;

    // TODO: remove this struct
    BoundingBox out_bbox;
    out_bbox.x_c = _out_bbox_values[good_bbox_id * 4 + 0];
    out_bbox.y_c = _out_bbox_values[good_bbox_id * 4 + 1];
    out_bbox.w = _out_bbox_values[good_bbox_id * 4 + 2];
    out_bbox.h = _out_bbox_values[good_bbox_id * 4 + 3];
    auto bb = rescale_bboxes(out_bbox, {frame.cols, frame.rows});

    bbox.x = bb[0];
    bbox.y = bb[1];
    bbox.width = (bb[2] - bb[0]) * 0.5f;
    bbox.height = (bb[3] - bb[1]) * 0.5f;

    // save it for use in tracker
    _frame_id_bbox = std::make_pair(frame_id, bbox);
    return true;
}

class Tracker
{

public:
    Tracker();
    ~Tracker()
    {
        exit = true;
        _obj_detector_thread.join();
    }
    bool init(const cv::Mat &frame, cv::Rect &bbox, double timeout = 5.0);
    bool process(const cv::Mat &frame, cv::Rect &bbox);
    void reinit(const cv::Mat &frame, const cv::Rect &bbox);
    void catchup_reinit();

private:
    cv::Ptr<cv::Tracker> _tracker = nullptr;
    std::unique_ptr<ObjDetertor> _detector = nullptr;
    boost::lockfree::spsc_queue<cv::Mat, boost::lockfree::capacity<100>> _frames;
    std::thread _obj_detector_thread;
    bool exit = false;
    std::atomic<bool> _allowed_to_swap;
};

Tracker::Tracker()
{
    std::atomic_init(&_allowed_to_swap, true);
    _detector = std::make_unique<ObjDetertor>();
    _obj_detector_thread = std::thread([this]()
                                       {
        while (true)
        {
            if (exit) 
                return;

            if (_frames.empty())
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            auto frame = _frames.front().clone();

            // this will take long time and will need to catchup with new entries
            bool success = _detector->detect(_frames.front());
            if (success) {
                this->catchup_reinit();
            }
        } });
}

bool Tracker::process(const cv::Mat &frame, cv::Rect &bbox)
{
    _frames.push(frame);

    if (_tracker == nullptr)
        return false;

    _allowed_to_swap = false;
    bool located = _tracker->update(frame, bbox);
    if (!located)
        _tracker.reset();
    _allowed_to_swap = true;

    return located;
}

void Tracker::catchup_reinit()
{
    std::pair<uint64_t, cv::Rect> frame_id_bbox;
    _detector->get_latest_bbox(frame_id_bbox);

    auto params = cv::TrackerKCF::Params();
    params.resize = true;
    auto local_tracker = cv::TrackerKCF::create(params);
    local_tracker->init(_frames.front(), frame_id_bbox.second);
    _frames.pop();
    while (!_frames.empty())
    {
        bool track = local_tracker->update(_frames.front(), frame_id_bbox.second);
        if (!track)
            return;

        _frames.pop();
    }
    while (true)
    {
        if (_allowed_to_swap)
        {
            _tracker = local_tracker;
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

#endif // TRACKER_H