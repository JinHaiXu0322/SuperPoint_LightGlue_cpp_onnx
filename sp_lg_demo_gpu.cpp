#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <array>
#include <cstring>
#include <sstream>

// 这个版本是“精简版”Demo：
// - 假设模型 I/O 与你打印出来的一致：
//   SuperPoint: input="image", outputs={"keypoints","scores","descriptors"}
//   LightGlue : inputs={"kpts0","kpts1","desc0","desc1"}, outputs={"matches0","mscores0"}
// - 假设 keypoints 是 float32 且为像素坐标，shape 为 [N,2] 或 [1,N,2]
// - 假设 descriptors 是 float32，shape 为 [1,N,256]
// - 假设 matches0 是 int64，shape 为 [M,2]
// 删除了 sp_lg_demo.cpp 里的大量兼容判断、自动反归一化、以及 mapping 形式 matches0 的解析。

static std::string elementTypeToString(ONNXTensorElementDataType t) {
    switch (t) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED: return "UNDEFINED";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return "FLOAT";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return "UINT8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: return "INT8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: return "UINT16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: return "INT16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return "INT32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return "INT64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING: return "STRING";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return "BOOL";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return "FLOAT16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return "DOUBLE";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: return "UINT32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: return "UINT64";
        default: return "(unknown)";
    }
};

struct SPFeatures {
    std::vector<cv::KeyPoint> kpts;
    std::vector<float> desc; // [N*D]
    int64_t N = 0;
    int64_t D = 0;
};

struct MatchWithScore {
    int q = -1;
    int t = -1;
    float s = 0.f;
};

static std::vector<float> toFloatCHW01(const cv::Mat& gray, int H, int W) {
    cv::Mat r, f;
    cv::resize(gray, r, cv::Size(W, H), 0, 0, cv::INTER_LINEAR);
    r.convertTo(f, CV_32F, 1.0 / 255.0);
    std::vector<float> out((size_t)H * (size_t)W);
    std::memcpy(out.data(), f.ptr<float>(), out.size() * sizeof(float));
    return out;
}

static std::vector<cv::KeyPoint> parseKeypointsSimple(const Ort::Value& kp_tensor) {
    auto info = kp_tensor.GetTensorTypeAndShapeInfo();
    auto shp = info.GetShape();
    auto et = info.GetElementType();

    int64_t N = 0;
    if (shp.size() == 2 && shp[1] == 2) N = shp[0];
    else if (shp.size() == 3 && shp[0] == 1 && shp[2] == 2) N = shp[1];
    else throw std::runtime_error("Unexpected keypoints shape in simple demo.");

    std::vector<cv::KeyPoint> kpts;
    kpts.reserve((size_t)N);

    auto push_xy = [&](double x, double y) {
        kpts.emplace_back(cv::Point2f((float)x, (float)y), 2.f);
    };

    if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        const float* p = kp_tensor.GetTensorData<float>();
        for (int64_t i = 0; i < N; ++i) push_xy(p[(size_t)i * 2 + 0], p[(size_t)i * 2 + 1]);
    } else if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
        const double* p = kp_tensor.GetTensorData<double>();
        for (int64_t i = 0; i < N; ++i) push_xy(p[(size_t)i * 2 + 0], p[(size_t)i * 2 + 1]);
    } else if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        const int64_t* p = kp_tensor.GetTensorData<int64_t>();
        for (int64_t i = 0; i < N; ++i) push_xy((double)p[(size_t)i * 2 + 0], (double)p[(size_t)i * 2 + 1]);
    } else if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
        const int32_t* p = kp_tensor.GetTensorData<int32_t>();
        for (int64_t i = 0; i < N; ++i) push_xy((double)p[(size_t)i * 2 + 0], (double)p[(size_t)i * 2 + 1]);
    } else {
        throw std::runtime_error("Unsupported keypoints element type in simple demo.");
    }

    return kpts;
}

static SPFeatures runSuperPointSimple(Ort::Session& sp_sess,
                                     const cv::Mat& img,
                                     int H,
                                     int W,
                                     const std::string& tag) {
    auto t0 = std::chrono::steady_clock::now();

    cv::Mat gray;
    if (img.channels() == 3) cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    else gray = img;

    std::vector<float> input = toFloatCHW01(gray, H, W);
    std::array<int64_t, 4> in_shape{1, 1, (int64_t)H, (int64_t)W};

    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value in_tensor = Ort::Value::CreateTensor<float>(mem, input.data(), input.size(), in_shape.data(), in_shape.size());

    const char* in_names[] = {"image"};
    const char* out_names[] = {"keypoints", "scores", "descriptors"};

    auto t_run0 = std::chrono::steady_clock::now();
    auto outs = sp_sess.Run(Ort::RunOptions{nullptr}, in_names, &in_tensor, 1, out_names, 3);
    auto t_run1 = std::chrono::steady_clock::now();

    // debug: print keypoints type/shape to confirm we are running the latest binary
    {
        auto info = outs[0].GetTensorTypeAndShapeInfo();
        auto shp = info.GetShape();
        std::cout << "[Debug][" << tag << "] keypoints elem_type=" << elementTypeToString(info.GetElementType()) << " shape=[";
        for (size_t i = 0; i < shp.size(); ++i) std::cout << shp[i] << (i + 1 < shp.size() ? "," : "");
        std::cout << "]\n";
    }

    // keypoints: 常见为 int64(像素坐标) 或 float32(可能是像素坐标)
    std::vector<cv::KeyPoint> kpts = parseKeypointsSimple(outs[0]);
    int64_t N = (int64_t)kpts.size();

    // descriptors: float32, shape [1,N,D]
    const auto& desc = outs[2];
    auto d_info = desc.GetTensorTypeAndShapeInfo();
    auto d_shape = d_info.GetShape();
    if (d_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        throw std::runtime_error("SuperPoint descriptors must be float32 in this simple demo.");
    if (!(d_shape.size() == 3 && d_shape[0] == 1))
        throw std::runtime_error("Unexpected descriptors shape in simple demo.");

    int64_t Nd = d_shape[1];
    int64_t D = d_shape[2];
    const float* d_ptr = desc.GetTensorData<float>();

    SPFeatures f;
    f.kpts = std::move(kpts);
    f.N = N;
    f.D = D;
    f.desc.assign(d_ptr, d_ptr + (size_t)Nd * (size_t)D);

    auto t1 = std::chrono::steady_clock::now();
    double ms_total = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double ms_run = std::chrono::duration<double, std::milli>(t_run1 - t_run0).count();
    std::cout << "[Time][SuperPoint " << tag << "] total=" << ms_total << " ms, ort_run=" << ms_run << " ms\n";
    std::cout << "[SuperPoint " << tag << "] N=" << f.N << " D=" << f.D << "\n";

    return f;
}

static std::vector<MatchWithScore> runLightGlueSimple(Ort::Session& lg_sess,
                                                      const SPFeatures& f0,
                                                      const SPFeatures& f1,
                                                      float mscore_thresh) {
    auto t0 = std::chrono::steady_clock::now();

    // kpts: [1,N,2]
    std::vector<float> k0((size_t)f0.N * 2), k1((size_t)f1.N * 2);
    for (int64_t i = 0; i < f0.N; ++i) {
        k0[(size_t)i * 2 + 0] = f0.kpts[(size_t)i].pt.x;
        k0[(size_t)i * 2 + 1] = f0.kpts[(size_t)i].pt.y;
    }
    for (int64_t i = 0; i < f1.N; ++i) {
        k1[(size_t)i * 2 + 0] = f1.kpts[(size_t)i].pt.x;
        k1[(size_t)i * 2 + 1] = f1.kpts[(size_t)i].pt.y;
    }

    std::array<int64_t, 3> k0_shape{1, f0.N, 2};
    std::array<int64_t, 3> k1_shape{1, f1.N, 2};
    std::array<int64_t, 3> d0_shape{1, f0.N, f0.D};
    std::array<int64_t, 3> d1_shape{1, f1.N, f1.D};

    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value k0_t = Ort::Value::CreateTensor<float>(mem, k0.data(), k0.size(), k0_shape.data(), k0_shape.size());
    Ort::Value k1_t = Ort::Value::CreateTensor<float>(mem, k1.data(), k1.size(), k1_shape.data(), k1_shape.size());
    Ort::Value d0_t = Ort::Value::CreateTensor<float>(mem, const_cast<float*>(f0.desc.data()), f0.desc.size(), d0_shape.data(), d0_shape.size());
    Ort::Value d1_t = Ort::Value::CreateTensor<float>(mem, const_cast<float*>(f1.desc.data()), f1.desc.size(), d1_shape.data(), d1_shape.size());

    const char* in_names[] = {"kpts0", "kpts1", "desc0", "desc1"};
    const char* out_names[] = {"matches0", "mscores0"};

    std::array<Ort::Value, 4> in_tensors = {std::move(k0_t), std::move(k1_t), std::move(d0_t), std::move(d1_t)};

    auto t_run0 = std::chrono::steady_clock::now();
    auto outs = lg_sess.Run(Ort::RunOptions{nullptr}, in_names, in_tensors.data(), in_tensors.size(), out_names, 2);
    auto t_run1 = std::chrono::steady_clock::now();

    // matches0: int64, shape [M,2]
    const auto& m0 = outs[0];
    auto m0_info = m0.GetTensorTypeAndShapeInfo();
    auto m0_shape = m0_info.GetShape();
    if (m0_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
        throw std::runtime_error("LightGlue matches0 must be int64 in this simple demo.");
    if (!(m0_shape.size() == 2 && m0_shape[1] == 2))
        throw std::runtime_error("Unexpected matches0 shape in simple demo.");

    int64_t M = m0_shape[0];
    const int64_t* mptr = m0.GetTensorData<int64_t>();

    // mscores0: float32, shape [M]
    const auto& sc0 = outs[1];
    auto sc_info = sc0.GetTensorTypeAndShapeInfo();
    auto sc_shape = sc_info.GetShape();
    if (sc_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        throw std::runtime_error("LightGlue mscores0 must be float32 in this simple demo.");
    if (!((sc_shape.size() == 1 && sc_shape[0] == M) || (sc_shape.size() == 2 && sc_shape[0] == 1 && sc_shape[1] == M)))
        throw std::runtime_error("Unexpected mscores0 shape in simple demo.");

    const float* sptr = sc0.GetTensorData<float>();

    std::vector<MatchWithScore> pairs;
    pairs.reserve((size_t)M);
    for (int64_t i = 0; i < M; ++i) {
        int64_t a = mptr[(size_t)i * 2 + 0];
        int64_t b = mptr[(size_t)i * 2 + 1];
        float s = sptr[i];
        if (mscore_thresh >= 0.f && s < mscore_thresh) continue;
        pairs.push_back({(int)a, (int)b, s});
    }

    auto t1 = std::chrono::steady_clock::now();
    double ms_total = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double ms_run = std::chrono::duration<double, std::milli>(t_run1 - t_run0).count();
    std::cout << "[Time][LightGlue] total=" << ms_total << " ms, ort_run=" << ms_run
              << " ms, kpts0=" << f0.N << " kpts1=" << f1.N << "\n";
    std::cout << "[LightGlue] pairs=" << pairs.size() << "\n";

    return pairs;
}

static cv::Mat drawMatchesCustom(const cv::Mat& left,
                                 const cv::Mat& right,
                                 const std::vector<cv::KeyPoint>& k0,
                                 const std::vector<cv::KeyPoint>& k1,
                                 const std::vector<MatchWithScore>& pairs,
                                 int max_draw) {
    cv::Mat L = left.clone(), R = right.clone();
    if (L.channels() == 1) cv::cvtColor(L, L, cv::COLOR_GRAY2BGR);
    if (R.channels() == 1) cv::cvtColor(R, R, cv::COLOR_GRAY2BGR);

    cv::Mat canvas;
    cv::hconcat(L, R, canvas);
    int offset = L.cols;

    int draw_n = std::min<int>(max_draw, (int)pairs.size());
    for (int i = 0; i < draw_n; ++i) {
        const auto& m = pairs[i];
        cv::Point2f p0 = k0[m.q].pt;
        cv::Point2f p1 = k1[m.t].pt + cv::Point2f((float)offset, 0.f);
        cv::line(canvas, p0, p1, cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
        cv::circle(canvas, p0, 3, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
        cv::circle(canvas, p1, 3, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
    }

    cv::putText(canvas, "draw " + std::to_string(draw_n) + " matches", cv::Point(20, 40),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);
    return canvas;
}

// 打印当前 ORT build 可用的 EP 列表（不是“当前 session 实际使用的 EP”，但能判断是否包含 CUDA）
static void printAvailableProviders() {
    try {
        // 注意：不同 ORT 版本 Ort::GetApi() 返回的是引用
        const OrtApi& api = Ort::GetApi();

        char** providers = nullptr;
        int len = 0;

        OrtStatus* st = api.GetAvailableProviders(&providers, &len);
        if (st != nullptr) {
            const char* msg = api.GetErrorMessage(st);
            std::cout << "[EP] GetAvailableProviders failed: " << (msg ? msg : "(unknown)") << "\n";
            api.ReleaseStatus(st);
            return;
        }

        std::cout << "[EP] available providers=";
        for (int i = 0; i < len; ++i) {
            std::cout << (providers[i] ? providers[i] : "(null)") << (i + 1 < len ? "," : "");
        }
        std::cout << "\n";

        // 释放由 ORT 分配的 providers 数组
        (void)api.ReleaseAvailableProviders(providers, len);
    } catch (const std::exception& e) {
        std::cout << "[EP] GetAvailableProviders not supported in this ORT build: " << e.what() << "\n";
    } catch (...) {
        std::cout << "[EP] GetAvailableProviders not supported in this ORT build.\n";
    }
}

// 新增：打印某个 session 实际启用/优先级顺序的 EP（最可靠的“是否在用 CUDA”判断）
// 说明：不同 ORT 版本 C++/C API 差异很大，这里不再依赖“查询 session provider 列表”的 API。
// 我们改用 profiling 输出文件来确认：profile 里会记录每个节点由哪个 EP 执行。
static void printSessionProviders(const char* tag, Ort::Session& /*sess*/) {
    std::cout << "[EP][" << tag << "] active provider list query is not supported in this ORT headers; use --profile to confirm EP (CUDA/CPU) from the profile JSON.\n";
}

static bool tryParseInt(const std::string& s, int& out) {
    try {
        size_t idx = 0;
        int v = std::stoi(s, &idx);
        if (idx != s.size()) return false;
        out = v;
        return true;
    } catch (...) {
        return false;
    }
}

static void printUsage(const char* prog) {
    std::cout
        << "Usage (one-shot, compatible with previous version):\n"
        << "  " << prog << " <sp.onnx> <lg.onnx> <img0> <img1> <out.png> <H> <W> [iters] [--profile]\n\n"
        << "Usage (resident interactive):\n"
        << "  " << prog << " <sp.onnx> <lg.onnx> <H> <W> [iters] [warmup] [--profile]\n"
        << "  then input lines: <img0> <img1> [out.png], or 'q' to quit\n";
}

// 新增：解析可选参数（只支持 --profile）
struct ExtraFlags {
    bool enable_profile = false;
};

static ExtraFlags parseExtraFlags(int argc, char** argv) {
    ExtraFlags f;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--profile" || a == "--profiling") {
            f.enable_profile = true;
        }
    }
    return f;
}

// 新增：尝试启用 CUDA EP（尽量兼容不同 ORT 版本）
static bool tryEnableCuda(Ort::SessionOptions& opt, int device_id) {
    try {
        const OrtApi& api = Ort::GetApi();

        // 先检查 build 是否包含 CUDA EP
        char** providers = nullptr;
        int len = 0;
        OrtStatus* st0 = api.GetAvailableProviders(&providers, &len);
        if (st0 != nullptr) {
            api.ReleaseStatus(st0);
            return false;
        }
        bool has_cuda = false;
        for (int i = 0; i < len; ++i) {
            if (providers[i] && std::string(providers[i]) == "CUDAExecutionProvider") {
                has_cuda = true;
                break;
            }
        }
        api.ReleaseAvailableProviders(providers, len);
        if (!has_cuda) {
            std::cout << "[EP] CUDAExecutionProvider not in available providers. Will use CPU.\n";
            return false;
        }

        // 方案 A（新）：CUDAProviderOptionsV2 + SessionOptionsAppendExecutionProvider_CUDA_V2
        // 方案 B（旧）：OrtCUDAProviderOptions + SessionOptionsAppendExecutionProvider_CUDA
        // 两套 API 取决于你安装的 onnxruntime headers。

#if defined(ORT_API_VERSION) && (ORT_API_VERSION >= 14)
        // 先尝试 V2（多数 1.14+ 可用）
        {
            OrtCUDAProviderOptionsV2* cuda_options = nullptr;
            OrtStatus* st_create = api.CreateCUDAProviderOptions(&cuda_options);
            if (st_create != nullptr) {
                api.ReleaseStatus(st_create);
                cuda_options = nullptr;
            }

            if (cuda_options) {
                const char* keys[] = {"device_id"};
                std::string dev = std::to_string(device_id);
                const char* values[] = {dev.c_str()};
                OrtStatus* st_upd = api.UpdateCUDAProviderOptions(cuda_options, keys, values, 1);
                if (st_upd != nullptr) {
                    const char* msg = api.GetErrorMessage(st_upd);
                    std::cout << "[EP] UpdateCUDAProviderOptions failed: " << (msg ? msg : "(unknown)") << "\n";
                    api.ReleaseStatus(st_upd);
                    api.ReleaseCUDAProviderOptions(cuda_options);
                    cuda_options = nullptr;
                }
            }

            if (cuda_options) {
                OrtStatus* st_append = api.SessionOptionsAppendExecutionProvider_CUDA_V2((OrtSessionOptions*)opt, cuda_options);
                api.ReleaseCUDAProviderOptions(cuda_options);
                if (st_append == nullptr) {
                    std::cout << "[EP] CUDA EP enabled via CUDA_V2 (device_id=" << device_id << ")\n";
                    return true;
                }
                const char* msg = api.GetErrorMessage(st_append);
                std::cout << "[EP] SessionOptionsAppendExecutionProvider_CUDA_V2 failed: " << (msg ? msg : "(unknown)") << "\n";
                api.ReleaseStatus(st_append);
                // 继续尝试旧版
            }
        }
#endif

        // fallback：旧版 OrtCUDAProviderOptions
#ifdef ORT_CUDA_EP
        // 如果头文件定义了 ORT_CUDA_EP 之类宏，可以用于编译期判断；但不同版本不一致。
#endif
        {
            // 某些版本需要包含额外头文件才能看到 OrtCUDAProviderOptions；
            // 但在官方发布包里 onnxruntime_c_api.h 往往已经声明。
            OrtCUDAProviderOptions cuda_options; // NOLINT
            std::memset(&cuda_options, 0, sizeof(cuda_options));
            cuda_options.device_id = device_id;

            OrtStatus* st_append = api.SessionOptionsAppendExecutionProvider_CUDA((OrtSessionOptions*)opt, &cuda_options);
            if (st_append != nullptr) {
                const char* msg = api.GetErrorMessage(st_append);
                std::cout << "[EP] SessionOptionsAppendExecutionProvider_CUDA failed: " << (msg ? msg : "(unknown)") << "\n";
                api.ReleaseStatus(st_append);
                return false;
            }
            std::cout << "[EP] CUDA EP enabled via legacy CUDA options (device_id=" << device_id << ")\n";
            return true;
        }

    } catch (...) {
        std::cout << "[EP] Enabling CUDA EP threw exception. Will use CPU.\n";
        return false;
    }
}

int main(int argc, char** argv) {
    // 常驻模式：
    // 1) 启动时加载模型 (SuperPoint + LightGlue)
    // 2) 可选 warm-up
    // 3) 循环等待用户输入图片路径进行推理
    //
    // 用法：
    //   ./sp_lg_demo superpoint.onnx lightglue.onnx 480 752 [iters] [warmup]
    //   然后在 stdin 中输入：
    //     img0_path img1_path out_path
    //   或：
    //     img0_path img1_path
    //   输入 q 或 quit 退出。

    if (argc < 3) {
        printUsage(argv[0]);
        return 0;
    }

    // 解析额外 flag（在任何模式下都支持）
    ExtraFlags flags = parseExtraFlags(argc, argv);

    std::string sp_path = argv[1];
    std::string lg_path = argv[2];

    // 模式判断：
    // - one-shot: argv[3] argv[4] argv[5] 是图片路径，argv[6]/argv[7] 是 H/W
    // - resident: argv[3]/argv[4] 是 H/W
    bool one_shot = false;
    std::string img0_arg, img1_arg, out_arg;
    int H = 480, W = 752;
    int iters = 10;
    int warmup = 10;

    if (argc >= 8) {
        // 优先尝试 one-shot：要求 argv[6]/argv[7] 能解析为整数
        int Htmp = 0, Wtmp = 0;
        if (tryParseInt(argv[6], Htmp) && tryParseInt(argv[7], Wtmp)) {
            one_shot = true;
            img0_arg = argv[3];
            img1_arg = argv[4];
            out_arg = argv[5];
            H = Htmp;
            W = Wtmp;
            if (argc >= 9) {
                int it = 0;
                if (tryParseInt(argv[8], it) && it > 0) iters = it;
            }
            warmup = 0; // one-shot 默认不 warmup（避免额外耗时），需要的话可再加参数扩展
        }
    }

    if (!one_shot) {
        // resident 模式：argv[3]/argv[4] 必须是 H/W
        if (argc < 5) {
            printUsage(argv[0]);
            return 0;
        }
        if (!tryParseInt(argv[3], H) || !tryParseInt(argv[4], W)) {
            std::cout << "[Error] H/W parse failed. Did you mean to run one-shot mode?\n";
            printUsage(argv[0]);
            return 1;
        }
        if (argc >= 6) {
            int it = 0;
            if (tryParseInt(argv[5], it) && it > 0) iters = it;
        }
        if (argc >= 7) {
            int wu = 0;
            if (tryParseInt(argv[6], wu) && wu >= 0) warmup = wu;
        }
    }

    if (iters <= 0) iters = 1;
    if (warmup < 0) warmup = 0;

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "sp_lg_resident");
    Ort::SessionOptions opt;
    opt.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // 可选：开启 profiling，输出 JSON（用于确认实际 EP：CUDA/CPU）
    if (flags.enable_profile) {
        // ORT 会在运行结束/Session 释放时写出 profile 文件
#ifdef _WIN32
        opt.EnableProfiling(L"sp_lg_demo_profile");
#else
        opt.EnableProfiling("sp_lg_demo_profile");
#endif
    }

    // 默认尝试启用 CUDA EP（device_id=0）；失败自动回退 CPU
    int default_device_id = 0;
    bool use_cuda = tryEnableCuda(opt, default_device_id);

    // 会话只创建一次常驻
    Ort::Session sp_sess(env, sp_path.c_str(), opt);
    Ort::Session lg_sess(env, lg_path.c_str(), opt);

    std::cout << "[EP] auto_try_cuda=1"
              << " cuda_enabled=" << (use_cuda ? 1 : 0)
              << " device_id=" << default_device_id << "\n";

    // Re-print available providers to make it explicit in the logs which EPs are available
    printAvailableProviders();

    // 提示用户用 profiling 确认实际 EP
    if (flags.enable_profile) {
        std::cout << "[Profile] enabled. Profile JSON will be written on exit.\n";
    }

    std::cout << "[Config] mode=" << (one_shot ? "one-shot" : "resident")
              << " H=" << H << " W=" << W << " iters=" << iters << " warmup=" << warmup
              << " use_cuda=" << (use_cuda ? 1 : 0) << " device_id=" << default_device_id
              << " profiling=" << (flags.enable_profile ? 1 : 0) << "\n";

    float mscore_thresh = 0.1f;

    auto run_pair = [&](const cv::Mat& img0, const cv::Mat& img1, const std::string& out_p) {
        cv::Mat img0r, img1r;
        cv::resize(img0, img0r, cv::Size(W, H));
        cv::resize(img1, img1r, cv::Size(W, H));

        std::vector<MatchWithScore> last_pairs;
        SPFeatures last_f0, last_f1;

        auto t_all0 = std::chrono::steady_clock::now();
        for (int iter = 1; iter <= iters; ++iter) {
            std::cout << "\n===== Iteration " << iter << "/" << iters << " =====\n";

            auto t0 = std::chrono::steady_clock::now();
            auto f0 = runSuperPointSimple(sp_sess, img0, H, W, "img0");
            auto f1 = runSuperPointSimple(sp_sess, img1, H, W, "img1");
            auto pairs = runLightGlueSimple(lg_sess, f0, f1, mscore_thresh);
            auto t1 = std::chrono::steady_clock::now();

            double ms_iter = std::chrono::duration<double, std::milli>(t1 - t0).count();
            std::cout << "[Time][Iter " << iter << "] total_pipeline=" << ms_iter << " ms\n";

            last_f0 = std::move(f0);
            last_f1 = std::move(f1);
            last_pairs = std::move(pairs);
        }
        auto t_all1 = std::chrono::steady_clock::now();
        double ms_all = std::chrono::duration<double, std::milli>(t_all1 - t_all0).count();
        std::cout << "\n[Time] ran " << iters << " iterations, total=" << ms_all << " ms, avg=" << (ms_all / iters) << " ms\n";

        std::sort(last_pairs.begin(), last_pairs.end(),
                  [](const MatchWithScore& a, const MatchWithScore& b) { return a.s > b.s; });

        cv::Mat vis = drawMatchesCustom(img0r, img1r, last_f0.kpts, last_f1.kpts, last_pairs, 200);
        cv::imwrite(out_p, vis);
        std::cout << "Saved: " << out_p << "\n";

        cv::imshow("SP+LG matches", vis);
        cv::waitKey(1);
    };

    if (one_shot) {
        cv::Mat img0 = cv::imread(img0_arg, cv::IMREAD_UNCHANGED);
        cv::Mat img1 = cv::imread(img1_arg, cv::IMREAD_UNCHANGED);
        if (img0.empty() || img1.empty()) {
            std::cerr << "Failed to load images.\n";
            return -1;
        }
        run_pair(img0, img1, out_arg);
        return 0;
    }

    // warm-up：先跑几次，减少第一次的抖动（不计入后续统计）
    if (warmup > 0) {
        std::cout << "[Warmup] begin " << warmup << " iterations...\n";
        cv::Mat dummy(H, W, CV_8UC1, cv::Scalar(0));
        for (int i = 0; i < warmup; ++i) {
            auto f0 = runSuperPointSimple(sp_sess, dummy, H, W, "warm0");
            auto f1 = runSuperPointSimple(sp_sess, dummy, H, W, "warm1");
            (void)runLightGlueSimple(lg_sess, f0, f1, mscore_thresh);
        }
        std::cout << "[Warmup] done.\n";
    }

    std::cout << "\nEnter: <img0_path> <img1_path> [out_path]  (or 'q' to quit)\n";

    std::string line;
    while (std::cout << "> " && std::getline(std::cin, line)) {
        if (line.empty()) continue;
        if (line == "q" || line == "quit" || line == "exit") break;

        std::istringstream iss(line);
        std::string img0_p, img1_p, out_p;
        if (!(iss >> img0_p >> img1_p)) {
            std::cout << "[Input] invalid line. Expect: img0 img1 [out]\n";
            continue;
        }
        if (!(iss >> out_p)) out_p = "./output/matches.png";

        cv::Mat img0 = cv::imread(img0_p, cv::IMREAD_UNCHANGED);
        cv::Mat img1 = cv::imread(img1_p, cv::IMREAD_UNCHANGED);
        if (img0.empty() || img1.empty()) {
            std::cout << "[Input] failed to load images.\n";
            continue;
        }

        run_pair(img0, img1, out_p);
    }

    std::cout << "bye\n";

    // 如果开启 profiling：此 ORT 版本不一定提供 EndProfilingAllocated()，
    // 但 profile 文件会在 Session 释放/进程退出时自动写出。
    if (flags.enable_profile) {
        std::cout << "[Profile] enabled: profile files will be written automatically on exit (prefix=sp_lg_demo_profile).\n";
    }

    return 0;
}
