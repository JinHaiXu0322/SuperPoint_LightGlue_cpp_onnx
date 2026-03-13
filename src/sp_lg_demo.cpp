#include <opencv2/opencv.hpp> // OpenCV 头文件
#include <onnxruntime_cxx_api.h> // ONNX Runtime C++ API

#include <iostream> // 标准输入输出
#include <vector> // 动态数组
#include <string> // 字符串
#include <cstring> // C 字符串与内存操作
#include <algorithm> // 通用算法
#include <stdexcept> // 异常类型
#include <cmath> // 数学函数
#include <chrono> // 计时

struct SPFeatures { // SuperPoint 特征容器
    std::vector<cv::KeyPoint> kpts;     // 缩放后图像中的像素坐标关键点
    std::vector<float> scores;          // 关键点分数，长度 N
    std::vector<float> desc;            // 描述子，大小 N*D，行优先
    int64_t N = 0; // 关键点数量
    int64_t D = 0; // 描述子维度
    int H = 0, W = 0; // 特征对应的输入分辨率
};

struct MatchWithScore { // 带分数的匹配对
    int q = -1; // 查询图索引
    int t = -1; // 目标图索引
    float s = 0; // 匹配分数
};

static int findNameByKeyword(const std::vector<std::string>& names,
                             const std::vector<std::string>& kws) { // 在名字列表中按关键字匹配
    for (int i = 0; i < (int)names.size(); ++i) // 遍历所有名字
        for (auto& kw : kws) // 遍历所有关键字
            if (names[i].find(kw) != std::string::npos) return i; // 命中则返回索引
    return -1; // 未找到
}

static void printSessionIO(Ort::Session& sess,
                           Ort::AllocatorWithDefaultOptions& alloc,
                           std::vector<std::string>& in_names,
                           std::vector<std::string>& out_names) { // 打印并收集模型输入输出信息
    size_t n_in = sess.GetInputCount(); // 输入数量
    size_t n_out = sess.GetOutputCount(); // 输出数量
    in_names.clear(); out_names.clear(); // 清空容器
    in_names.reserve(n_in); out_names.reserve(n_out); // 预分配

    std::cout << "== Inputs ==\n"; // 打印输入信息
    for (size_t i = 0; i < n_in; ++i) { // 遍历输入
        auto n = sess.GetInputNameAllocated(i, alloc); // 获取输入名称
        in_names.emplace_back(n.get()); // 保存名称
        auto ti = sess.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo(); // 获取类型与形状
        auto shp = ti.GetShape(); // 形状向量
        std::cout << i << ": " << in_names.back() << "  shape=["; // 打印名称与形状
        for (size_t k = 0; k < shp.size(); ++k) std::cout << shp[k] << (k+1<shp.size()? ",":""); // 打印维度
        std::cout << "]\n"; // 结束一行
    }

    std::cout << "== Outputs ==\n"; // 打印输出信息
    for (size_t i = 0; i < n_out; ++i) { // 遍历输出
        auto n = sess.GetOutputNameAllocated(i, alloc); // 获取输出名称
        out_names.emplace_back(n.get()); // 保存名称
        auto ti = sess.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo(); // 获取类型与形状
        auto shp = ti.GetShape(); // 形状向量
        std::cout << i << ": " << out_names.back() << "  shape=["; // 打印名称与形状
        for (size_t k = 0; k < shp.size(); ++k) std::cout << shp[k] << (k+1<shp.size()? ",":""); // 打印维度
        std::cout << "]\n"; // 结束一行
    }
}

static std::vector<float> toFloatCHW01(const cv::Mat& gray, int H, int W) { // 灰度图缩放并归一化到 [0,1]
    cv::Mat r, f; // r 为缩放图，f 为 float 图
    cv::resize(gray, r, cv::Size(W, H), 0, 0, cv::INTER_LINEAR); // 缩放到指定尺寸
    r.convertTo(f, CV_32F, 1.0/255.0); // 转为 float 并归一化
    std::vector<float> out((size_t)H * (size_t)W); // 输出缓冲区
    std::memcpy(out.data(), f.ptr<float>(), out.size()*sizeof(float)); // 拷贝到连续数组
    return out; // 返回 CHW 中的单通道数据
}

static void printKptRange(const std::vector<cv::KeyPoint>& kpts, const std::string& tag) { // 打印关键点范围
    if (kpts.empty()) return; // 空则直接返回
    float minx=1e9f, miny=1e9f, maxx=-1e9f, maxy=-1e9f; // 初始化极值
    for (auto& k : kpts) { // 遍历关键点
        minx = std::min(minx, k.pt.x); maxx = std::max(maxx, k.pt.x); // 更新 x 范围
        miny = std::min(miny, k.pt.y); maxy = std::max(maxy, k.pt.y); // 更新 y 范围
    }
    std::cout << "[KptRange " << tag << "] x:[" << minx << "," << maxx
              << "] y:[" << miny << "," << maxy << "]\n"; // 打印范围
}

// 如果 keypoints 是 float 且看起来在 [0,1] 或 [-1,1]，就转换到像素坐标
static void maybeDenormalizeFloatKpts(std::vector<cv::KeyPoint>& kpts, int W, int H) { // 可能的归一化坐标还原
    if (kpts.empty()) return; // 空则返回

    int n = (int)kpts.size(); // 关键点数量
    int cnt01 = 0, cnt11 = 0; // 统计在区间内的比例
    for (auto& k : kpts) { // 遍历关键点
        float x = k.pt.x, y = k.pt.y; // 当前坐标
        if (x >= 0.f && x <= 1.f && y >= 0.f && y <= 1.f) cnt01++; // 落在 [0,1]
        if (x >= -1.f && x <= 1.f && y >= -1.f && y <= 1.f) cnt11++; // 落在 [-1,1]
    }
    float r01 = (float)cnt01 / (float)n; // [0,1] 占比
    float r11 = (float)cnt11 / (float)n; // [-1,1] 占比

    if (r01 > 0.90f) { // 判断为 [0,1] 归一化
        for (auto& k : kpts) { // 逐点还原
            k.pt.x *= (float)(W - 1); // 还原 x
            k.pt.y *= (float)(H - 1); // 还原 y
        }
        std::cout << "[AutoFix] float keypoints in [0,1] (ratio="<<r01<<") -> pixel\n"; // 日志
    } else if (r11 > 0.90f) { // 判断为 [-1,1] 归一化
        for (auto& k : kpts) { // 逐点还原
            k.pt.x = (k.pt.x + 1.f) * 0.5f * (float)(W - 1); // 还原 x
            k.pt.y = (k.pt.y + 1.f) * 0.5f * (float)(H - 1); // 还原 y
        }
        std::cout << "[AutoFix] float keypoints in [-1,1] (ratio="<<r11<<") -> pixel\n"; // 日志
    } else { // 看起来已是像素坐标
        std::cout << "[AutoFix] float keypoints look like pixel already (r01="<<r01<<", r11="<<r11<<")\n"; // 日志
    }

    // clamp
    for (auto& k : kpts) { // 限制范围
        k.pt.x = std::min(std::max(k.pt.x, 0.f), (float)(W - 1)); // 夹紧 x
        k.pt.y = std::min(std::max(k.pt.y, 0.f), (float)(H - 1)); // 夹紧 y
    }
}

// ✅关键：从 ORT tensor 中“按正确类型+shape”提取 keypoints
static std::vector<cv::KeyPoint> extractKeypointsFromOrt(const Ort::Value& kp_tensor, int W, int H) { // 提取关键点
    auto info = kp_tensor.GetTensorTypeAndShapeInfo(); // 获取类型与形状
    auto shp = info.GetShape(); // 形状
    auto et = info.GetElementType(); // 元素类型

    auto elem_cnt = (int64_t)info.GetElementCount(); // 元素总数
    if (elem_cnt < 2) return {}; // 少于 1 个点则返回空

    // 统一把数据读成 double，再转 float，方便兼容 int/float
    std::vector<double> buf((size_t)elem_cnt); // 临时缓冲

    if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) { // float 类型
        const float* p = kp_tensor.GetTensorData<float>(); // 数据指针
        for (int64_t i=0;i<elem_cnt;++i) buf[(size_t)i] = p[i]; // 拷贝
    } else if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) { // double 类型
        const double* p = kp_tensor.GetTensorData<double>(); // 数据指针
        for (int64_t i=0;i<elem_cnt;++i) buf[(size_t)i] = p[i]; // 拷贝
    } else if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) { // int64 类型
        const int64_t* p = kp_tensor.GetTensorData<int64_t>(); // 数据指针
        for (int64_t i=0;i<elem_cnt;++i) buf[(size_t)i] = (double)p[i]; // 拷贝并转换
    } else if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) { // int32 类型
        const int32_t* p = kp_tensor.GetTensorData<int32_t>(); // 数据指针
        for (int64_t i=0;i<elem_cnt;++i) buf[(size_t)i] = (double)p[i]; // 拷贝并转换
    } else { // 其他类型不支持
        throw std::runtime_error("Unsupported keypoints element type in ONNX output."); // 抛异常
    }

    // 解析 shape：支持 [N,2] / [1,N,2] / [1,2,N] / [2,N]
    std::vector<cv::KeyPoint> kpts; // 输出关键点列表

    auto make_kpt = [&](double x, double y){ // 创建关键点
        return cv::KeyPoint((float)x, (float)y, 2.0f); // 使用固定大小
    };

    if (shp.size() >= 2 && shp.back() == 2) { // ... [N,2]
        // ... x,y interleaved
        int64_t N = elem_cnt / 2; // 点数量
        kpts.reserve((size_t)N); // 预分配
        for (int64_t i=0;i<N;++i) { // 逐点读取
            double x = buf[(size_t)i*2 + 0]; // x
            double y = buf[(size_t)i*2 + 1]; // y
            kpts.push_back(make_kpt(x,y)); // 保存
        }
    } else if (shp.size() >= 2 && shp[shp.size()-2] == 2) { // ... [*,2,N]
        // ... [*,2,N]  two rows: x[0..N-1], y[0..N-1]
        int64_t N = shp.back(); // 点数量
        if (2*N > elem_cnt) throw std::runtime_error("Keypoints shape mismatch."); // 检查
        kpts.reserve((size_t)N); // 预分配
        for (int64_t i=0;i<N;++i) { // 逐点读取
            double x = buf[(size_t)0*N + (size_t)i]; // x
            double y = buf[(size_t)1*N + (size_t)i]; // y
            kpts.push_back(make_kpt(x,y)); // 保存
        }
    } else { // 兜底解析
        // 兜底：按 elem_cnt/2 解析
        int64_t N = elem_cnt / 2; // 点数量
        kpts.reserve((size_t)N); // 预分配
        for (int64_t i=0;i<N;++i) { // 逐点读取
            double x = buf[(size_t)i*2 + 0]; // x
            double y = buf[(size_t)i*2 + 1]; // y
            kpts.push_back(make_kpt(x,y)); // 保存
        }
    }

    // 如果 keypoints 原本是 float，可能是归一化，需要恢复到像素
    if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || et == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
        maybeDenormalizeFloatKpts(kpts, W, H); // 自动判断并还原
    }

    // 如果是 int 类型，一般已经是像素坐标；做 clamp
    for (auto& k : kpts) {
        k.pt.x = std::min(std::max(k.pt.x, 0.f), (float)(W - 1)); // 夹紧 x
        k.pt.y = std::min(std::max(k.pt.y, 0.f), (float)(H - 1)); // 夹紧 y
    }
    return kpts; // 返回关键点
}

static SPFeatures runSuperPoint(Ort::Session& sp_sess,
                               const cv::Mat& img,
                               int H, int W,
                               const std::string& tag) { // 运行 SuperPoint 模型
    auto t0 = std::chrono::steady_clock::now(); // 总计时开始

    SPFeatures feat; // 输出特征
    feat.H = H; feat.W = W; // 记录输入尺寸

    cv::Mat gray; // 灰度图
    if (img.channels() == 3) cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY); // 转灰度
    else gray = img; // 已是灰度

    std::vector<float> input = toFloatCHW01(gray, H, W); // 预处理
    std::array<int64_t,4> in_shape{1,1,(int64_t)H,(int64_t)W}; // NCHW

    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault); // CPU 内存
    Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
        mem, input.data(), input.size(), in_shape.data(), in_shape.size()
    ); // 创建输入张量

    Ort::AllocatorWithDefaultOptions alloc; // 默认分配器
    std::vector<std::string> in_names, out_names; // 输入输出名称
    printSessionIO(sp_sess, alloc, in_names, out_names); // 打印 IO

    std::vector<const char*> in_c{in_names[0].c_str()}; // 输入名
    std::vector<const char*> out_c; // 输出名
    for (auto& s: out_names) out_c.push_back(s.c_str()); // 收集输出名

    auto t_run0 = std::chrono::steady_clock::now(); // 推理开始
    auto outs = sp_sess.Run(Ort::RunOptions{nullptr},
                            in_c.data(), &in_tensor, 1,
                            out_c.data(), out_c.size()); // 推理
    auto t_run1 = std::chrono::steady_clock::now(); // 推理结束

    int k_i = findNameByKeyword(out_names, {"keypoints", "kpts"}); // keypoints 索引
    int s_i = findNameByKeyword(out_names, {"scores", "score"}); // scores 索引
    int d_i = findNameByKeyword(out_names, {"descriptors", "desc"}); // descriptors 索引
    if (k_i < 0 || d_i < 0) throw std::runtime_error("SuperPoint outputs not found."); // 必须存在

    // ✅ 用正确类型提取 keypoints
    feat.kpts = extractKeypointsFromOrt(outs[k_i], W, H); // 解析关键点
    feat.N = (int64_t)feat.kpts.size(); // 记录数量
    printKptRange(feat.kpts, tag + " pixel"); // 打印范围

    // scores (optional)
    if (s_i >= 0) {
        auto s_info = outs[s_i].GetTensorTypeAndShapeInfo(); // 分数信息
        const float* s_ptr = outs[s_i].GetTensorData<float>(); // 分数数据
        int64_t Ns = (int64_t)s_info.GetElementCount(); // 分数个数
        feat.scores.assign(s_ptr, s_ptr + Ns); // 拷贝分数
    }

    // descriptors
    auto d_info = outs[d_i].GetTensorTypeAndShapeInfo(); // 描述子信息
    auto d_shp = d_info.GetShape(); // 描述子形状
    const float* d_ptr = outs[d_i].GetTensorData<float>(); // 描述子数据

    int64_t N = 0, D = 0; // 数量与维度
    if (d_shp.size()==3) { N = d_shp[1]; D = d_shp[2]; } // [1,N,D]
    else if (d_shp.size()==2) { N = d_shp[0]; D = d_shp[1]; } // [N,D]
    else throw std::runtime_error("Unexpected descriptor shape."); // 其他形状报错

    feat.D = D; // 保存维度
    feat.desc.assign(d_ptr, d_ptr + (size_t)N*(size_t)D); // 拷贝描述子

    auto t1 = std::chrono::steady_clock::now(); // 总计时结束
    double ms_total = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double ms_run   = std::chrono::duration<double, std::milli>(t_run1 - t_run0).count();
    std::cout << "[Time][SuperPoint " << tag << "] total=" << ms_total << " ms, ort_run=" << ms_run << " ms\n";

    std::cout << "[SuperPoint " << tag << "] N=" << feat.N << " D=" << feat.D << "\n"; // 打印统计
    return feat; // 返回特征
}

// matches0: [M,2] 或 mapping
static std::vector<MatchWithScore> parseMatches0(const Ort::Value& matches0_tensor,
                                                 const Ort::Value* mscores0_tensor,
                                                 int64_t N0, int64_t N1,
                                                 float mscore_thresh) { // 解析匹配结果
    auto info = matches0_tensor.GetTensorTypeAndShapeInfo(); // 形状信息
    auto shp  = info.GetShape(); // 形状
    const int64_t* mptr = matches0_tensor.GetTensorData<int64_t>(); // 匹配索引

    const float* sptr = nullptr; // 分数指针
    size_t score_cnt = 0; // 分数数量
    if (mscores0_tensor) {
        auto sinfo = mscores0_tensor->GetTensorTypeAndShapeInfo(); // 分数信息
        score_cnt = sinfo.GetElementCount(); // 分数数量
        sptr = mscores0_tensor->GetTensorData<float>(); // 分数数据
    }

    std::vector<MatchWithScore> out; // 输出匹配列表

    if (shp.size() == 2 && shp[1] == 2) { // [M,2] 形式
        int64_t M = shp[0]; // 匹配数量
        out.reserve((size_t)M); // 预分配
        for (int64_t i=0;i<M;++i) { // 遍历匹配
            int64_t a = mptr[i*2+0]; // 查询索引
            int64_t b = mptr[i*2+1]; // 目标索引
            if (a < 0 || a >= N0 || b < 0 || b >= N1) continue; // 越界跳过
            float sc = (sptr && (size_t)i < score_cnt) ? sptr[i] : 0.0f; // 分数
            if (mscore_thresh >= 0.0f && sc < mscore_thresh) continue; // 过滤阈值
            out.push_back({(int)a,(int)b,sc}); // 保存匹配
        }
        return out; // 返回
    }

    int64_t L = 0; // 匹配数量
    if (shp.size() == 2 && shp[0] == 1) L = shp[1]; // [1,L]
    else if (shp.size() == 1) L = shp[0]; // [L]
    else throw std::runtime_error("Unexpected matches0 shape."); // 其他形状报错

    out.reserve((size_t)L); // 预分配
    for (int64_t i=0;i<L;++i) {
        int64_t j = mptr[i]; // 目标索引
        if (i < 0 || i >= N0 || j < 0 || j >= N1) continue; // 越界跳过
        float sc = (sptr && (size_t)i < score_cnt) ? sptr[i] : 0.0f; // 分数
        if (mscore_thresh >= 0.0f && sc < mscore_thresh) continue; // 过滤阈值
        out.push_back({(int)i,(int)j,sc}); // 保存匹配
    }
    return out; // 返回
}

static std::vector<MatchWithScore> runLightGlue(Ort::Session& lg_sess,
                                                const SPFeatures& f0,
                                                const SPFeatures& f1,
                                                float mscore_thresh,
                                                bool print_debug=true) { // 运行 LightGlue
    auto t0 = std::chrono::steady_clock::now(); // 总计时开始

    Ort::AllocatorWithDefaultOptions alloc; // 默认分配器
    std::vector<std::string> in_names, out_names; // 输入输出名称
    printSessionIO(lg_sess, alloc, in_names, out_names); // 打印 IO

    // feed pixel kpts
    std::vector<float> k0((size_t)f0.N*2), k1((size_t)f1.N*2); // 关键点数组
    for (int64_t i=0;i<f0.N;++i) {
        k0[(size_t)i*2]   = f0.kpts[(size_t)i].pt.x; // x
        k0[(size_t)i*2+1] = f0.kpts[(size_t)i].pt.y; // y
    }
    for (int64_t i=0;i<f1.N;++i) {
        k1[(size_t)i*2]   = f1.kpts[(size_t)i].pt.x; // x
        k1[(size_t)i*2+1] = f1.kpts[(size_t)i].pt.y; // y
    }

    std::array<int64_t,3> k0_shape{1,f0.N,2}, k1_shape{1,f1.N,2}; // 关键点形状
    std::array<int64_t,3> d0_shape{1,(int64_t)(f0.desc.size()/f0.D),f0.D}; // 描述子形状
    std::array<int64_t,3> d1_shape{1,(int64_t)(f1.desc.size()/f1.D),f1.D}; // 描述子形状

    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault); // CPU 内存
    Ort::Value k0_t = Ort::Value::CreateTensor<float>(mem, k0.data(), k0.size(), k0_shape.data(), k0_shape.size()); // kpts0
    Ort::Value k1_t = Ort::Value::CreateTensor<float>(mem, k1.data(), k1.size(), k1_shape.data(), k1_shape.size()); // kpts1
    Ort::Value d0_t = Ort::Value::CreateTensor<float>(mem, const_cast<float*>(f0.desc.data()), f0.desc.size(), d0_shape.data(), d0_shape.size()); // desc0
    Ort::Value d1_t = Ort::Value::CreateTensor<float>(mem, const_cast<float*>(f1.desc.data()), f1.desc.size(), d1_shape.data(), d1_shape.size()); // desc1

    std::vector<const char*> in_c; // 输入名数组
    std::vector<Ort::Value> in_v; // 输入值数组
    for (auto& name : in_names) { // 根据名称匹配输入
        const std::string& n = name; // 当前名称
        if (n.find("kpts0")!=std::string::npos) { in_c.push_back(name.c_str()); in_v.push_back(std::move(k0_t)); } // kpts0
        else if (n.find("kpts1")!=std::string::npos) { in_c.push_back(name.c_str()); in_v.push_back(std::move(k1_t)); } // kpts1
        else if (n.find("desc0")!=std::string::npos) { in_c.push_back(name.c_str()); in_v.push_back(std::move(d0_t)); } // desc0
        else if (n.find("desc1")!=std::string::npos) { in_c.push_back(name.c_str()); in_v.push_back(std::move(d1_t)); } // desc1
        else std::cerr << "[Warn] Unhandled LightGlue input: " << n << "\n"; // 未处理输入
    }

    std::vector<const char*> out_c; // 输出名数组
    for (auto& s: out_names) out_c.push_back(s.c_str()); // 收集输出名

    auto t_run0 = std::chrono::steady_clock::now(); // 推理开始
    auto outs = lg_sess.Run(Ort::RunOptions{nullptr},
                            in_c.data(), in_v.data(), in_v.size(),
                            out_c.data(), out_c.size()); // 推理
    auto t_run1 = std::chrono::steady_clock::now(); // 推理结束

    int m0_i = findNameByKeyword(out_names, {"matches0"}); // matches0 索引
    int sc_i = findNameByKeyword(out_names, {"mscores0"}); // mscores0 索引

    if (print_debug) {
        auto m0_info = outs[m0_i].GetTensorTypeAndShapeInfo(); // matches0 信息
        auto m0_shape = m0_info.GetShape(); // matches0 形状
        std::cout << "[Debug] matches0 shape=["; // 打印形状
        for (size_t i=0;i<m0_shape.size();++i) std::cout << m0_shape[i] << (i+1<m0_shape.size()? ",":""); // 维度
        std::cout << "]\n"; // 结束一行
        if (sc_i >= 0) {
            auto sc_info = outs[sc_i].GetTensorTypeAndShapeInfo(); // 分数信息
            auto sc_shape = sc_info.GetShape(); // 分数形状
            std::cout << "[Debug] mscores0 shape=["; // 打印形状
            for (size_t i=0;i<sc_shape.size();++i) std::cout << sc_shape[i] << (i+1<sc_shape.size()? ",":""); // 维度
            std::cout << "]\n"; // 结束一行
        }
    }

    const Ort::Value* score_ptr = (sc_i >= 0) ? &outs[sc_i] : nullptr; // 分数指针
    auto pairs = parseMatches0(outs[m0_i], score_ptr, f0.N, f1.N, mscore_thresh); // 解析匹配

    auto t1 = std::chrono::steady_clock::now(); // 总计时结束
    double ms_total = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double ms_run   = std::chrono::duration<double, std::milli>(t_run1 - t_run0).count();
    std::cout << "[Time][LightGlue] total=" << ms_total << " ms, ort_run=" << ms_run
              << " ms, kpts0=" << f0.N << " kpts1=" << f1.N << "\n";

    std::cout << "[LightGlue] pairs=" << pairs.size() << "\n"; // 打印数量
    return pairs; // 返回匹配
}

static cv::Mat drawMatchesCustom(const cv::Mat& left, const cv::Mat& right,
                                 const std::vector<cv::KeyPoint>& k0,
                                 const std::vector<cv::KeyPoint>& k1,
                                 const std::vector<MatchWithScore>& pairs,
                                 int max_draw) { // 自定义绘制匹配
    cv::Mat L = left.clone(), R = right.clone(); // 克隆输入
    if (L.channels() == 1) cv::cvtColor(L, L, cv::COLOR_GRAY2BGR); // 转为彩色
    if (R.channels() == 1) cv::cvtColor(R, R, cv::COLOR_GRAY2BGR); // 转为彩色

    cv::Mat canvas; // 拼接画布
    cv::hconcat(L, R, canvas); // 水平拼接
    int offset = L.cols; // 右图 x 偏移

    int draw_n = std::min<int>(max_draw, (int)pairs.size()); // 实际绘制数量
    for (int i=0;i<draw_n;++i) { // 绘制每条匹配
        const auto& m = pairs[i]; // 当前匹配
        cv::Point2f p0 = k0[m.q].pt; // 左图点
        cv::Point2f p1 = k1[m.t].pt + cv::Point2f((float)offset, 0.f); // 右图点(平移)

        cv::line(canvas, p0, p1, cv::Scalar(255, 0, 0), 2, cv::LINE_AA); // 连线
        cv::circle(canvas, p0, 3, cv::Scalar(0, 0, 255), -1, cv::LINE_AA); // 左点
        cv::circle(canvas, p1, 3, cv::Scalar(0, 255, 0), -1, cv::LINE_AA); // 右点
    }

    cv::putText(canvas, "draw " + std::to_string(draw_n) + " matches",
                cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                cv::Scalar(0,255,255), 2); // 绘制文本
    return canvas; // 返回可视化结果
}

int main(int argc, char** argv) { // 程序入口
    // ./sp_lg_demo sp.onnx lg.onnx img0 img1 out.png 480 752
    std::string sp_path = (argc>1)? argv[1] : "~/sp_lg_demo/models/superpoint.onnx"; // SuperPoint 模型路径
    std::string lg_path = (argc>2)? argv[2] : "~/sp_lg_demo/models/lightglue_sim.onnx"; // LightGlue 模型路径
    std::string img0_p  = (argc>3)? argv[3] : "~/sp_lg_demo/data/img0.jpg"; // 图像0路径
    std::string img1_p  = (argc>4)? argv[4] : "~/sp_lg_demo/img1.jpg"; // 图像1路径
    std::string out_p   = (argc>5)? argv[5] : "~/sp_lg_demo/output/matches.png"; // 输出图路径
    int H = (argc>6)? std::stoi(argv[6]) : 480; // 高度
    int W = (argc>7)? std::stoi(argv[7]) : 752; // 宽度

    cv::Mat img0 = cv::imread(img0_p, cv::IMREAD_UNCHANGED); // 读取图像0
    cv::Mat img1 = cv::imread(img1_p, cv::IMREAD_UNCHANGED); // 读取图像1
    if (img0.empty() || img1.empty()) {
        std::cerr << "Failed to load images.\n"; // 读取失败
        return -1; // 返回错误
    }

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "sp_lg"); // ORT 环境
    Ort::SessionOptions opt; // 会话配置
    opt.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL); // 开启优化

    Ort::Session sp_sess(env, sp_path.c_str(), opt); // SuperPoint 会话
    Ort::Session lg_sess(env, lg_path.c_str(), opt); // LightGlue 会话

    SPFeatures f0 = runSuperPoint(sp_sess, img0, H, W, "img0"); // 提取图像0特征
    SPFeatures f1 = runSuperPoint(sp_sess, img1, H, W, "img1"); // 提取图像1特征

    float mscore_thresh = 0.1f; // 可改成 0.2f/0.3f 过滤弱匹配
    auto pairs = runLightGlue(lg_sess, f0, f1, mscore_thresh, true); // 进行匹配

    std::sort(pairs.begin(), pairs.end(),
              [](const MatchWithScore& a, const MatchWithScore& b){ return a.s > b.s; }); // 按分数排序

    int max_draw = 200; // 最多绘制条数

    cv::Mat img0r, img1r; // 缩放后的图像
    cv::resize(img0, img0r, cv::Size(W,H)); // 缩放图像0
    cv::resize(img1, img1r, cv::Size(W,H)); // 缩放图像1

    cv::Mat vis = drawMatchesCustom(img0r, img1r, f0.kpts, f1.kpts, pairs, max_draw); // 绘制匹配

    cv::imwrite(out_p, vis); // 保存结果
    std::cout << "Saved: " << out_p << "\n"; // 打印保存路径
    cv::imshow("SP+LG matches", vis); // 显示窗口
    cv::waitKey(0); // 等待按键
    return 0; // 正常退出
}