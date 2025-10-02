#include <ceres/ceres.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <chrono>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <random>


// 1表示开启，0表示关闭
#define ENABLE_REAL_TIME_VIS 0    
#define ENABLE_FINAL_VIS 0        
#define ENABLE_RESIDUAL_ANALYSIS 0
#define ENABLE_LOG 0              



struct TrajectoryData {
    double t;  
    double x;  
    double y;  
};


template <typename T>
void ProjectileModel(T t, T x0, T y0, T v0, T theta, T g, T k, T& x_pred, T& y_pred, bool invert_y = true) {
    
    T v0x = v0 * cos(theta);  
    T v0y = v0 * sin(theta);  
    
    
    x_pred = x0 + (v0x / k) * (T(1.0) - exp(-k * t));
    
    
    T term_y = (v0y + (g / k)) / k;
    T y_raw = y0 + term_y * (T(1.0) - exp(-k * t)) - (g / k) * t;
    
    
    if (invert_y) {
        y_pred = y0 - (y_raw - y0);  
    } else {
        y_pred = y_raw;
    }
}


struct ProjectileCostFunctor {
    ProjectileCostFunctor(double t, double x_obs, double y_obs, double x0, double y0)
        : t_(t), x_obs_(x_obs), y_obs_(y_obs), x0_(x0), y0_(y0) {}

    template <typename T>
    bool operator()(const T* const v0, const T* const theta, const T* const g, const T* const k, T* residuals) const {
        T x_pred, y_pred;
        ProjectileModel(T(t_), T(x0_), T(y0_), v0[0], theta[0], g[0], k[0], x_pred, y_pred, true);
        
        
        residuals[0] = (x_obs_ - x_pred) * T(1.0);
        residuals[1] = (y_obs_ - y_pred) * T(1.1);  
        return true;
    }

private:
    const double t_;       
    const double x_obs_;   
    const double y_obs_;   
    const double x0_;      
    const double y0_;      
};


std::vector<TrajectoryData> extractTrajectory(const std::string& video_path, 
                                             const cv::Size& circle_size_range,
                                             bool interpolate = true) {
    std::vector<TrajectoryData> trajectory;
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
#if ENABLE_LOG
        LOG(FATAL) << "无法打开视频文件: " << video_path;
#else
        std::cerr << "无法打开视频文件: " << video_path << std::endl;
        exit(-1);
#endif
    }

    const double fps = cap.get(cv::CAP_PROP_FPS);
    const int frame_count = cap.get(cv::CAP_PROP_FRAME_COUNT);
#if ENABLE_LOG
    LOG(INFO) << "视频信息: FPS=" << fps << ", 总帧数=" << frame_count;
#endif

    cv::Mat frame, gray, blurred;
    int frame_idx = 0;
    const int min_radius = circle_size_range.width;
    const int max_radius = circle_size_range.height;

#if ENABLE_REAL_TIME_VIS
    cv::namedWindow("Projectile Tracking", cv::WINDOW_NORMAL);
#endif

    
    while (cap.read(frame)) {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, blurred, cv::Size(9, 9), 2.0, 2.0);

        std::vector<cv::Vec3f> circles;
        cv::HoughCircles(blurred, circles, cv::HOUGH_GRADIENT,
                        1.5, gray.rows / 8, 100, 30, min_radius, max_radius);

        if (!circles.empty()) {
            cv::Vec3f circle = circles[0];
            cv::Point2f center(circle[0], circle[1]);
            
            double t = frame_idx / fps;
            trajectory.push_back({t, center.x, center.y});

#if ENABLE_REAL_TIME_VIS
            
            cv::circle(frame, center, circle[2], cv::Scalar(0, 255, 0), 2);
            cv::circle(frame, center, 2, cv::Scalar(0, 0, 255), -1);
#endif
        }

#if ENABLE_REAL_TIME_VIS
        cv::imshow("Projectile Tracking", frame);
        if (cv::waitKey(5) == 27) break;
#endif

        frame_idx++;
    }

    cap.release();
#if ENABLE_REAL_TIME_VIS
    cv::destroyWindow("Projectile Tracking");
#endif

    
    std::sort(trajectory.begin(), trajectory.end(), 
              [](const TrajectoryData& a, const TrajectoryData& b) {
                  return a.t < b.t;
              });

#if ENABLE_LOG
    LOG(INFO) << "原始轨迹点数量: " << trajectory.size();
#endif

    
    if (interpolate && trajectory.size() > 2) {
        std::vector<TrajectoryData> interpolated;
        double total_time = trajectory.back().t - trajectory.front().t;
        double avg_interval = total_time / (trajectory.size() - 1);
        double step = avg_interval / 5.0;  
        
        for (size_t i = 0; i < trajectory.size() - 1; ++i) {
            const auto& p1 = trajectory[i];
            const auto& p2 = trajectory[i+1];
            
            interpolated.push_back(p1);
            
            
            int num_steps = static_cast<int>((p2.t - p1.t) / step);
            for (int j = 1; j < num_steps; ++j) {
                double ratio = static_cast<double>(j) / num_steps;
                TrajectoryData interp;
                interp.t = p1.t + (p2.t - p1.t) * ratio;
                interp.x = p1.x + (p2.x - p1.x) * ratio;
                interp.y = p1.y + (p2.y - p1.y) * ratio;
                interpolated.push_back(interp);
            }
        }
        
        interpolated.push_back(trajectory.back());
        trajectory = interpolated;
#if ENABLE_LOG
        LOG(INFO) << "插值后轨迹点数量: " << trajectory.size();
#endif
    }

    if (trajectory.size() < 20) {
#if ENABLE_LOG
        LOG(FATAL) << "轨迹点数量不足（需≥20），无法进行高精度拟合";
#else
        std::cerr << "轨迹点数量不足（需≥20），无法进行高精度拟合" << std::endl;
        exit(-1);
#endif
    }
    return trajectory;
}


void multiStartOptimization(const std::vector<TrajectoryData>& trajectory,
                           double x0, double y0,
                           double& v0, double& theta, double& g, double& k,
                           const double estimated_min_g = 100.0,  
                           const double estimated_max_g = 1000.0) {  
    
    std::vector<std::tuple<double, double, double, double>> initial_guesses;
    
    if (trajectory.size() > 4) {
        double t_diff = trajectory[4].t - trajectory[0].t;
        double dx = trajectory[4].x - trajectory[0].x;
        double dy = trajectory[4].y - trajectory[0].y;
        double init_v0 = sqrt(dx*dx + dy*dy) / t_diff;
        double init_theta = atan2(dy, dx);  
        
        
        for (double v_ratio : {0.7, 0.85, 1.0, 1.15, 1.3}) {  
            for (double theta_offset : {-0.15, -0.075, 0.0, 0.075, 0.15}) {  
                for (double k_val : {0.01, 0.03, 0.05, 0.07, 0.09}) {  
                    
                    std::random_device rd;
                    std::mt19937 gen(rd());
                    std::uniform_real_distribution<> g_dist(estimated_min_g, estimated_max_g);
                    double g_guess = g_dist(gen);
                    
                    initial_guesses.emplace_back(
                        init_v0 * v_ratio,
                        init_theta + theta_offset,
                        g_guess,
                        k_val
                    );
                }
            }
        }
    } else {
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> v_dist(50.0, 500.0);         
        std::uniform_real_distribution<> theta_dist(0.087, M_PI/2);    
        std::uniform_real_distribution<> g_dist(estimated_min_g, estimated_max_g);  
        std::uniform_real_distribution<> k_dist(0.008, 0.12);          
        
        for (int i = 0; i < 30; ++i) {  
            initial_guesses.emplace_back(
                v_dist(gen),
                theta_dist(gen),
                g_dist(gen),
                k_dist(gen)
            );
        }
    }

    double best_cost = INFINITY;
    double best_v0 = v0, best_theta = theta, best_g = g, best_k = k;

    
    for (const auto& guess : initial_guesses) {
        double curr_v0 = std::get<0>(guess);
        double curr_theta = std::get<1>(guess);
        double curr_g = std::get<2>(guess);
        double curr_k = std::get<3>(guess);

        ceres::Problem problem;
        for (const auto& data : trajectory) {
            ceres::CostFunction* cost_function =
                new ceres::AutoDiffCostFunction<ProjectileCostFunctor, 2, 1, 1, 1, 1>(
                    new ProjectileCostFunctor(data.t, data.x, data.y, x0, y0));
            problem.AddResidualBlock(cost_function, nullptr, 
                                    &curr_v0, &curr_theta, &curr_g, &curr_k);
        }

        
        problem.SetParameterLowerBound(&curr_v0, 0, 30.0);
        problem.SetParameterUpperBound(&curr_v0, 0, 600.0);
        problem.SetParameterLowerBound(&curr_theta, 0, 0.05);  
        problem.SetParameterUpperBound(&curr_theta, 0, M_PI-0.05);  
        problem.SetParameterLowerBound(&curr_g, 0, estimated_min_g * 0.8);  
        problem.SetParameterUpperBound(&curr_g, 0, estimated_max_g * 1.2);  
        problem.SetParameterLowerBound(&curr_k, 0, 1e-6);
        problem.SetParameterUpperBound(&curr_k, 0, 0.25);

        
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.max_num_iterations = 200;
        options.minimizer_progress_to_stdout = false;
        options.gradient_tolerance = 1e-10;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        if (summary.final_cost < best_cost) {
            best_cost = summary.final_cost;
            best_v0 = curr_v0;
            best_theta = curr_theta;
            best_g = curr_g;
            best_k = curr_k;
        }
    }

    v0 = best_v0;
    theta = best_theta;
    g = best_g;
    k = best_k;
#if ENABLE_LOG
    LOG(INFO) << "多初始值优化完成: 最优v0=" << v0 
              << ", 角度=" << theta*180/M_PI << "度, g=" << g << ", k=" << k;
#endif
}


void analyzeAndVisualize(const std::vector<TrajectoryData>& trajectory,
                        double x0, double y0,
                        double v0, double theta, double g, double k,
                        const std::string& save_path = "trajectory_comparison.png") {
    std::vector<double> x_obs, y_obs, t_list;
    std::vector<double> x_pred, y_pred;
    std::vector<double> residuals;

    for (const auto& data : trajectory) {
        t_list.push_back(data.t);
        x_obs.push_back(data.x);
        y_obs.push_back(data.y);

        
        double x_p, y_p;
        ProjectileModel(data.t, x0, y0, v0, theta, g, k, x_p, y_p, true);
        x_pred.push_back(x_p);
        y_pred.push_back(y_p);

        residuals.push_back(sqrt(pow(data.x - x_p, 2) + pow(data.y - y_p, 2)));
    }

#if ENABLE_RESIDUAL_ANALYSIS
    
    double mean_res = std::accumulate(residuals.begin(), residuals.end(), 0.0) / residuals.size();
    double max_res = *std::max_element(residuals.begin(), residuals.end());
    std::cout << "\n===== 拟合效果分析 =====" << std::endl;
    std::cout << "平均残差: " << std::fixed << std::setprecision(4) << mean_res << " 像素" << std::endl;
    std::cout << "最大残差: " << std::fixed << std::setprecision(4) << max_res << " 像素" << std::endl;
    std::cout << "残差标准差: " << std::fixed << std::setprecision(4) 
              << sqrt(std::inner_product(residuals.begin(), residuals.end(), residuals.begin(), 0.0) 
                      / residuals.size() - mean_res*mean_res) << " 像素" << std::endl;
#endif

#if ENABLE_FINAL_VIS
    
    double x_min = *std::min_element(x_obs.begin(), x_obs.end()) - 50;
    double x_max = *std::max_element(x_obs.begin(), x_obs.end()) + 50;
    double y_min = *std::min_element(y_obs.begin(), y_obs.end()) - 50;
    double y_max = *std::max_element(y_obs.begin(), y_obs.end()) + 50;
    cv::Mat canvas(cv::Size(x_max - x_min, y_max - y_min), CV_8UC3, cv::Scalar(255, 255, 255));

    
    for (size_t i = 0; i < x_obs.size(); i++) {
        cv::Point pt_obs(x_obs[i] - x_min, y_obs[i] - y_min);
        cv::circle(canvas, pt_obs, 2, cv::Scalar(255, 0, 0), -1);
        if (i > 0) {
            cv::Point pt_prev(x_obs[i-1] - x_min, y_obs[i-1] - y_min);
            cv::line(canvas, pt_prev, pt_obs, cv::Scalar(200, 200, 255), 1);
        }
    }

    
    for (size_t i = 0; i < x_pred.size(); i++) {
        cv::Point pt_pred(x_pred[i] - x_min, y_pred[i] - y_min);
        cv::circle(canvas, pt_pred, 1, cv::Scalar(0, 0, 255), -1);
        if (i > 0) {
            cv::Point pt_prev(x_pred[i-1] - x_min, y_pred[i-1] - y_min);
            cv::line(canvas, pt_prev, pt_pred, cv::Scalar(0, 0, 255), 2);
        }
    }

    
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) 
       << "v0=" << v0 << "px/s, 角度=" << theta*180/M_PI << "°, "
       << "g=" << g << "px/s², k=" << std::setprecision(4) << k;
    cv::putText(canvas, ss.str(), cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
    cv::putText(canvas, "观测轨迹(蓝) | 预测轨迹(红)", cv::Point(20, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);

    cv::imwrite(save_path, canvas);
    cv::imshow("观测轨迹 vs 预测轨迹", canvas);
    cv::waitKey(0);
    cv::destroyWindow("观测轨迹 vs 预测轨迹");
#if ENABLE_LOG
    LOG(INFO) << "轨迹对比图已保存至: " << save_path;
#endif
#endif  
}  

int main(int argc, char**argv) {
    google::InitGoogleLogging(argv[0]);
    google::SetLogDestination(google::GLOG_INFO, "trajectory_optimization.log");

    if (argc != 2) {
#if ENABLE_LOG
        LOG(FATAL) << "使用方法: " << argv[0] << " [视频文件路径]";
#else
        std::cerr << "使用方法: " << argv[0] << " [视频文件路径]" << std::endl;
        return -1;
#endif
    }

    
    const cv::Size circle_size_range(3, 20); 
    const double estimated_min_g = 100.0;     
    const double estimated_max_g = 1000.0;    

    
    auto start_time = std::chrono::steady_clock::now();
    std::vector<TrajectoryData> trajectory = extractTrajectory(argv[1], circle_size_range, true);
    auto end_time = std::chrono::steady_clock::now();
#if ENABLE_LOG
    LOG(INFO) << "轨迹提取耗时: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() 
              << "ms";
#endif

    
    double x0 = trajectory[0].x;
    double y0 = trajectory[0].y;
    double v0 = 150.0;        
    double theta = M_PI/4;    
    double g = 500.0;         
    double k = 0.05;         

    
    multiStartOptimization(trajectory, x0, y0, v0, theta, g, k, estimated_min_g, estimated_max_g);

    
    ceres::Problem problem;
    for (const auto& data : trajectory) {
        ceres::CostFunction* cost_function =
            new ceres::AutoDiffCostFunction<ProjectileCostFunctor, 2, 1, 1, 1, 1>(
                new ProjectileCostFunctor(data.t, data.x, data.y, x0, y0));
        problem.AddResidualBlock(cost_function, nullptr, &v0, &theta, &g, &k);
    }

    
    problem.SetParameterLowerBound(&v0, 0, 50.0);
    problem.SetParameterUpperBound(&v0, 0, 500.0);
    problem.SetParameterLowerBound(&theta, 0, 0.05);  
    problem.SetParameterUpperBound(&theta, 0, M_PI-0.05);  
    problem.SetParameterLowerBound(&g, 0, estimated_min_g);  
    problem.SetParameterUpperBound(&g, 0, estimated_max_g);  
    problem.SetParameterLowerBound(&k, 0, 1e-5);
    problem.SetParameterUpperBound(&k, 0, 0.2);

    
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 1200;  
    options.minimizer_progress_to_stdout = true;
    options.gradient_tolerance = 1e-16;  
    options.function_tolerance = 1e-16;
    options.parameter_tolerance = 1e-16;
    options.use_nonmonotonic_steps = true;  

    
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << "\n优化摘要:\n" << summary.BriefReport() << std::endl;

    
    std::cout << "\n===== 最终拟合参数 =====" << std::endl;
    std::cout << "初始位置: x0=" << x0 << " px, y0=" << y0 << " px" << std::endl;
    std::cout << "初始速度 v0: " << std::fixed << std::setprecision(2) << v0 << " px/s" << std::endl;
    std::cout << "发射角度 theta: " << std::fixed << std::setprecision(2) << theta*180/M_PI << " 度" << std::endl;
    std::cout << "重力加速度 g: " << std::fixed << std::setprecision(2) << g << " px/s²" << std::endl;
    std::cout << "空气阻力系数 k: " << std::fixed << std::setprecision(4) << k << " 1/s" << std::endl;

    
    analyzeAndVisualize(trajectory, x0, y0, v0, theta, g, k);

    return 0;
}
