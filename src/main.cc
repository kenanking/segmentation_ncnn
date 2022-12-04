#include <filesystem>
#include <iostream>
#include <ostream>
#include <string>

#include <layer.h>
#include <net.h>
#include <opencv2/opencv.hpp>

#include "ops/topk.h"
#include "utils/tic_toc.h"

// 注册自定义层
DEFINE_LAYER_CREATOR(TopK)

const unsigned char COLOR[19][3] = {
    {128, 64, 128},  // 0 - road
    {244, 35, 232},  // 1 - sidewalk
    {70, 70, 70},    // 2 - building
    {102, 102, 156}, // 3 - wall
    {190, 153, 153}, // 4 - fence
    {153, 153, 153}, // 5 - pole
    {250, 170, 30},  // 6 - traffic light
    {220, 220, 0},   // 7 - traffic sign
    {107, 142, 35},  // 8 - vegetation
    {152, 251, 152}, // 9 - terrain
    {70, 130, 180},  // 10 - sky
    {220, 20, 60},   // 11 - person
    {255, 0, 0},     // 12 - rider
    {0, 0, 142},     // 13 - car
    {0, 0, 70},      // 14 - truck
    {0, 60, 100},    // 15 - bus
    {0, 80, 100},    // 16 - train
    {0, 0, 230},     // 17 - motorcycle
    {119, 11, 32}    // 18 - bicycle
};

const int INPUT_W = 1024;
const int INPUT_H = 512;

const float MEAN_VALS[3] = {123.675f, 116.28f, 103.53f};
const float NORM_VALS[3] = {1 / 58.395f, 1 / 57.12f, 1 / 57.375f};

int main(int, char **) {
  std::string image_dir = "../data";
  std::string output_dir = "../output/";

  // get all images in image_dir
  std::vector<std::string> image_paths;
  for (const auto &entry : std::filesystem::directory_iterator(image_dir)) {
    std::string path = entry.path();
    if (path.find(".jpg") != std::string::npos ||
        path.find(".png") != std::string::npos) {
      image_paths.push_back(path);
    }
  }

  // sort image_paths
  std::sort(image_paths.begin(), image_paths.end());

  std::string model_param = "../fastscnn/end2end.param";
  std::string model_bin = "../fastscnn/end2end.bin";

  TicToc timer;

  ncnn::Net net;
  net.register_custom_layer("TopK", TopK_layer_creator);
  net.opt.num_threads = 8;
  net.opt.lightmode = true;
  net.opt.use_vulkan_compute = true;

  net.load_param(model_param.c_str());
  net.load_model(model_bin.c_str());

  int num = 0;
  for (auto &image_path : image_paths) {

    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
      std::cerr << "Failed to load image: " << image_path << std::endl;
      return -1;
    }
    int img_w = img.cols;
    int img_h = img.rows;

    cv::resize(img, img, cv::Size(INPUT_W, INPUT_H));

    ncnn::Mat in = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR2RGB,
                                          img.cols, img.rows);
    in.substract_mean_normalize(MEAN_VALS, NORM_VALS);

    timer.tic();
    ncnn::Extractor ex = net.create_extractor();
    ex.input("input", in);
    ncnn::Mat out;
    ex.extract("output", out);
    std::cout << "Inference time: " << timer.toc() * 1000 << " ms" << std::endl;

    cv::Mat color(out.h, out.w, CV_8UC3);

    float *out_data = (float *)out.data;
    unsigned char *colordata = color.data;

    for (int i = 0; i < out.h; i++) {
      for (int j = 0; j < out.w; j++) {
        int idx = (int)out_data[i * out.w + j];
        if (idx >= 11 && idx <= 18) {
          colordata[(i * out.w + j) * 3 + 0] = COLOR[idx][0];
          colordata[(i * out.w + j) * 3 + 1] = COLOR[idx][1];
          colordata[(i * out.w + j) * 3 + 2] = COLOR[idx][2];
        } else {
          colordata[(i * out.w + j) * 3 + 0] = 255;
          colordata[(i * out.w + j) * 3 + 1] = 255;
          colordata[(i * out.w + j) * 3 + 2] = 255;
        }
      }
    }

    cv::addWeighted(img, 0.5, color, 0.5, 0, color);

    std::ostringstream oss;
    oss << output_dir << "output_" << num << ".jpg";
    num += 1;

    // cv::imwrite(oss.str(), color);
    cv::imshow("result", color);
    cv::waitKey(1);
  }

  return 0;
}
