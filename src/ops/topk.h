#ifndef TOPK_H
#define TOPK_H

#include <layer.h>

class TopK : public ncnn::Layer {
public:
  TopK();
  virtual int load_param(const ncnn::ParamDict &pd);
  virtual int forward(const std::vector<ncnn::Mat> &bottom_blobs,
                      std::vector<ncnn::Mat> &top_blobs,
                      const ncnn::Option &opt) const;

public:
  int axis;
  int largest;
  int sorted;
  int keep_dims;
};

#endif // TOPK_H