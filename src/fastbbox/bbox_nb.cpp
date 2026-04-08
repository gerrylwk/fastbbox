/**
 * Nanobind implementation of bounding box IoU calculations.
 * 
 * This file provides the same functionality as bbox.pyx but using nanobind
 * for Python bindings instead of Cython.
 */

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cmath>
#include <algorithm>

namespace nb = nanobind;

using FloatArray2D = nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig, nb::device::cpu>;
using FloatArray2DIn = nb::ndarray<const float, nb::shape<-1, 4>, nb::c_contig, nb::device::cpu>;
using FloatArray2DIn5 = nb::ndarray<const float, nb::shape<-1, 5>, nb::c_contig, nb::device::cpu>;

/**
 * Compute IoU overlaps between two sets of boxes.
 *
 * Parameters
 * ----------
 * boxes: (N, 4) float32 array [x1, y1, x2, y2]
 * query_boxes: (K, 4) float32 array [x1, y1, x2, y2]
 *
 * Returns
 * -------
 * overlaps: (N, K) float32 array of IoU
 */
nb::ndarray<nb::numpy, float, nb::shape<-1, -1>> bbox_overlaps(
    FloatArray2DIn boxes,
    FloatArray2DIn query_boxes
) {
    size_t N = boxes.shape(0);
    size_t K = query_boxes.shape(0);
    
    // Allocate output array
    float* result_data = new float[N * K]();
    
    const float* boxes_ptr = boxes.data();
    const float* query_ptr = query_boxes.data();
    
    for (size_t k = 0; k < K; ++k) {
        float query_area = (query_ptr[k * 4 + 2] - query_ptr[k * 4 + 0]) *
                          (query_ptr[k * 4 + 3] - query_ptr[k * 4 + 1]);
        
        for (size_t n = 0; n < N; ++n) {
            float iw = std::min(boxes_ptr[n * 4 + 2], query_ptr[k * 4 + 2]) -
                      std::max(boxes_ptr[n * 4 + 0], query_ptr[k * 4 + 0]);
            
            if (iw > 0) {
                float ih = std::min(boxes_ptr[n * 4 + 3], query_ptr[k * 4 + 3]) -
                          std::max(boxes_ptr[n * 4 + 1], query_ptr[k * 4 + 1]);
                
                if (ih > 0) {
                    float inter = iw * ih;
                    float box_area = (boxes_ptr[n * 4 + 2] - boxes_ptr[n * 4 + 0]) *
                                    (boxes_ptr[n * 4 + 3] - boxes_ptr[n * 4 + 1]);
                    float ua = box_area + query_area - inter;
                    result_data[n * K + k] = inter / ua;
                }
            }
        }
    }
    
    // Create output ndarray with ownership
    size_t shape[2] = {N, K};
    nb::capsule owner(result_data, [](void* p) noexcept { delete[] static_cast<float*>(p); });
    
    return nb::ndarray<nb::numpy, float, nb::shape<-1, -1>>(
        result_data, 2, shape, owner
    );
}

/**
 * Compute Generalized IoU (GIoU) between two sets of boxes.
 */
nb::ndarray<nb::numpy, float, nb::shape<-1, -1>> generalized_iou(
    FloatArray2DIn boxes,
    FloatArray2DIn query_boxes
) {
    size_t N = boxes.shape(0);
    size_t K = query_boxes.shape(0);
    
    float* result_data = new float[N * K]();
    
    const float* boxes_ptr = boxes.data();
    const float* query_ptr = query_boxes.data();
    
    for (size_t k = 0; k < K; ++k) {
        float query_area = (query_ptr[k * 4 + 2] - query_ptr[k * 4 + 0]) *
                          (query_ptr[k * 4 + 3] - query_ptr[k * 4 + 1]);
        
        for (size_t n = 0; n < N; ++n) {
            // Calculate intersection
            float iw = std::min(boxes_ptr[n * 4 + 2], query_ptr[k * 4 + 2]) -
                      std::max(boxes_ptr[n * 4 + 0], query_ptr[k * 4 + 0]);
            float ih = std::min(boxes_ptr[n * 4 + 3], query_ptr[k * 4 + 3]) -
                      std::max(boxes_ptr[n * 4 + 1], query_ptr[k * 4 + 1]);
            
            float inter = 0.0f;
            if (iw > 0 && ih > 0) {
                inter = iw * ih;
            }
            
            // Calculate union and IoU
            float box_area = (boxes_ptr[n * 4 + 2] - boxes_ptr[n * 4 + 0]) *
                            (boxes_ptr[n * 4 + 3] - boxes_ptr[n * 4 + 1]);
            float ua = box_area + query_area - inter;
            
            float iou = 0.0f;
            if (ua > 0) {
                iou = inter / ua;
            }
            
            // Calculate smallest enclosing box
            float enc_x1 = std::min(boxes_ptr[n * 4 + 0], query_ptr[k * 4 + 0]);
            float enc_y1 = std::min(boxes_ptr[n * 4 + 1], query_ptr[k * 4 + 1]);
            float enc_x2 = std::max(boxes_ptr[n * 4 + 2], query_ptr[k * 4 + 2]);
            float enc_y2 = std::max(boxes_ptr[n * 4 + 3], query_ptr[k * 4 + 3]);
            
            float enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1);
            
            // Calculate GIoU
            float giou_val = iou;
            if (enc_area > 0) {
                float coverage_ratio = (enc_area - ua) / enc_area;
                giou_val = iou - coverage_ratio;
            }
            
            result_data[n * K + k] = giou_val;
        }
    }
    
    size_t shape[2] = {N, K};
    nb::capsule owner(result_data, [](void* p) noexcept { delete[] static_cast<float*>(p); });
    
    return nb::ndarray<nb::numpy, float, nb::shape<-1, -1>>(
        result_data, 2, shape, owner
    );
}

/**
 * Compute Distance IoU (DIoU) between two sets of boxes.
 */
nb::ndarray<nb::numpy, float, nb::shape<-1, -1>> distance_iou(
    FloatArray2DIn boxes,
    FloatArray2DIn query_boxes
) {
    size_t N = boxes.shape(0);
    size_t K = query_boxes.shape(0);
    
    float* result_data = new float[N * K]();
    
    const float* boxes_ptr = boxes.data();
    const float* query_ptr = query_boxes.data();
    
    for (size_t k = 0; k < K; ++k) {
        float query_area = (query_ptr[k * 4 + 2] - query_ptr[k * 4 + 0]) *
                          (query_ptr[k * 4 + 3] - query_ptr[k * 4 + 1]);
        float query_cx = (query_ptr[k * 4 + 0] + query_ptr[k * 4 + 2]) / 2.0f;
        float query_cy = (query_ptr[k * 4 + 1] + query_ptr[k * 4 + 3]) / 2.0f;
        
        for (size_t n = 0; n < N; ++n) {
            // Calculate intersection
            float iw = std::min(boxes_ptr[n * 4 + 2], query_ptr[k * 4 + 2]) -
                      std::max(boxes_ptr[n * 4 + 0], query_ptr[k * 4 + 0]);
            float ih = std::min(boxes_ptr[n * 4 + 3], query_ptr[k * 4 + 3]) -
                      std::max(boxes_ptr[n * 4 + 1], query_ptr[k * 4 + 1]);
            
            float inter = 0.0f;
            if (iw > 0 && ih > 0) {
                inter = iw * ih;
            }
            
            // Calculate union and IoU
            float box_area = (boxes_ptr[n * 4 + 2] - boxes_ptr[n * 4 + 0]) *
                            (boxes_ptr[n * 4 + 3] - boxes_ptr[n * 4 + 1]);
            float ua = box_area + query_area - inter;
            
            float iou = 0.0f;
            if (ua > 0) {
                iou = inter / ua;
            }
            
            // Calculate box centers
            float box_cx = (boxes_ptr[n * 4 + 0] + boxes_ptr[n * 4 + 2]) / 2.0f;
            float box_cy = (boxes_ptr[n * 4 + 1] + boxes_ptr[n * 4 + 3]) / 2.0f;
            
            // Calculate center distance squared
            float center_dist_sq = (box_cx - query_cx) * (box_cx - query_cx) +
                                  (box_cy - query_cy) * (box_cy - query_cy);
            
            // Calculate smallest enclosing box diagonal squared
            float enc_x1 = std::min(boxes_ptr[n * 4 + 0], query_ptr[k * 4 + 0]);
            float enc_y1 = std::min(boxes_ptr[n * 4 + 1], query_ptr[k * 4 + 1]);
            float enc_x2 = std::max(boxes_ptr[n * 4 + 2], query_ptr[k * 4 + 2]);
            float enc_y2 = std::max(boxes_ptr[n * 4 + 3], query_ptr[k * 4 + 3]);
            
            float diagonal_sq = (enc_x2 - enc_x1) * (enc_x2 - enc_x1) +
                               (enc_y2 - enc_y1) * (enc_y2 - enc_y1);
            
            // Calculate DIoU
            float diou_val = iou;
            if (diagonal_sq > 0) {
                diou_val = iou - center_dist_sq / diagonal_sq;
            }
            
            result_data[n * K + k] = diou_val;
        }
    }
    
    size_t shape[2] = {N, K};
    nb::capsule owner(result_data, [](void* p) noexcept { delete[] static_cast<float*>(p); });
    
    return nb::ndarray<nb::numpy, float, nb::shape<-1, -1>>(
        result_data, 2, shape, owner
    );
}

/**
 * Compute Complete IoU (CIoU) between two sets of boxes.
 */
nb::ndarray<nb::numpy, float, nb::shape<-1, -1>> complete_iou(
    FloatArray2DIn boxes,
    FloatArray2DIn query_boxes
) {
    size_t N = boxes.shape(0);
    size_t K = query_boxes.shape(0);
    
    float* result_data = new float[N * K]();
    
    const float* boxes_ptr = boxes.data();
    const float* query_ptr = query_boxes.data();
    
    const float pi = 3.14159265359f;
    
    for (size_t k = 0; k < K; ++k) {
        float query_area = (query_ptr[k * 4 + 2] - query_ptr[k * 4 + 0]) *
                          (query_ptr[k * 4 + 3] - query_ptr[k * 4 + 1]);
        float query_cx = (query_ptr[k * 4 + 0] + query_ptr[k * 4 + 2]) / 2.0f;
        float query_cy = (query_ptr[k * 4 + 1] + query_ptr[k * 4 + 3]) / 2.0f;
        float query_w = query_ptr[k * 4 + 2] - query_ptr[k * 4 + 0];
        float query_h = query_ptr[k * 4 + 3] - query_ptr[k * 4 + 1];
        
        for (size_t n = 0; n < N; ++n) {
            // Calculate intersection
            float iw = std::min(boxes_ptr[n * 4 + 2], query_ptr[k * 4 + 2]) -
                      std::max(boxes_ptr[n * 4 + 0], query_ptr[k * 4 + 0]);
            float ih = std::min(boxes_ptr[n * 4 + 3], query_ptr[k * 4 + 3]) -
                      std::max(boxes_ptr[n * 4 + 1], query_ptr[k * 4 + 1]);
            
            float inter = 0.0f;
            if (iw > 0 && ih > 0) {
                inter = iw * ih;
            }
            
            // Calculate union and IoU
            float box_area = (boxes_ptr[n * 4 + 2] - boxes_ptr[n * 4 + 0]) *
                            (boxes_ptr[n * 4 + 3] - boxes_ptr[n * 4 + 1]);
            float ua = box_area + query_area - inter;
            
            float iou = 0.0f;
            if (ua > 0) {
                iou = inter / ua;
            }
            
            // Calculate box centers and dimensions
            float box_cx = (boxes_ptr[n * 4 + 0] + boxes_ptr[n * 4 + 2]) / 2.0f;
            float box_cy = (boxes_ptr[n * 4 + 1] + boxes_ptr[n * 4 + 3]) / 2.0f;
            float box_w = boxes_ptr[n * 4 + 2] - boxes_ptr[n * 4 + 0];
            float box_h = boxes_ptr[n * 4 + 3] - boxes_ptr[n * 4 + 1];
            
            // Calculate center distance squared
            float center_dist_sq = (box_cx - query_cx) * (box_cx - query_cx) +
                                  (box_cy - query_cy) * (box_cy - query_cy);
            
            // Calculate smallest enclosing box diagonal squared
            float enc_x1 = std::min(boxes_ptr[n * 4 + 0], query_ptr[k * 4 + 0]);
            float enc_y1 = std::min(boxes_ptr[n * 4 + 1], query_ptr[k * 4 + 1]);
            float enc_x2 = std::max(boxes_ptr[n * 4 + 2], query_ptr[k * 4 + 2]);
            float enc_y2 = std::max(boxes_ptr[n * 4 + 3], query_ptr[k * 4 + 3]);
            
            float diagonal_sq = (enc_x2 - enc_x1) * (enc_x2 - enc_x1) +
                               (enc_y2 - enc_y1) * (enc_y2 - enc_y1);
            
            // Calculate aspect ratio consistency v
            float v = 0.0f;
            if (query_w > 0 && query_h > 0 && box_w > 0 && box_h > 0) {
                float atan_diff = std::atan2(query_w, query_h) - std::atan2(box_w, box_h);
                v = (4.0f / (pi * pi)) * atan_diff * atan_diff;
            }
            
            // Calculate alpha parameter
            float alpha = 0.0f;
            if (iou > 0) {
                alpha = v / (1 - iou + v + 1e-8f);
            }
            
            // Calculate CIoU
            float ciou_val = iou;
            if (diagonal_sq > 0) {
                ciou_val = iou - center_dist_sq / diagonal_sq - alpha * v;
            }
            
            result_data[n * K + k] = ciou_val;
        }
    }
    
    size_t shape[2] = {N, K};
    nb::capsule owner(result_data, [](void* p) noexcept { delete[] static_cast<float*>(p); });
    
    return nb::ndarray<nb::numpy, float, nb::shape<-1, -1>>(
        result_data, 2, shape, owner
    );
}

/**
 * Compute Efficient IoU (EIoU) between two sets of boxes.
 */
nb::ndarray<nb::numpy, float, nb::shape<-1, -1>> efficient_iou(
    FloatArray2DIn boxes,
    FloatArray2DIn query_boxes
) {
    size_t N = boxes.shape(0);
    size_t K = query_boxes.shape(0);
    
    float* result_data = new float[N * K]();
    
    const float* boxes_ptr = boxes.data();
    const float* query_ptr = query_boxes.data();
    
    for (size_t k = 0; k < K; ++k) {
        float query_area = (query_ptr[k * 4 + 2] - query_ptr[k * 4 + 0]) *
                          (query_ptr[k * 4 + 3] - query_ptr[k * 4 + 1]);
        float query_cx = (query_ptr[k * 4 + 0] + query_ptr[k * 4 + 2]) / 2.0f;
        float query_cy = (query_ptr[k * 4 + 1] + query_ptr[k * 4 + 3]) / 2.0f;
        float query_w = query_ptr[k * 4 + 2] - query_ptr[k * 4 + 0];
        float query_h = query_ptr[k * 4 + 3] - query_ptr[k * 4 + 1];
        
        for (size_t n = 0; n < N; ++n) {
            // Calculate intersection
            float iw = std::min(boxes_ptr[n * 4 + 2], query_ptr[k * 4 + 2]) -
                      std::max(boxes_ptr[n * 4 + 0], query_ptr[k * 4 + 0]);
            float ih = std::min(boxes_ptr[n * 4 + 3], query_ptr[k * 4 + 3]) -
                      std::max(boxes_ptr[n * 4 + 1], query_ptr[k * 4 + 1]);
            
            float inter = 0.0f;
            if (iw > 0 && ih > 0) {
                inter = iw * ih;
            }
            
            // Calculate union and IoU
            float box_area = (boxes_ptr[n * 4 + 2] - boxes_ptr[n * 4 + 0]) *
                            (boxes_ptr[n * 4 + 3] - boxes_ptr[n * 4 + 1]);
            float ua = box_area + query_area - inter;
            
            float iou = 0.0f;
            if (ua > 0) {
                iou = inter / ua;
            }
            
            // Calculate box centers and dimensions
            float box_cx = (boxes_ptr[n * 4 + 0] + boxes_ptr[n * 4 + 2]) / 2.0f;
            float box_cy = (boxes_ptr[n * 4 + 1] + boxes_ptr[n * 4 + 3]) / 2.0f;
            float box_w = boxes_ptr[n * 4 + 2] - boxes_ptr[n * 4 + 0];
            float box_h = boxes_ptr[n * 4 + 3] - boxes_ptr[n * 4 + 1];
            
            // Calculate center distance squared
            float center_dist_sq = (box_cx - query_cx) * (box_cx - query_cx) +
                                  (box_cy - query_cy) * (box_cy - query_cy);
            
            // Calculate smallest enclosing box
            float enc_x1 = std::min(boxes_ptr[n * 4 + 0], query_ptr[k * 4 + 0]);
            float enc_y1 = std::min(boxes_ptr[n * 4 + 1], query_ptr[k * 4 + 1]);
            float enc_x2 = std::max(boxes_ptr[n * 4 + 2], query_ptr[k * 4 + 2]);
            float enc_y2 = std::max(boxes_ptr[n * 4 + 3], query_ptr[k * 4 + 3]);
            
            float diagonal_sq = (enc_x2 - enc_x1) * (enc_x2 - enc_x1) +
                               (enc_y2 - enc_y1) * (enc_y2 - enc_y1);
            
            float enc_w = enc_x2 - enc_x1;
            float enc_h = enc_y2 - enc_y1;
            float enc_w_sq = enc_w * enc_w;
            float enc_h_sq = enc_h * enc_h;
            
            // Calculate width and height differences squared
            float width_diff_sq = (box_w - query_w) * (box_w - query_w);
            float height_diff_sq = (box_h - query_h) * (box_h - query_h);
            
            // Calculate EIoU
            float eiou_val = iou;
            if (diagonal_sq > 0) {
                eiou_val -= center_dist_sq / diagonal_sq;
            }
            if (enc_w_sq > 0) {
                eiou_val -= width_diff_sq / enc_w_sq;
            }
            if (enc_h_sq > 0) {
                eiou_val -= height_diff_sq / enc_h_sq;
            }
            
            result_data[n * K + k] = eiou_val;
        }
    }
    
    size_t shape[2] = {N, K};
    nb::capsule owner(result_data, [](void* p) noexcept { delete[] static_cast<float*>(p); });
    
    return nb::ndarray<nb::numpy, float, nb::shape<-1, -1>>(
        result_data, 2, shape, owner
    );
}

/**
 * Compute Normalized Wasserstein Distance (NWD) between two sets of boxes.
 */
nb::ndarray<nb::numpy, float, nb::shape<-1, -1>> normalized_wasserstein_distance(
    FloatArray2DIn boxes,
    FloatArray2DIn query_boxes,
    float tau = 1.0f
) {
    size_t N = boxes.shape(0);
    size_t K = query_boxes.shape(0);
    
    float* result_data = new float[N * K]();
    
    const float* boxes_ptr = boxes.data();
    const float* query_ptr = query_boxes.data();
    
    for (size_t k = 0; k < K; ++k) {
        float query_cx = (query_ptr[k * 4 + 0] + query_ptr[k * 4 + 2]) / 2.0f;
        float query_cy = (query_ptr[k * 4 + 1] + query_ptr[k * 4 + 3]) / 2.0f;
        float query_w_half = (query_ptr[k * 4 + 2] - query_ptr[k * 4 + 0]) / 2.0f;
        float query_h_half = (query_ptr[k * 4 + 3] - query_ptr[k * 4 + 1]) / 2.0f;
        
        for (size_t n = 0; n < N; ++n) {
            // Calculate box centers and half-dimensions
            float box_cx = (boxes_ptr[n * 4 + 0] + boxes_ptr[n * 4 + 2]) / 2.0f;
            float box_cy = (boxes_ptr[n * 4 + 1] + boxes_ptr[n * 4 + 3]) / 2.0f;
            float box_w_half = (boxes_ptr[n * 4 + 2] - boxes_ptr[n * 4 + 0]) / 2.0f;
            float box_h_half = (boxes_ptr[n * 4 + 3] - boxes_ptr[n * 4 + 1]) / 2.0f;
            
            // Mean difference squared
            float mean_diff_sq = (box_cx - query_cx) * (box_cx - query_cx) +
                                (box_cy - query_cy) * (box_cy - query_cy);
            
            // Covariance difference squared
            float cov_diff_sq = (box_w_half - query_w_half) * (box_w_half - query_w_half) +
                               (box_h_half - query_h_half) * (box_h_half - query_h_half);
            
            // Wasserstein-2 distance squared
            float wasserstein_sq = mean_diff_sq + cov_diff_sq;
            
            // Apply exponential normalization
            float nwd_val = 1.0f;
            if (wasserstein_sq >= 0) {
                nwd_val = std::exp(-std::sqrt(wasserstein_sq) / tau);
            }
            
            result_data[n * K + k] = nwd_val;
        }
    }
    
    size_t shape[2] = {N, K};
    nb::capsule owner(result_data, [](void* p) noexcept { delete[] static_cast<float*>(p); });
    
    return nb::ndarray<nb::numpy, float, nb::shape<-1, -1>>(
        result_data, 2, shape, owner
    );
}

NB_MODULE(bbox, m) {
    m.doc() = "Fast bounding box IoU calculations using nanobind";
    
    m.def("bbox_overlaps", &bbox_overlaps,
          nb::arg("boxes"), nb::arg("query_boxes"),
          "Compute IoU overlaps between two sets of boxes.\n\n"
          "Parameters\n"
          "----------\n"
          "boxes: (N, 4) float32 array [x1, y1, x2, y2]\n"
          "query_boxes: (K, 4) float32 array [x1, y1, x2, y2]\n\n"
          "Returns\n"
          "-------\n"
          "overlaps: (N, K) float32 array of IoU");
    
    m.def("generalized_iou", &generalized_iou,
          nb::arg("boxes"), nb::arg("query_boxes"),
          "Compute Generalized IoU (GIoU) between two sets of boxes.");
    
    m.def("distance_iou", &distance_iou,
          nb::arg("boxes"), nb::arg("query_boxes"),
          "Compute Distance IoU (DIoU) between two sets of boxes.");
    
    m.def("complete_iou", &complete_iou,
          nb::arg("boxes"), nb::arg("query_boxes"),
          "Compute Complete IoU (CIoU) between two sets of boxes.");
    
    m.def("efficient_iou", &efficient_iou,
          nb::arg("boxes"), nb::arg("query_boxes"),
          "Compute Efficient IoU (EIoU) between two sets of boxes.");
    
    m.def("normalized_wasserstein_distance", &normalized_wasserstein_distance,
          nb::arg("boxes"), nb::arg("query_boxes"), nb::arg("tau") = 1.0f,
          "Compute Normalized Wasserstein Distance (NWD) between two sets of boxes.");
}
