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

using DoubleArray2DIn = nb::ndarray<const double, nb::shape<-1, 4>, nb::c_contig, nb::device::cpu>;

/**
 * Compute IoU overlaps between two sets of boxes.
 *
 * Parameters
 * ----------
 * boxes: (N, 4) float64 array [x1, y1, x2, y2]
 * query_boxes: (K, 4) float64 array [x1, y1, x2, y2]
 *
 * Returns
 * -------
 * overlaps: (N, K) float64 array of IoU
 */
nb::ndarray<nb::numpy, double, nb::shape<-1, -1>> bbox_overlaps(
    DoubleArray2DIn boxes,
    DoubleArray2DIn query_boxes
) {
    size_t N = boxes.shape(0);
    size_t K = query_boxes.shape(0);
    
    // Allocate output array
    double* result_data = new double[N * K]();
    
    const double* boxes_ptr = boxes.data();
    const double* query_ptr = query_boxes.data();
    
    for (size_t k = 0; k < K; ++k) {
        double query_area = (query_ptr[k * 4 + 2] - query_ptr[k * 4 + 0]) *
                          (query_ptr[k * 4 + 3] - query_ptr[k * 4 + 1]);
        
        for (size_t n = 0; n < N; ++n) {
            double iw = std::min(boxes_ptr[n * 4 + 2], query_ptr[k * 4 + 2]) -
                      std::max(boxes_ptr[n * 4 + 0], query_ptr[k * 4 + 0]);
            
            if (iw > 0) {
                double ih = std::min(boxes_ptr[n * 4 + 3], query_ptr[k * 4 + 3]) -
                          std::max(boxes_ptr[n * 4 + 1], query_ptr[k * 4 + 1]);
                
                if (ih > 0) {
                    double inter = iw * ih;
                    double box_area = (boxes_ptr[n * 4 + 2] - boxes_ptr[n * 4 + 0]) *
                                    (boxes_ptr[n * 4 + 3] - boxes_ptr[n * 4 + 1]);
                    double ua = box_area + query_area - inter;
                    result_data[n * K + k] = inter / ua;
                }
            }
        }
    }
    
    // Create output ndarray with ownership
    size_t shape[2] = {N, K};
    nb::capsule owner(result_data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
    
    return nb::ndarray<nb::numpy, double, nb::shape<-1, -1>>(
        result_data, 2, shape, owner
    );
}

/**
 * Compute Generalized IoU (GIoU) between two sets of boxes.
 */
nb::ndarray<nb::numpy, double, nb::shape<-1, -1>> generalized_iou(
    DoubleArray2DIn boxes,
    DoubleArray2DIn query_boxes
) {
    size_t N = boxes.shape(0);
    size_t K = query_boxes.shape(0);
    
    double* result_data = new double[N * K]();
    
    const double* boxes_ptr = boxes.data();
    const double* query_ptr = query_boxes.data();
    
    for (size_t k = 0; k < K; ++k) {
        double query_area = (query_ptr[k * 4 + 2] - query_ptr[k * 4 + 0]) *
                          (query_ptr[k * 4 + 3] - query_ptr[k * 4 + 1]);
        
        for (size_t n = 0; n < N; ++n) {
            // Calculate intersection
            double iw = std::min(boxes_ptr[n * 4 + 2], query_ptr[k * 4 + 2]) -
                      std::max(boxes_ptr[n * 4 + 0], query_ptr[k * 4 + 0]);
            double ih = std::min(boxes_ptr[n * 4 + 3], query_ptr[k * 4 + 3]) -
                      std::max(boxes_ptr[n * 4 + 1], query_ptr[k * 4 + 1]);
            
            double inter = 0.0;
            if (iw > 0 && ih > 0) {
                inter = iw * ih;
            }
            
            // Calculate union and IoU
            double box_area = (boxes_ptr[n * 4 + 2] - boxes_ptr[n * 4 + 0]) *
                            (boxes_ptr[n * 4 + 3] - boxes_ptr[n * 4 + 1]);
            double ua = box_area + query_area - inter;
            
            double iou = 0.0;
            if (ua > 0) {
                iou = inter / ua;
            }
            
            // Calculate smallest enclosing box
            double enc_x1 = std::min(boxes_ptr[n * 4 + 0], query_ptr[k * 4 + 0]);
            double enc_y1 = std::min(boxes_ptr[n * 4 + 1], query_ptr[k * 4 + 1]);
            double enc_x2 = std::max(boxes_ptr[n * 4 + 2], query_ptr[k * 4 + 2]);
            double enc_y2 = std::max(boxes_ptr[n * 4 + 3], query_ptr[k * 4 + 3]);
            
            double enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1);
            
            // Calculate GIoU
            double giou_val = iou;
            if (enc_area > 0) {
                double coverage_ratio = (enc_area - ua) / enc_area;
                giou_val = iou - coverage_ratio;
            }
            
            result_data[n * K + k] = giou_val;
        }
    }
    
    size_t shape[2] = {N, K};
    nb::capsule owner(result_data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
    
    return nb::ndarray<nb::numpy, double, nb::shape<-1, -1>>(
        result_data, 2, shape, owner
    );
}

/**
 * Compute Distance IoU (DIoU) between two sets of boxes.
 */
nb::ndarray<nb::numpy, double, nb::shape<-1, -1>> distance_iou(
    DoubleArray2DIn boxes,
    DoubleArray2DIn query_boxes
) {
    size_t N = boxes.shape(0);
    size_t K = query_boxes.shape(0);
    
    double* result_data = new double[N * K]();
    
    const double* boxes_ptr = boxes.data();
    const double* query_ptr = query_boxes.data();
    
    for (size_t k = 0; k < K; ++k) {
        double query_area = (query_ptr[k * 4 + 2] - query_ptr[k * 4 + 0]) *
                          (query_ptr[k * 4 + 3] - query_ptr[k * 4 + 1]);
        double query_cx = (query_ptr[k * 4 + 0] + query_ptr[k * 4 + 2]) / 2.0;
        double query_cy = (query_ptr[k * 4 + 1] + query_ptr[k * 4 + 3]) / 2.0;
        
        for (size_t n = 0; n < N; ++n) {
            // Calculate intersection
            double iw = std::min(boxes_ptr[n * 4 + 2], query_ptr[k * 4 + 2]) -
                      std::max(boxes_ptr[n * 4 + 0], query_ptr[k * 4 + 0]);
            double ih = std::min(boxes_ptr[n * 4 + 3], query_ptr[k * 4 + 3]) -
                      std::max(boxes_ptr[n * 4 + 1], query_ptr[k * 4 + 1]);
            
            double inter = 0.0;
            if (iw > 0 && ih > 0) {
                inter = iw * ih;
            }
            
            // Calculate union and IoU
            double box_area = (boxes_ptr[n * 4 + 2] - boxes_ptr[n * 4 + 0]) *
                            (boxes_ptr[n * 4 + 3] - boxes_ptr[n * 4 + 1]);
            double ua = box_area + query_area - inter;
            
            double iou = 0.0;
            if (ua > 0) {
                iou = inter / ua;
            }
            
            // Calculate box centers
            double box_cx = (boxes_ptr[n * 4 + 0] + boxes_ptr[n * 4 + 2]) / 2.0;
            double box_cy = (boxes_ptr[n * 4 + 1] + boxes_ptr[n * 4 + 3]) / 2.0;
            
            // Calculate center distance squared
            double center_dist_sq = (box_cx - query_cx) * (box_cx - query_cx) +
                                  (box_cy - query_cy) * (box_cy - query_cy);
            
            // Calculate smallest enclosing box diagonal squared
            double enc_x1 = std::min(boxes_ptr[n * 4 + 0], query_ptr[k * 4 + 0]);
            double enc_y1 = std::min(boxes_ptr[n * 4 + 1], query_ptr[k * 4 + 1]);
            double enc_x2 = std::max(boxes_ptr[n * 4 + 2], query_ptr[k * 4 + 2]);
            double enc_y2 = std::max(boxes_ptr[n * 4 + 3], query_ptr[k * 4 + 3]);
            
            double diagonal_sq = (enc_x2 - enc_x1) * (enc_x2 - enc_x1) +
                               (enc_y2 - enc_y1) * (enc_y2 - enc_y1);
            
            // Calculate DIoU
            double diou_val = iou;
            if (diagonal_sq > 0) {
                diou_val = iou - center_dist_sq / diagonal_sq;
            }
            
            result_data[n * K + k] = diou_val;
        }
    }
    
    size_t shape[2] = {N, K};
    nb::capsule owner(result_data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
    
    return nb::ndarray<nb::numpy, double, nb::shape<-1, -1>>(
        result_data, 2, shape, owner
    );
}

/**
 * Compute Complete IoU (CIoU) between two sets of boxes.
 */
nb::ndarray<nb::numpy, double, nb::shape<-1, -1>> complete_iou(
    DoubleArray2DIn boxes,
    DoubleArray2DIn query_boxes
) {
    size_t N = boxes.shape(0);
    size_t K = query_boxes.shape(0);
    
    double* result_data = new double[N * K]();
    
    const double* boxes_ptr = boxes.data();
    const double* query_ptr = query_boxes.data();
    
    constexpr double pi = 3.14159265358979323846;
    
    for (size_t k = 0; k < K; ++k) {
        double query_area = (query_ptr[k * 4 + 2] - query_ptr[k * 4 + 0]) *
                          (query_ptr[k * 4 + 3] - query_ptr[k * 4 + 1]);
        double query_cx = (query_ptr[k * 4 + 0] + query_ptr[k * 4 + 2]) / 2.0;
        double query_cy = (query_ptr[k * 4 + 1] + query_ptr[k * 4 + 3]) / 2.0;
        double query_w = query_ptr[k * 4 + 2] - query_ptr[k * 4 + 0];
        double query_h = query_ptr[k * 4 + 3] - query_ptr[k * 4 + 1];
        
        for (size_t n = 0; n < N; ++n) {
            // Calculate intersection
            double iw = std::min(boxes_ptr[n * 4 + 2], query_ptr[k * 4 + 2]) -
                      std::max(boxes_ptr[n * 4 + 0], query_ptr[k * 4 + 0]);
            double ih = std::min(boxes_ptr[n * 4 + 3], query_ptr[k * 4 + 3]) -
                      std::max(boxes_ptr[n * 4 + 1], query_ptr[k * 4 + 1]);
            
            double inter = 0.0;
            if (iw > 0 && ih > 0) {
                inter = iw * ih;
            }
            
            // Calculate union and IoU
            double box_area = (boxes_ptr[n * 4 + 2] - boxes_ptr[n * 4 + 0]) *
                            (boxes_ptr[n * 4 + 3] - boxes_ptr[n * 4 + 1]);
            double ua = box_area + query_area - inter;
            
            double iou = 0.0;
            if (ua > 0) {
                iou = inter / ua;
            }
            
            // Calculate box centers and dimensions
            double box_cx = (boxes_ptr[n * 4 + 0] + boxes_ptr[n * 4 + 2]) / 2.0;
            double box_cy = (boxes_ptr[n * 4 + 1] + boxes_ptr[n * 4 + 3]) / 2.0;
            double box_w = boxes_ptr[n * 4 + 2] - boxes_ptr[n * 4 + 0];
            double box_h = boxes_ptr[n * 4 + 3] - boxes_ptr[n * 4 + 1];
            
            // Calculate center distance squared
            double center_dist_sq = (box_cx - query_cx) * (box_cx - query_cx) +
                                  (box_cy - query_cy) * (box_cy - query_cy);
            
            // Calculate smallest enclosing box diagonal squared
            double enc_x1 = std::min(boxes_ptr[n * 4 + 0], query_ptr[k * 4 + 0]);
            double enc_y1 = std::min(boxes_ptr[n * 4 + 1], query_ptr[k * 4 + 1]);
            double enc_x2 = std::max(boxes_ptr[n * 4 + 2], query_ptr[k * 4 + 2]);
            double enc_y2 = std::max(boxes_ptr[n * 4 + 3], query_ptr[k * 4 + 3]);
            
            double diagonal_sq = (enc_x2 - enc_x1) * (enc_x2 - enc_x1) +
                               (enc_y2 - enc_y1) * (enc_y2 - enc_y1);
            
            // Calculate aspect ratio consistency v
            double v = 0.0;
            if (query_w > 0 && query_h > 0 && box_w > 0 && box_h > 0) {
                double atan_diff = std::atan2(query_w, query_h) - std::atan2(box_w, box_h);
                v = (4.0 / (pi * pi)) * atan_diff * atan_diff;
            }
            
            // Calculate alpha parameter
            double alpha = 0.0;
            if (iou > 0) {
                alpha = v / (1 - iou + v + 1e-8);
            }
            
            // Calculate CIoU
            double ciou_val = iou;
            if (diagonal_sq > 0) {
                ciou_val = iou - center_dist_sq / diagonal_sq - alpha * v;
            }
            
            result_data[n * K + k] = ciou_val;
        }
    }
    
    size_t shape[2] = {N, K};
    nb::capsule owner(result_data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
    
    return nb::ndarray<nb::numpy, double, nb::shape<-1, -1>>(
        result_data, 2, shape, owner
    );
}

/**
 * Compute Efficient IoU (EIoU) between two sets of boxes.
 */
nb::ndarray<nb::numpy, double, nb::shape<-1, -1>> efficient_iou(
    DoubleArray2DIn boxes,
    DoubleArray2DIn query_boxes
) {
    size_t N = boxes.shape(0);
    size_t K = query_boxes.shape(0);
    
    double* result_data = new double[N * K]();
    
    const double* boxes_ptr = boxes.data();
    const double* query_ptr = query_boxes.data();
    
    for (size_t k = 0; k < K; ++k) {
        double query_area = (query_ptr[k * 4 + 2] - query_ptr[k * 4 + 0]) *
                          (query_ptr[k * 4 + 3] - query_ptr[k * 4 + 1]);
        double query_cx = (query_ptr[k * 4 + 0] + query_ptr[k * 4 + 2]) / 2.0;
        double query_cy = (query_ptr[k * 4 + 1] + query_ptr[k * 4 + 3]) / 2.0;
        double query_w = query_ptr[k * 4 + 2] - query_ptr[k * 4 + 0];
        double query_h = query_ptr[k * 4 + 3] - query_ptr[k * 4 + 1];
        
        for (size_t n = 0; n < N; ++n) {
            // Calculate intersection
            double iw = std::min(boxes_ptr[n * 4 + 2], query_ptr[k * 4 + 2]) -
                      std::max(boxes_ptr[n * 4 + 0], query_ptr[k * 4 + 0]);
            double ih = std::min(boxes_ptr[n * 4 + 3], query_ptr[k * 4 + 3]) -
                      std::max(boxes_ptr[n * 4 + 1], query_ptr[k * 4 + 1]);
            
            double inter = 0.0;
            if (iw > 0 && ih > 0) {
                inter = iw * ih;
            }
            
            // Calculate union and IoU
            double box_area = (boxes_ptr[n * 4 + 2] - boxes_ptr[n * 4 + 0]) *
                            (boxes_ptr[n * 4 + 3] - boxes_ptr[n * 4 + 1]);
            double ua = box_area + query_area - inter;
            
            double iou = 0.0;
            if (ua > 0) {
                iou = inter / ua;
            }
            
            // Calculate box centers and dimensions
            double box_cx = (boxes_ptr[n * 4 + 0] + boxes_ptr[n * 4 + 2]) / 2.0;
            double box_cy = (boxes_ptr[n * 4 + 1] + boxes_ptr[n * 4 + 3]) / 2.0;
            double box_w = boxes_ptr[n * 4 + 2] - boxes_ptr[n * 4 + 0];
            double box_h = boxes_ptr[n * 4 + 3] - boxes_ptr[n * 4 + 1];
            
            // Calculate center distance squared
            double center_dist_sq = (box_cx - query_cx) * (box_cx - query_cx) +
                                  (box_cy - query_cy) * (box_cy - query_cy);
            
            // Calculate smallest enclosing box
            double enc_x1 = std::min(boxes_ptr[n * 4 + 0], query_ptr[k * 4 + 0]);
            double enc_y1 = std::min(boxes_ptr[n * 4 + 1], query_ptr[k * 4 + 1]);
            double enc_x2 = std::max(boxes_ptr[n * 4 + 2], query_ptr[k * 4 + 2]);
            double enc_y2 = std::max(boxes_ptr[n * 4 + 3], query_ptr[k * 4 + 3]);
            
            double diagonal_sq = (enc_x2 - enc_x1) * (enc_x2 - enc_x1) +
                               (enc_y2 - enc_y1) * (enc_y2 - enc_y1);
            
            double enc_w = enc_x2 - enc_x1;
            double enc_h = enc_y2 - enc_y1;
            double enc_w_sq = enc_w * enc_w;
            double enc_h_sq = enc_h * enc_h;
            
            // Calculate width and height differences squared
            double width_diff_sq = (box_w - query_w) * (box_w - query_w);
            double height_diff_sq = (box_h - query_h) * (box_h - query_h);
            
            // Calculate EIoU
            double eiou_val = iou;
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
    nb::capsule owner(result_data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
    
    return nb::ndarray<nb::numpy, double, nb::shape<-1, -1>>(
        result_data, 2, shape, owner
    );
}

/**
 * Compute Normalized Wasserstein Distance (NWD) between two sets of boxes.
 */
nb::ndarray<nb::numpy, double, nb::shape<-1, -1>> normalized_wasserstein_distance(
    DoubleArray2DIn boxes,
    DoubleArray2DIn query_boxes,
    double tau = 1.0
) {
    size_t N = boxes.shape(0);
    size_t K = query_boxes.shape(0);
    
    double* result_data = new double[N * K]();
    
    const double* boxes_ptr = boxes.data();
    const double* query_ptr = query_boxes.data();
    
    for (size_t k = 0; k < K; ++k) {
        double query_cx = (query_ptr[k * 4 + 0] + query_ptr[k * 4 + 2]) / 2.0;
        double query_cy = (query_ptr[k * 4 + 1] + query_ptr[k * 4 + 3]) / 2.0;
        double query_w_half = (query_ptr[k * 4 + 2] - query_ptr[k * 4 + 0]) / 2.0;
        double query_h_half = (query_ptr[k * 4 + 3] - query_ptr[k * 4 + 1]) / 2.0;
        
        for (size_t n = 0; n < N; ++n) {
            // Calculate box centers and half-dimensions
            double box_cx = (boxes_ptr[n * 4 + 0] + boxes_ptr[n * 4 + 2]) / 2.0;
            double box_cy = (boxes_ptr[n * 4 + 1] + boxes_ptr[n * 4 + 3]) / 2.0;
            double box_w_half = (boxes_ptr[n * 4 + 2] - boxes_ptr[n * 4 + 0]) / 2.0;
            double box_h_half = (boxes_ptr[n * 4 + 3] - boxes_ptr[n * 4 + 1]) / 2.0;
            
            // Mean difference squared
            double mean_diff_sq = (box_cx - query_cx) * (box_cx - query_cx) +
                                (box_cy - query_cy) * (box_cy - query_cy);
            
            // Covariance difference squared
            double cov_diff_sq = (box_w_half - query_w_half) * (box_w_half - query_w_half) +
                               (box_h_half - query_h_half) * (box_h_half - query_h_half);
            
            // Wasserstein-2 distance squared
            double wasserstein_sq = mean_diff_sq + cov_diff_sq;
            
            // Apply exponential normalization
            double nwd_val = 1.0;
            if (wasserstein_sq >= 0) {
                nwd_val = std::exp(-std::sqrt(wasserstein_sq) / tau);
            }
            
            result_data[n * K + k] = nwd_val;
        }
    }
    
    size_t shape[2] = {N, K};
    nb::capsule owner(result_data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
    
    return nb::ndarray<nb::numpy, double, nb::shape<-1, -1>>(
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
          "boxes: (N, 4) float64 array [x1, y1, x2, y2]\n"
          "query_boxes: (K, 4) float64 array [x1, y1, x2, y2]\n\n"
          "Returns\n"
          "-------\n"
          "overlaps: (N, K) float64 array of IoU");
    
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
          nb::arg("boxes"), nb::arg("query_boxes"), nb::arg("tau") = 1.0,
          "Compute Normalized Wasserstein Distance (NWD) between two sets of boxes.");
}
