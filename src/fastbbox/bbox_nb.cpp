/**
 * Nanobind implementation of bounding box IoU calculations.
 *
 * Accepts float32 and float16 input arrays.  All computation is performed
 * in float32; output is always float32.
 */

#include "fastbbox_fp16.h"
#include <cmath>
#include <algorithm>

/**
 * Compute IoU overlaps between two sets of boxes.
 *
 * Parameters
 * ----------
 * boxes: (N, 4) float32 or float16 array [x1, y1, x2, y2]
 * query_boxes: (K, 4) float32 or float16 array [x1, y1, x2, y2]
 *
 * Returns
 * -------
 * overlaps: (N, K) float32 array of IoU
 */
nb::ndarray<nb::numpy, float, nb::shape<-1, -1>> bbox_overlaps(
    Array2D4 boxes,
    Array2D4 query_boxes
) {
    const size_t N = boxes.shape(0);
    const size_t K = query_boxes.shape(0);

    F32View<Array2D4> bv(boxes);
    F32View<Array2D4> qv(query_boxes);
    const float* boxes_ptr = bv.ptr;
    const float* query_ptr = qv.ptr;

    float* result_data = new float[N * K]();

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
                    float inter    = iw * ih;
                    float box_area = (boxes_ptr[n * 4 + 2] - boxes_ptr[n * 4 + 0]) *
                                     (boxes_ptr[n * 4 + 3] - boxes_ptr[n * 4 + 1]);
                    float ua = box_area + query_area - inter;
                    result_data[n * K + k] = inter / ua;
                }
            }
        }
    }

    size_t shape[2] = {N, K};
    nb::capsule owner(result_data, [](void* p) noexcept { delete[] static_cast<float*>(p); });
    return nb::ndarray<nb::numpy, float, nb::shape<-1, -1>>(result_data, 2, shape, owner);
}

/**
 * Compute Generalized IoU (GIoU) between two sets of boxes.
 */
nb::ndarray<nb::numpy, float, nb::shape<-1, -1>> generalized_iou(
    Array2D4 boxes,
    Array2D4 query_boxes
) {
    const size_t N = boxes.shape(0);
    const size_t K = query_boxes.shape(0);

    F32View<Array2D4> bv(boxes);
    F32View<Array2D4> qv(query_boxes);
    const float* boxes_ptr = bv.ptr;
    const float* query_ptr = qv.ptr;

    float* result_data = new float[N * K]();

    for (size_t k = 0; k < K; ++k) {
        float query_area = (query_ptr[k * 4 + 2] - query_ptr[k * 4 + 0]) *
                           (query_ptr[k * 4 + 3] - query_ptr[k * 4 + 1]);

        for (size_t n = 0; n < N; ++n) {
            float iw = std::min(boxes_ptr[n * 4 + 2], query_ptr[k * 4 + 2]) -
                       std::max(boxes_ptr[n * 4 + 0], query_ptr[k * 4 + 0]);
            float ih = std::min(boxes_ptr[n * 4 + 3], query_ptr[k * 4 + 3]) -
                       std::max(boxes_ptr[n * 4 + 1], query_ptr[k * 4 + 1]);

            float inter = (iw > 0 && ih > 0) ? iw * ih : 0.0f;

            float box_area = (boxes_ptr[n * 4 + 2] - boxes_ptr[n * 4 + 0]) *
                             (boxes_ptr[n * 4 + 3] - boxes_ptr[n * 4 + 1]);
            float ua  = box_area + query_area - inter;
            float iou = (ua > 0) ? inter / ua : 0.0f;

            float enc_x1 = std::min(boxes_ptr[n * 4 + 0], query_ptr[k * 4 + 0]);
            float enc_y1 = std::min(boxes_ptr[n * 4 + 1], query_ptr[k * 4 + 1]);
            float enc_x2 = std::max(boxes_ptr[n * 4 + 2], query_ptr[k * 4 + 2]);
            float enc_y2 = std::max(boxes_ptr[n * 4 + 3], query_ptr[k * 4 + 3]);
            float enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1);

            float giou_val = iou;
            if (enc_area > 0)
                giou_val = iou - (enc_area - ua) / enc_area;

            result_data[n * K + k] = giou_val;
        }
    }

    size_t shape[2] = {N, K};
    nb::capsule owner(result_data, [](void* p) noexcept { delete[] static_cast<float*>(p); });
    return nb::ndarray<nb::numpy, float, nb::shape<-1, -1>>(result_data, 2, shape, owner);
}

/**
 * Compute Distance IoU (DIoU) between two sets of boxes.
 */
nb::ndarray<nb::numpy, float, nb::shape<-1, -1>> distance_iou(
    Array2D4 boxes,
    Array2D4 query_boxes
) {
    const size_t N = boxes.shape(0);
    const size_t K = query_boxes.shape(0);

    F32View<Array2D4> bv(boxes);
    F32View<Array2D4> qv(query_boxes);
    const float* boxes_ptr = bv.ptr;
    const float* query_ptr = qv.ptr;

    float* result_data = new float[N * K]();

    for (size_t k = 0; k < K; ++k) {
        float query_area = (query_ptr[k * 4 + 2] - query_ptr[k * 4 + 0]) *
                           (query_ptr[k * 4 + 3] - query_ptr[k * 4 + 1]);
        float query_cx   = (query_ptr[k * 4 + 0] + query_ptr[k * 4 + 2]) / 2.0f;
        float query_cy   = (query_ptr[k * 4 + 1] + query_ptr[k * 4 + 3]) / 2.0f;

        for (size_t n = 0; n < N; ++n) {
            float iw = std::min(boxes_ptr[n * 4 + 2], query_ptr[k * 4 + 2]) -
                       std::max(boxes_ptr[n * 4 + 0], query_ptr[k * 4 + 0]);
            float ih = std::min(boxes_ptr[n * 4 + 3], query_ptr[k * 4 + 3]) -
                       std::max(boxes_ptr[n * 4 + 1], query_ptr[k * 4 + 1]);

            float inter = (iw > 0 && ih > 0) ? iw * ih : 0.0f;

            float box_area = (boxes_ptr[n * 4 + 2] - boxes_ptr[n * 4 + 0]) *
                             (boxes_ptr[n * 4 + 3] - boxes_ptr[n * 4 + 1]);
            float ua  = box_area + query_area - inter;
            float iou = (ua > 0) ? inter / ua : 0.0f;

            float box_cx = (boxes_ptr[n * 4 + 0] + boxes_ptr[n * 4 + 2]) / 2.0f;
            float box_cy = (boxes_ptr[n * 4 + 1] + boxes_ptr[n * 4 + 3]) / 2.0f;

            float center_dist_sq = (box_cx - query_cx) * (box_cx - query_cx) +
                                   (box_cy - query_cy) * (box_cy - query_cy);

            float enc_x1 = std::min(boxes_ptr[n * 4 + 0], query_ptr[k * 4 + 0]);
            float enc_y1 = std::min(boxes_ptr[n * 4 + 1], query_ptr[k * 4 + 1]);
            float enc_x2 = std::max(boxes_ptr[n * 4 + 2], query_ptr[k * 4 + 2]);
            float enc_y2 = std::max(boxes_ptr[n * 4 + 3], query_ptr[k * 4 + 3]);

            float diagonal_sq = (enc_x2 - enc_x1) * (enc_x2 - enc_x1) +
                                 (enc_y2 - enc_y1) * (enc_y2 - enc_y1);

            float diou_val = iou;
            if (diagonal_sq > 0)
                diou_val = iou - center_dist_sq / diagonal_sq;

            result_data[n * K + k] = diou_val;
        }
    }

    size_t shape[2] = {N, K};
    nb::capsule owner(result_data, [](void* p) noexcept { delete[] static_cast<float*>(p); });
    return nb::ndarray<nb::numpy, float, nb::shape<-1, -1>>(result_data, 2, shape, owner);
}

/**
 * Compute Complete IoU (CIoU) between two sets of boxes.
 */
nb::ndarray<nb::numpy, float, nb::shape<-1, -1>> complete_iou(
    Array2D4 boxes,
    Array2D4 query_boxes
) {
    const size_t N = boxes.shape(0);
    const size_t K = query_boxes.shape(0);

    F32View<Array2D4> bv(boxes);
    F32View<Array2D4> qv(query_boxes);
    const float* boxes_ptr = bv.ptr;
    const float* query_ptr = qv.ptr;

    float* result_data = new float[N * K]();

    constexpr float pi = 3.14159265359f;

    for (size_t k = 0; k < K; ++k) {
        float query_area = (query_ptr[k * 4 + 2] - query_ptr[k * 4 + 0]) *
                           (query_ptr[k * 4 + 3] - query_ptr[k * 4 + 1]);
        float query_cx   = (query_ptr[k * 4 + 0] + query_ptr[k * 4 + 2]) / 2.0f;
        float query_cy   = (query_ptr[k * 4 + 1] + query_ptr[k * 4 + 3]) / 2.0f;
        float query_w    = query_ptr[k * 4 + 2] - query_ptr[k * 4 + 0];
        float query_h    = query_ptr[k * 4 + 3] - query_ptr[k * 4 + 1];

        for (size_t n = 0; n < N; ++n) {
            float iw = std::min(boxes_ptr[n * 4 + 2], query_ptr[k * 4 + 2]) -
                       std::max(boxes_ptr[n * 4 + 0], query_ptr[k * 4 + 0]);
            float ih = std::min(boxes_ptr[n * 4 + 3], query_ptr[k * 4 + 3]) -
                       std::max(boxes_ptr[n * 4 + 1], query_ptr[k * 4 + 1]);

            float inter = (iw > 0 && ih > 0) ? iw * ih : 0.0f;

            float box_area = (boxes_ptr[n * 4 + 2] - boxes_ptr[n * 4 + 0]) *
                             (boxes_ptr[n * 4 + 3] - boxes_ptr[n * 4 + 1]);
            float ua  = box_area + query_area - inter;
            float iou = (ua > 0) ? inter / ua : 0.0f;

            float box_cx = (boxes_ptr[n * 4 + 0] + boxes_ptr[n * 4 + 2]) / 2.0f;
            float box_cy = (boxes_ptr[n * 4 + 1] + boxes_ptr[n * 4 + 3]) / 2.0f;
            float box_w  = boxes_ptr[n * 4 + 2] - boxes_ptr[n * 4 + 0];
            float box_h  = boxes_ptr[n * 4 + 3] - boxes_ptr[n * 4 + 1];

            float center_dist_sq = (box_cx - query_cx) * (box_cx - query_cx) +
                                   (box_cy - query_cy) * (box_cy - query_cy);

            float enc_x1 = std::min(boxes_ptr[n * 4 + 0], query_ptr[k * 4 + 0]);
            float enc_y1 = std::min(boxes_ptr[n * 4 + 1], query_ptr[k * 4 + 1]);
            float enc_x2 = std::max(boxes_ptr[n * 4 + 2], query_ptr[k * 4 + 2]);
            float enc_y2 = std::max(boxes_ptr[n * 4 + 3], query_ptr[k * 4 + 3]);

            float diagonal_sq = (enc_x2 - enc_x1) * (enc_x2 - enc_x1) +
                                 (enc_y2 - enc_y1) * (enc_y2 - enc_y1);

            float v = 0.0f;
            if (query_w > 0 && query_h > 0 && box_w > 0 && box_h > 0) {
                float atan_diff = std::atan2(query_w, query_h) - std::atan2(box_w, box_h);
                v = (4.0f / (pi * pi)) * atan_diff * atan_diff;
            }

            float alpha    = (iou > 0) ? v / (1.0f - iou + v + 1e-8f) : 0.0f;
            float ciou_val = iou;
            if (diagonal_sq > 0)
                ciou_val = iou - center_dist_sq / diagonal_sq - alpha * v;

            result_data[n * K + k] = ciou_val;
        }
    }

    size_t shape[2] = {N, K};
    nb::capsule owner(result_data, [](void* p) noexcept { delete[] static_cast<float*>(p); });
    return nb::ndarray<nb::numpy, float, nb::shape<-1, -1>>(result_data, 2, shape, owner);
}

/**
 * Compute Efficient IoU (EIoU) between two sets of boxes.
 */
nb::ndarray<nb::numpy, float, nb::shape<-1, -1>> efficient_iou(
    Array2D4 boxes,
    Array2D4 query_boxes
) {
    const size_t N = boxes.shape(0);
    const size_t K = query_boxes.shape(0);

    F32View<Array2D4> bv(boxes);
    F32View<Array2D4> qv(query_boxes);
    const float* boxes_ptr = bv.ptr;
    const float* query_ptr = qv.ptr;

    float* result_data = new float[N * K]();

    for (size_t k = 0; k < K; ++k) {
        float query_area = (query_ptr[k * 4 + 2] - query_ptr[k * 4 + 0]) *
                           (query_ptr[k * 4 + 3] - query_ptr[k * 4 + 1]);
        float query_cx   = (query_ptr[k * 4 + 0] + query_ptr[k * 4 + 2]) / 2.0f;
        float query_cy   = (query_ptr[k * 4 + 1] + query_ptr[k * 4 + 3]) / 2.0f;
        float query_w    = query_ptr[k * 4 + 2] - query_ptr[k * 4 + 0];
        float query_h    = query_ptr[k * 4 + 3] - query_ptr[k * 4 + 1];

        for (size_t n = 0; n < N; ++n) {
            float iw = std::min(boxes_ptr[n * 4 + 2], query_ptr[k * 4 + 2]) -
                       std::max(boxes_ptr[n * 4 + 0], query_ptr[k * 4 + 0]);
            float ih = std::min(boxes_ptr[n * 4 + 3], query_ptr[k * 4 + 3]) -
                       std::max(boxes_ptr[n * 4 + 1], query_ptr[k * 4 + 1]);

            float inter = (iw > 0 && ih > 0) ? iw * ih : 0.0f;

            float box_area = (boxes_ptr[n * 4 + 2] - boxes_ptr[n * 4 + 0]) *
                             (boxes_ptr[n * 4 + 3] - boxes_ptr[n * 4 + 1]);
            float ua  = box_area + query_area - inter;
            float iou = (ua > 0) ? inter / ua : 0.0f;

            float box_cx = (boxes_ptr[n * 4 + 0] + boxes_ptr[n * 4 + 2]) / 2.0f;
            float box_cy = (boxes_ptr[n * 4 + 1] + boxes_ptr[n * 4 + 3]) / 2.0f;
            float box_w  = boxes_ptr[n * 4 + 2] - boxes_ptr[n * 4 + 0];
            float box_h  = boxes_ptr[n * 4 + 3] - boxes_ptr[n * 4 + 1];

            float center_dist_sq = (box_cx - query_cx) * (box_cx - query_cx) +
                                   (box_cy - query_cy) * (box_cy - query_cy);

            float enc_x1   = std::min(boxes_ptr[n * 4 + 0], query_ptr[k * 4 + 0]);
            float enc_y1   = std::min(boxes_ptr[n * 4 + 1], query_ptr[k * 4 + 1]);
            float enc_x2   = std::max(boxes_ptr[n * 4 + 2], query_ptr[k * 4 + 2]);
            float enc_y2   = std::max(boxes_ptr[n * 4 + 3], query_ptr[k * 4 + 3]);
            float enc_w    = enc_x2 - enc_x1;
            float enc_h    = enc_y2 - enc_y1;
            float diag_sq  = enc_w * enc_w + enc_h * enc_h;
            float enc_w_sq = enc_w * enc_w;
            float enc_h_sq = enc_h * enc_h;

            float width_diff_sq  = (box_w - query_w) * (box_w - query_w);
            float height_diff_sq = (box_h - query_h) * (box_h - query_h);

            float eiou_val = iou;
            if (diag_sq  > 0) eiou_val -= center_dist_sq  / diag_sq;
            if (enc_w_sq > 0) eiou_val -= width_diff_sq   / enc_w_sq;
            if (enc_h_sq > 0) eiou_val -= height_diff_sq  / enc_h_sq;

            result_data[n * K + k] = eiou_val;
        }
    }

    size_t shape[2] = {N, K};
    nb::capsule owner(result_data, [](void* p) noexcept { delete[] static_cast<float*>(p); });
    return nb::ndarray<nb::numpy, float, nb::shape<-1, -1>>(result_data, 2, shape, owner);
}

/**
 * Compute Normalized Wasserstein Distance (NWD) between two sets of boxes.
 */
nb::ndarray<nb::numpy, float, nb::shape<-1, -1>> normalized_wasserstein_distance(
    Array2D4 boxes,
    Array2D4 query_boxes,
    float tau = 1.0f
) {
    const size_t N = boxes.shape(0);
    const size_t K = query_boxes.shape(0);

    F32View<Array2D4> bv(boxes);
    F32View<Array2D4> qv(query_boxes);
    const float* boxes_ptr = bv.ptr;
    const float* query_ptr = qv.ptr;

    float* result_data = new float[N * K]();

    for (size_t k = 0; k < K; ++k) {
        float query_cx     = (query_ptr[k * 4 + 0] + query_ptr[k * 4 + 2]) / 2.0f;
        float query_cy     = (query_ptr[k * 4 + 1] + query_ptr[k * 4 + 3]) / 2.0f;
        float query_w_half = (query_ptr[k * 4 + 2] - query_ptr[k * 4 + 0]) / 2.0f;
        float query_h_half = (query_ptr[k * 4 + 3] - query_ptr[k * 4 + 1]) / 2.0f;

        for (size_t n = 0; n < N; ++n) {
            float box_cx     = (boxes_ptr[n * 4 + 0] + boxes_ptr[n * 4 + 2]) / 2.0f;
            float box_cy     = (boxes_ptr[n * 4 + 1] + boxes_ptr[n * 4 + 3]) / 2.0f;
            float box_w_half = (boxes_ptr[n * 4 + 2] - boxes_ptr[n * 4 + 0]) / 2.0f;
            float box_h_half = (boxes_ptr[n * 4 + 3] - boxes_ptr[n * 4 + 1]) / 2.0f;

            float mean_diff_sq = (box_cx - query_cx) * (box_cx - query_cx) +
                                 (box_cy - query_cy) * (box_cy - query_cy);
            float cov_diff_sq  = (box_w_half - query_w_half) * (box_w_half - query_w_half) +
                                 (box_h_half - query_h_half) * (box_h_half - query_h_half);

            result_data[n * K + k] = std::exp(-std::sqrt(mean_diff_sq + cov_diff_sq) / tau);
        }
    }

    size_t shape[2] = {N, K};
    nb::capsule owner(result_data, [](void* p) noexcept { delete[] static_cast<float*>(p); });
    return nb::ndarray<nb::numpy, float, nb::shape<-1, -1>>(result_data, 2, shape, owner);
}

NB_MODULE(bbox, m) {
    m.doc() = "Fast bounding box IoU calculations using nanobind (float32 and float16 input)";

    m.def("bbox_overlaps", &bbox_overlaps,
          nb::arg("boxes"), nb::arg("query_boxes"),
          "Compute IoU overlaps between two sets of boxes.\n\n"
          "Parameters\n----------\n"
          "boxes: (N, 4) float32 or float16 array [x1, y1, x2, y2]\n"
          "query_boxes: (K, 4) float32 or float16 array [x1, y1, x2, y2]\n\n"
          "Returns\n-------\noverlaps: (N, K) float32 array of IoU");

    m.def("generalized_iou", &generalized_iou,
          nb::arg("boxes"), nb::arg("query_boxes"),
          "Compute Generalized IoU (GIoU) between two sets of boxes. "
          "Accepts float32 or float16 input.");

    m.def("distance_iou", &distance_iou,
          nb::arg("boxes"), nb::arg("query_boxes"),
          "Compute Distance IoU (DIoU) between two sets of boxes. "
          "Accepts float32 or float16 input.");

    m.def("complete_iou", &complete_iou,
          nb::arg("boxes"), nb::arg("query_boxes"),
          "Compute Complete IoU (CIoU) between two sets of boxes. "
          "Accepts float32 or float16 input.");

    m.def("efficient_iou", &efficient_iou,
          nb::arg("boxes"), nb::arg("query_boxes"),
          "Compute Efficient IoU (EIoU) between two sets of boxes. "
          "Accepts float32 or float16 input.");

    m.def("normalized_wasserstein_distance", &normalized_wasserstein_distance,
          nb::arg("boxes"), nb::arg("query_boxes"), nb::arg("tau") = 1.0f,
          "Compute Normalized Wasserstein Distance (NWD) between two sets of boxes. "
          "Accepts float32 or float16 input.");
}
