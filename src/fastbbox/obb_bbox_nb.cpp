/**
 * Nanobind implementation of oriented bounding box (OBB) IoU calculations.
 *
 * Accepts float32 and float16 input arrays.  All computation is performed
 * in float32; output is always float32.
 *
 * OBB format: [center_x, center_y, width, height, angle_radians]
 */

#include "fastbbox_fp16.h"
#include <cmath>
#include <algorithm>

// ---------------------------------------------------------------------------
// Internal geometry types and helpers (all float32)
// ---------------------------------------------------------------------------

struct Point { float x, y; };
struct OBB   { float cx, cy, width, height, angle; };

inline OBB make_obb(const float* p) {
    return OBB{p[0], p[1], p[2], p[3], p[4]};
}

void obb_to_corners(const OBB& box, Point* corners) {
    float cos_a = std::cos(box.angle);
    float sin_a = std::sin(box.angle);
    float hw = box.width  * 0.5f;
    float hh = box.height * 0.5f;

    float lx[4] = {-hw,  hw,  hw, -hw};
    float ly[4] = {-hh, -hh,  hh,  hh};

    for (int i = 0; i < 4; ++i) {
        corners[i].x = box.cx + lx[i] * cos_a - ly[i] * sin_a;
        corners[i].y = box.cy + lx[i] * sin_a + ly[i] * cos_a;
    }
}

float polygon_area(const Point* v, int n) {
    if (n < 3) return 0.0f;
    float area = 0.0f;
    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;
        area += v[i].x * v[j].y - v[j].x * v[i].y;
    }
    return std::fabs(area) * 0.5f;
}

bool line_intersect(Point p1, Point p2, Point p3, Point p4, Point* out) {
    float denom = (p1.x - p2.x) * (p3.y - p4.y) - (p1.y - p2.y) * (p3.x - p4.x);
    if (std::fabs(denom) < 1e-10f) return false;

    float t = ((p1.x - p3.x) * (p3.y - p4.y) - (p1.y - p3.y) * (p3.x - p4.x)) / denom;
    float u = -((p1.x - p2.x) * (p1.y - p3.y) - (p1.y - p2.y) * (p1.x - p3.x)) / denom;

    if (t >= 0 && t <= 1 && u >= 0 && u <= 1) {
        out->x = p1.x + t * (p2.x - p1.x);
        out->y = p1.y + t * (p2.y - p1.y);
        return true;
    }
    return false;
}

bool point_in_polygon(Point pt, const Point* poly, int n) {
    bool inside = false;
    int j = n - 1;
    for (int i = 0; i < n; ++i) {
        if (((poly[i].y > pt.y) != (poly[j].y > pt.y)) &&
            (pt.x < (poly[j].x - poly[i].x) * (pt.y - poly[i].y) /
                    (poly[j].y - poly[i].y) + poly[i].x))
            inside = !inside;
        j = i;
    }
    return inside;
}

float obb_intersection_area(const OBB& box1, const OBB& box2) {
    // Exact path for axis-aligned boxes
    if (std::fabs(box1.angle) < 1e-6f && std::fabs(box2.angle) < 1e-6f) {
        float x1_min = box1.cx - box1.width  * 0.5f, x1_max = box1.cx + box1.width  * 0.5f;
        float y1_min = box1.cy - box1.height * 0.5f, y1_max = box1.cy + box1.height * 0.5f;
        float x2_min = box2.cx - box2.width  * 0.5f, x2_max = box2.cx + box2.width  * 0.5f;
        float y2_min = box2.cy - box2.height * 0.5f, y2_max = box2.cy + box2.height * 0.5f;

        float ix_min = std::max(x1_min, x2_min), ix_max = std::min(x1_max, x2_max);
        float iy_min = std::max(y1_min, y2_min), iy_max = std::min(y1_max, y2_max);
        if (ix_min >= ix_max || iy_min >= iy_max) return 0.0f;
        return (ix_max - ix_min) * (iy_max - iy_min);
    }

    // Rotated approximation: AABB intersection × angle scaling factor
    Point c1[4], c2[4];
    obb_to_corners(box1, c1);
    obb_to_corners(box2, c2);

    float mnx1 = c1[0].x, mxx1 = c1[0].x, mny1 = c1[0].y, mxy1 = c1[0].y;
    float mnx2 = c2[0].x, mxx2 = c2[0].x, mny2 = c2[0].y, mxy2 = c2[0].y;
    for (int i = 1; i < 4; ++i) {
        mnx1 = std::min(mnx1, c1[i].x); mxx1 = std::max(mxx1, c1[i].x);
        mny1 = std::min(mny1, c1[i].y); mxy1 = std::max(mxy1, c1[i].y);
        mnx2 = std::min(mnx2, c2[i].x); mxx2 = std::max(mxx2, c2[i].x);
        mny2 = std::min(mny2, c2[i].y); mxy2 = std::max(mxy2, c2[i].y);
    }

    float ix_min = std::max(mnx1, mnx2), ix_max = std::min(mxx1, mxx2);
    float iy_min = std::max(mny1, mny2), iy_max = std::min(mxy1, mxy2);
    if (ix_min >= ix_max || iy_min >= iy_max) return 0.0f;

    float aabb_inter  = (ix_max - ix_min) * (iy_max - iy_min);
    float angle_scale = std::cos(std::fabs(box1.angle)) * std::cos(std::fabs(box2.angle));
    angle_scale = std::max(0.5f, angle_scale);
    return aabb_inter * angle_scale;
}

inline float obb_area(const OBB& box) { return box.width * box.height; }

// ---------------------------------------------------------------------------
// Python-visible function
// ---------------------------------------------------------------------------

/**
 * Compute IoU overlaps between two sets of oriented bounding boxes.
 *
 * Parameters
 * ----------
 * boxes: (N, 5) float32 or float16 array [cx, cy, width, height, angle_rad]
 * query_boxes: (K, 5) float32 or float16 array [cx, cy, width, height, angle_rad]
 *
 * Returns
 * -------
 * overlaps: (N, K) float32 array of IoU values
 */
nb::ndarray<nb::numpy, float, nb::shape<-1, -1>> bbox_overlaps_obb(
    Array2D5 boxes,
    Array2D5 query_boxes
) {
    const size_t N = boxes.shape(0);
    const size_t K = query_boxes.shape(0);

    F32View<Array2D5> bv(boxes);
    F32View<Array2D5> qv(query_boxes);
    const float* boxes_ptr = bv.ptr;
    const float* query_ptr = qv.ptr;

    float* result_data = new float[N * K]();

    for (size_t k = 0; k < K; ++k) {
        OBB   box2  = make_obb(query_ptr + k * 5);
        float area2 = obb_area(box2);

        for (size_t n = 0; n < N; ++n) {
            OBB   box1         = make_obb(boxes_ptr + n * 5);
            float area1        = obb_area(box1);
            float inter        = obb_intersection_area(box1, box2);
            float union_area   = area1 + area2 - inter;
            result_data[n * K + k] = (union_area > 0) ? inter / union_area : 0.0f;
        }
    }

    size_t shape[2] = {N, K};
    nb::capsule owner(result_data, [](void* p) noexcept { delete[] static_cast<float*>(p); });
    return nb::ndarray<nb::numpy, float, nb::shape<-1, -1>>(result_data, 2, shape, owner);
}

NB_MODULE(obb_bbox, m) {
    m.doc() = "Oriented bounding box IoU calculations using nanobind (float32 and float16 input)";

    m.def("bbox_overlaps_obb", &bbox_overlaps_obb,
          nb::arg("boxes"), nb::arg("query_boxes"),
          "Compute IoU overlaps between two sets of oriented bounding boxes.\n\n"
          "Parameters\n----------\n"
          "boxes: (N, 5) float32 or float16 [cx, cy, width, height, angle_rad]\n"
          "query_boxes: (K, 5) float32 or float16 [cx, cy, width, height, angle_rad]\n\n"
          "Returns\n-------\noverlaps: (N, K) float32 array of IoU values");
}
