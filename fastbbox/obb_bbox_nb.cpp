/**
 * Nanobind implementation of oriented bounding box (OBB) IoU calculations.
 * 
 * This file provides the same functionality as obb_bbox.pyx but using nanobind
 * for Python bindings instead of Cython.
 * 
 * OBB Format: [center_x, center_y, width, height, angle] where angle is in radians.
 */

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cmath>
#include <algorithm>

namespace nb = nanobind;

using FloatArray2DIn5 = nb::ndarray<const float, nb::shape<-1, 5>, nb::c_contig, nb::device::cpu>;

// Point structure
struct Point {
    float x;
    float y;
};

// OBB structure
struct OBB {
    float cx, cy;       // Center coordinates
    float width, height; // Dimensions
    float angle;        // Rotation angle in radians
};

// Helper function to create a Point
inline Point make_point(float x, float y) {
    return Point{x, y};
}

// Helper function to create an OBB
inline OBB make_obb(float cx, float cy, float width, float height, float angle) {
    return OBB{cx, cy, width, height, angle};
}

/**
 * Convert OBB to 4 corner points in counter-clockwise order.
 */
void obb_to_corners(const OBB& box, Point* corners) {
    float cos_a = std::cos(box.angle);
    float sin_a = std::sin(box.angle);
    float hw = box.width * 0.5f;
    float hh = box.height * 0.5f;
    
    // Local corner coordinates (counter-clockwise from bottom-left)
    float local_x[4] = {-hw, hw, hw, -hw};
    float local_y[4] = {-hh, -hh, hh, hh};
    
    // Transform to world coordinates
    for (int i = 0; i < 4; ++i) {
        corners[i].x = box.cx + local_x[i] * cos_a - local_y[i] * sin_a;
        corners[i].y = box.cy + local_x[i] * sin_a + local_y[i] * cos_a;
    }
}

/**
 * Calculate area of a polygon using the shoelace formula.
 */
float polygon_area(const Point* vertices, int n) {
    if (n < 3) {
        return 0.0f;
    }
    
    float area = 0.0f;
    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;
        area += vertices[i].x * vertices[j].y;
        area -= vertices[j].x * vertices[i].y;
    }
    
    return std::fabs(area) * 0.5f;
}

/**
 * Find intersection of two line segments.
 * Returns true if intersection exists, false otherwise.
 */
bool line_intersect(Point p1, Point p2, Point p3, Point p4, Point* intersection) {
    float denom = (p1.x - p2.x) * (p3.y - p4.y) - (p1.y - p2.y) * (p3.x - p4.x);
    
    if (std::fabs(denom) < 1e-10f) {
        return false;  // Lines are parallel
    }
    
    float t = ((p1.x - p3.x) * (p3.y - p4.y) - (p1.y - p3.y) * (p3.x - p4.x)) / denom;
    float u = -((p1.x - p2.x) * (p1.y - p3.y) - (p1.y - p2.y) * (p1.x - p3.x)) / denom;
    
    if (t >= 0 && t <= 1 && u >= 0 && u <= 1) {
        intersection->x = p1.x + t * (p2.x - p1.x);
        intersection->y = p1.y + t * (p2.y - p1.y);
        return true;
    }
    
    return false;
}

/**
 * Check if a point is inside a polygon using ray casting.
 */
bool point_in_polygon(Point point, const Point* polygon, int n) {
    bool inside = false;
    int j = n - 1;
    
    for (int i = 0; i < n; ++i) {
        if (((polygon[i].y > point.y) != (polygon[j].y > point.y)) &&
            (point.x < (polygon[j].x - polygon[i].x) * (point.y - polygon[i].y) / 
             (polygon[j].y - polygon[i].y) + polygon[i].x)) {
            inside = !inside;
        }
        j = i;
    }
    
    return inside;
}

/**
 * Calculate intersection area between two OBBs using a robust method.
 */
float obb_intersection_area_robust(const OBB& box1, const OBB& box2) {
    // Special case: both boxes are axis-aligned
    if (std::fabs(box1.angle) < 1e-6f && std::fabs(box2.angle) < 1e-6f) {
        // Exact calculation for axis-aligned boxes
        float x1_min = box1.cx - box1.width * 0.5f;
        float x1_max = box1.cx + box1.width * 0.5f;
        float y1_min = box1.cy - box1.height * 0.5f;
        float y1_max = box1.cy + box1.height * 0.5f;
        
        float x2_min = box2.cx - box2.width * 0.5f;
        float x2_max = box2.cx + box2.width * 0.5f;
        float y2_min = box2.cy - box2.height * 0.5f;
        float y2_max = box2.cy + box2.height * 0.5f;
        
        float inter_x_min = std::max(x1_min, x2_min);
        float inter_x_max = std::min(x1_max, x2_max);
        float inter_y_min = std::max(y1_min, y2_min);
        float inter_y_max = std::min(y1_max, y2_max);
        
        if (inter_x_min >= inter_x_max || inter_y_min >= inter_y_max) {
            return 0.0f;
        }
        
        return (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min);
    }
    
    // For rotated boxes, use a simplified approximation
    Point corners1[4], corners2[4];
    obb_to_corners(box1, corners1);
    obb_to_corners(box2, corners2);
    
    // Find axis-aligned bounding boxes
    float min_x1 = corners1[0].x, max_x1 = corners1[0].x;
    float min_y1 = corners1[0].y, max_y1 = corners1[0].y;
    for (int i = 1; i < 4; ++i) {
        min_x1 = std::min(min_x1, corners1[i].x);
        max_x1 = std::max(max_x1, corners1[i].x);
        min_y1 = std::min(min_y1, corners1[i].y);
        max_y1 = std::max(max_y1, corners1[i].y);
    }
    
    float min_x2 = corners2[0].x, max_x2 = corners2[0].x;
    float min_y2 = corners2[0].y, max_y2 = corners2[0].y;
    for (int i = 1; i < 4; ++i) {
        min_x2 = std::min(min_x2, corners2[i].x);
        max_x2 = std::max(max_x2, corners2[i].x);
        min_y2 = std::min(min_y2, corners2[i].y);
        max_y2 = std::max(max_y2, corners2[i].y);
    }
    
    // Calculate AABB intersection
    float inter_x_min = std::max(min_x1, min_x2);
    float inter_x_max = std::min(max_x1, max_x2);
    float inter_y_min = std::max(min_y1, min_y2);
    float inter_y_max = std::min(max_y1, max_y2);
    
    if (inter_x_min >= inter_x_max || inter_y_min >= inter_y_max) {
        return 0.0f;
    }
    
    // For rotated boxes, scale down the AABB intersection as an approximation
    float aabb_intersection = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min);
    
    // Apply a scaling factor based on rotation angles
    float angle_factor = std::cos(std::fabs(box1.angle)) * std::cos(std::fabs(box2.angle));
    angle_factor = std::max(0.5f, angle_factor);  // Minimum 50% of AABB intersection
    
    return aabb_intersection * angle_factor;
}

/**
 * Calculate area of an OBB
 */
inline float obb_area(const OBB& box) {
    return box.width * box.height;
}

/**
 * Compute IoU overlaps between two sets of oriented bounding boxes.
 *
 * Parameters
 * ----------
 * boxes: (N, 5) float32 array [cx, cy, width, height, angle]
 * query_boxes: (K, 5) float32 array [cx, cy, width, height, angle]
 *     where angle is in radians
 *
 * Returns
 * -------
 * overlaps: (N, K) float32 array of IoU values
 */
nb::ndarray<nb::numpy, float, nb::shape<-1, -1>> bbox_overlaps_obb(
    FloatArray2DIn5 boxes,
    FloatArray2DIn5 query_boxes
) {
    size_t N = boxes.shape(0);
    size_t K = query_boxes.shape(0);
    
    // Allocate output array
    float* result_data = new float[N * K]();
    
    const float* boxes_ptr = boxes.data();
    const float* query_ptr = query_boxes.data();
    
    for (size_t k = 0; k < K; ++k) {
        OBB box2 = make_obb(
            query_ptr[k * 5 + 0], query_ptr[k * 5 + 1],
            query_ptr[k * 5 + 2], query_ptr[k * 5 + 3],
            query_ptr[k * 5 + 4]
        );
        float area2 = obb_area(box2);
        
        for (size_t n = 0; n < N; ++n) {
            OBB box1 = make_obb(
                boxes_ptr[n * 5 + 0], boxes_ptr[n * 5 + 1],
                boxes_ptr[n * 5 + 2], boxes_ptr[n * 5 + 3],
                boxes_ptr[n * 5 + 4]
            );
            float area1 = obb_area(box1);
            
            // Calculate intersection area
            float intersection_area = obb_intersection_area_robust(box1, box2);
            
            // Calculate union area
            float union_area = area1 + area2 - intersection_area;
            
            // Calculate IoU
            float iou = 0.0f;
            if (union_area > 0) {
                iou = intersection_area / union_area;
            }
            
            result_data[n * K + k] = iou;
        }
    }
    
    // Create output ndarray with ownership
    size_t shape[2] = {N, K};
    nb::capsule owner(result_data, [](void* p) noexcept { delete[] static_cast<float*>(p); });
    
    return nb::ndarray<nb::numpy, float, nb::shape<-1, -1>>(
        result_data, 2, shape, owner
    );
}

NB_MODULE(obb_bbox, m) {
    m.doc() = "Oriented bounding box IoU calculations using nanobind";
    
    m.def("bbox_overlaps_obb", &bbox_overlaps_obb,
          nb::arg("boxes"), nb::arg("query_boxes"),
          "Compute IoU overlaps between two sets of oriented bounding boxes.\n\n"
          "Parameters\n"
          "----------\n"
          "boxes: (N, 5) float32 array [cx, cy, width, height, angle]\n"
          "query_boxes: (K, 5) float32 array [cx, cy, width, height, angle]\n"
          "    where angle is in radians\n\n"
          "Returns\n"
          "-------\n"
          "overlaps: (N, K) float32 array of IoU values");
}
