# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

import numpy as np
cimport numpy as np
from libc.math cimport atan2, sqrt, exp   

# Define type
ctypedef np.float32_t DTYPE_t

def bbox_overlaps(np.ndarray[DTYPE_t, ndim=2] boxes,
                  np.ndarray[DTYPE_t, ndim=2] query_boxes):
    """
    Compute IoU overlaps between two sets of boxes.

    Parameters
    ----------
    boxes: (N, 4) float32 array [x1, y1, x2, y2]
    query_boxes: (K, 4) float32 array [x1, y1, x2, y2]

    Returns
    -------
    overlaps: (N, K) float32 array of IoU
    """
    cdef unsigned int N = boxes.shape[0]
    cdef unsigned int K = query_boxes.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] overlaps = np.zeros((N, K), dtype=np.float32)

    cdef unsigned int n, k
    cdef DTYPE_t iw, ih, inter, ua
    cdef DTYPE_t box_area, query_area

    for k in range(K):
        query_area = (query_boxes[k, 2] - query_boxes[k, 0]) * \
                     (query_boxes[k, 3] - query_boxes[k, 1])

        for n in range(N):
            iw = min(boxes[n, 2], query_boxes[k, 2]) - \
                 max(boxes[n, 0], query_boxes[k, 0])
            if iw > 0:
                ih = min(boxes[n, 3], query_boxes[k, 3]) - \
                     max(boxes[n, 1], query_boxes[k, 1])
                if ih > 0:
                    inter = iw * ih
                    box_area = (boxes[n, 2] - boxes[n, 0]) * \
                               (boxes[n, 3] - boxes[n, 1])
                    ua = box_area + query_area - inter
                    overlaps[n, k] = inter / ua
    return overlaps


def generalized_iou(np.ndarray[DTYPE_t, ndim=2] boxes,
                    np.ndarray[DTYPE_t, ndim=2] query_boxes):
    """
    Compute Generalized IoU (GIoU) between two sets of boxes.
    
    GIoU = IoU - |C - (A ∪ B)| / |C|
    where C is the smallest enclosing box that covers both A and B.
    
    Paper: "Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression"
    https://arxiv.org/abs/1902.09630

    Parameters
    ----------
    boxes: (N, 4) float32 array [x1, y1, x2, y2]
    query_boxes: (K, 4) float32 array [x1, y1, x2, y2]

    Returns
    -------
    giou: (N, K) float32 array of Generalized IoU values
    """
    cdef unsigned int N = boxes.shape[0]
    cdef unsigned int K = query_boxes.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] giou_matrix = np.zeros((N, K), dtype=np.float32)

    cdef unsigned int n, k
    cdef DTYPE_t iw, ih, inter, ua, iou
    cdef DTYPE_t box_area, query_area
    cdef DTYPE_t enclosing_x1, enclosing_y1, enclosing_x2, enclosing_y2
    cdef DTYPE_t enclosing_area, coverage_ratio, giou_val

    for k in range(K):
        query_area = (query_boxes[k, 2] - query_boxes[k, 0]) * \
                     (query_boxes[k, 3] - query_boxes[k, 1])

        for n in range(N):
            # Calculate intersection
            iw = min(boxes[n, 2], query_boxes[k, 2]) - \
                 max(boxes[n, 0], query_boxes[k, 0])
            if iw > 0:
                ih = min(boxes[n, 3], query_boxes[k, 3]) - \
                     max(boxes[n, 1], query_boxes[k, 1])
                if ih > 0:
                    inter = iw * ih
                else:
                    inter = 0
            else:
                inter = 0

            # Calculate union and IoU
            box_area = (boxes[n, 2] - boxes[n, 0]) * \
                       (boxes[n, 3] - boxes[n, 1])
            ua = box_area + query_area - inter
            
            if ua > 0:
                iou = inter / ua
            else:
                iou = 0

            # Calculate smallest enclosing box
            enclosing_x1 = min(boxes[n, 0], query_boxes[k, 0])
            enclosing_y1 = min(boxes[n, 1], query_boxes[k, 1])
            enclosing_x2 = max(boxes[n, 2], query_boxes[k, 2])
            enclosing_y2 = max(boxes[n, 3], query_boxes[k, 3])
            
            enclosing_area = (enclosing_x2 - enclosing_x1) * \
                            (enclosing_y2 - enclosing_y1)

            # Calculate GIoU
            if enclosing_area > 0:
                coverage_ratio = (enclosing_area - ua) / enclosing_area
                giou_val = iou - coverage_ratio
            else:
                giou_val = iou

            giou_matrix[n, k] = giou_val

    return giou_matrix


def distance_iou(np.ndarray[DTYPE_t, ndim=2] boxes,
                 np.ndarray[DTYPE_t, ndim=2] query_boxes):
    """
    Compute Distance IoU (DIoU) between two sets of boxes.
    
    DIoU = IoU - ρ²(b, b_gt) / c²
    where:
    - ρ²(b, b_gt) is the squared Euclidean distance between box centers
    - c² is the squared diagonal length of the smallest enclosing box
    
    Paper: "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression"
    https://arxiv.org/abs/1911.08287

    Parameters
    ----------
    boxes: (N, 4) float32 array [x1, y1, x2, y2]
    query_boxes: (K, 4) float32 array [x1, y1, x2, y2]

    Returns
    -------
    diou: (N, K) float32 array of Distance IoU values
    """
    cdef unsigned int N = boxes.shape[0]
    cdef unsigned int K = query_boxes.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] diou_matrix = np.zeros((N, K), dtype=np.float32)

    cdef unsigned int n, k
    cdef DTYPE_t iw, ih, inter, ua, iou
    cdef DTYPE_t box_area, query_area
    cdef DTYPE_t box_cx, box_cy, query_cx, query_cy
    cdef DTYPE_t center_distance_sq, diagonal_sq
    cdef DTYPE_t enclosing_x1, enclosing_y1, enclosing_x2, enclosing_y2
    cdef DTYPE_t diou_val

    for k in range(K):
        query_area = (query_boxes[k, 2] - query_boxes[k, 0]) * \
                     (query_boxes[k, 3] - query_boxes[k, 1])
        query_cx = (query_boxes[k, 0] + query_boxes[k, 2]) / 2.0
        query_cy = (query_boxes[k, 1] + query_boxes[k, 3]) / 2.0

        for n in range(N):
            # Calculate intersection
            iw = min(boxes[n, 2], query_boxes[k, 2]) - \
                 max(boxes[n, 0], query_boxes[k, 0])
            if iw > 0:
                ih = min(boxes[n, 3], query_boxes[k, 3]) - \
                     max(boxes[n, 1], query_boxes[k, 1])
                if ih > 0:
                    inter = iw * ih
                else:
                    inter = 0
            else:
                inter = 0

            # Calculate union and IoU
            box_area = (boxes[n, 2] - boxes[n, 0]) * \
                       (boxes[n, 3] - boxes[n, 1])
            ua = box_area + query_area - inter
            
            if ua > 0:
                iou = inter / ua
            else:
                iou = 0

            # Calculate box centers
            box_cx = (boxes[n, 0] + boxes[n, 2]) / 2.0
            box_cy = (boxes[n, 1] + boxes[n, 3]) / 2.0

            # Calculate center distance squared
            center_distance_sq = (box_cx - query_cx) * (box_cx - query_cx) + \
                                (box_cy - query_cy) * (box_cy - query_cy)

            # Calculate smallest enclosing box diagonal squared
            enclosing_x1 = min(boxes[n, 0], query_boxes[k, 0])
            enclosing_y1 = min(boxes[n, 1], query_boxes[k, 1])
            enclosing_x2 = max(boxes[n, 2], query_boxes[k, 2])
            enclosing_y2 = max(boxes[n, 3], query_boxes[k, 3])
            
            diagonal_sq = (enclosing_x2 - enclosing_x1) * (enclosing_x2 - enclosing_x1) + \
                         (enclosing_y2 - enclosing_y1) * (enclosing_y2 - enclosing_y1)

            # Calculate DIoU
            if diagonal_sq > 0:
                diou_val = iou - center_distance_sq / diagonal_sq
            else:
                diou_val = iou

            diou_matrix[n, k] = diou_val

    return diou_matrix


def complete_iou(np.ndarray[DTYPE_t, ndim=2] boxes,
                 np.ndarray[DTYPE_t, ndim=2] query_boxes):
    """
    Compute Complete IoU (CIoU) between two sets of boxes.
    
    CIoU = IoU - ρ²(b, b_gt) / c² - α * v
    where:
    - ρ²(b, b_gt) is the squared Euclidean distance between box centers  
    - c² is the squared diagonal length of the smallest enclosing box
    - v measures the consistency of aspect ratio
    - α is a positive trade-off parameter
    
    Paper: "Enhancing Geometric Factors in Model Learning and Inference for Object Detection and Instance Segmentation"
    https://arxiv.org/abs/1911.08287

    Parameters
    ----------
    boxes: (N, 4) float32 array [x1, y1, x2, y2]
    query_boxes: (K, 4) float32 array [x1, y1, x2, y2]

    Returns
    -------
    ciou: (N, K) float32 array of Complete IoU values
    """
    cdef unsigned int N = boxes.shape[0]
    cdef unsigned int K = query_boxes.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] ciou_matrix = np.zeros((N, K), dtype=np.float32)

    cdef unsigned int n, k
    cdef DTYPE_t iw, ih, inter, ua, iou
    cdef DTYPE_t box_area, query_area
    cdef DTYPE_t box_cx, box_cy, query_cx, query_cy
    cdef DTYPE_t box_w, box_h, query_w, query_h
    cdef DTYPE_t center_distance_sq, diagonal_sq
    cdef DTYPE_t enclosing_x1, enclosing_y1, enclosing_x2, enclosing_y2
    cdef DTYPE_t v, alpha, ciou_val
    cdef DTYPE_t pi = 3.14159265359
    cdef DTYPE_t atan_diff, aspect_ratio_penalty

    for k in range(K):
        query_area = (query_boxes[k, 2] - query_boxes[k, 0]) * \
                     (query_boxes[k, 3] - query_boxes[k, 1])
        query_cx = (query_boxes[k, 0] + query_boxes[k, 2]) / 2.0
        query_cy = (query_boxes[k, 1] + query_boxes[k, 3]) / 2.0
        query_w = query_boxes[k, 2] - query_boxes[k, 0]
        query_h = query_boxes[k, 3] - query_boxes[k, 1]

        for n in range(N):
            # Calculate intersection
            iw = min(boxes[n, 2], query_boxes[k, 2]) - \
                 max(boxes[n, 0], query_boxes[k, 0])
            if iw > 0:
                ih = min(boxes[n, 3], query_boxes[k, 3]) - \
                     max(boxes[n, 1], query_boxes[k, 1])
                if ih > 0:
                    inter = iw * ih
                else:
                    inter = 0
            else:
                inter = 0

            # Calculate union and IoU
            box_area = (boxes[n, 2] - boxes[n, 0]) * \
                       (boxes[n, 3] - boxes[n, 1])
            ua = box_area + query_area - inter
            
            if ua > 0:
                iou = inter / ua
            else:
                iou = 0

            # Calculate box centers and dimensions
            box_cx = (boxes[n, 0] + boxes[n, 2]) / 2.0
            box_cy = (boxes[n, 1] + boxes[n, 3]) / 2.0
            box_w = boxes[n, 2] - boxes[n, 0]
            box_h = boxes[n, 3] - boxes[n, 1]

            # Calculate center distance squared
            center_distance_sq = (box_cx - query_cx) * (box_cx - query_cx) + \
                                (box_cy - query_cy) * (box_cy - query_cy)

            # Calculate smallest enclosing box diagonal squared
            enclosing_x1 = min(boxes[n, 0], query_boxes[k, 0])
            enclosing_y1 = min(boxes[n, 1], query_boxes[k, 1])
            enclosing_x2 = max(boxes[n, 2], query_boxes[k, 2])
            enclosing_y2 = max(boxes[n, 3], query_boxes[k, 3])
            
            diagonal_sq = (enclosing_x2 - enclosing_x1) * (enclosing_x2 - enclosing_x1) + \
                         (enclosing_y2 - enclosing_y1) * (enclosing_y2 - enclosing_y1)

            # Calculate aspect ratio consistency v
            if query_w > 0 and query_h > 0 and box_w > 0 and box_h > 0:
                # Use atan2 for more stable computation
                atan_diff = atan2(query_w, query_h) - atan2(box_w, box_h)
                v = (4.0 / (pi * pi)) * atan_diff * atan_diff
            else:
                v = 0

            # Calculate alpha parameter
            if iou > 0:
                alpha = v / (1 - iou + v + 1e-8)  # Add small epsilon for numerical stability
            else:
                alpha = 0

            # Calculate CIoU
            if diagonal_sq > 0:
                ciou_val = iou - center_distance_sq / diagonal_sq - alpha * v
            else:
                ciou_val = iou

            ciou_matrix[n, k] = ciou_val

    return ciou_matrix


def efficient_iou(np.ndarray[DTYPE_t, ndim=2] boxes,
                  np.ndarray[DTYPE_t, ndim=2] query_boxes):
    """
    Compute Efficient IoU (EIoU) between two sets of boxes.
    
    EIoU = IoU - ρ²(b, b_gt) / c² - ρ²(w, w_gt) / c_w² - ρ²(h, h_gt) / c_h²
    where:
    - ρ²(b, b_gt) is the squared Euclidean distance between box centers
    - c² is the squared diagonal length of the smallest enclosing box
    - ρ²(w, w_gt) is the squared difference between widths
    - ρ²(h, h_gt) is the squared difference between heights
    - c_w², c_h² are the squared widths and heights of the enclosing box
    
    Paper: "Focal and Efficient IOU Loss for Accurate Bounding Box Regression"
    https://arxiv.org/abs/2101.08158

    Parameters
    ----------
    boxes: (N, 4) float32 array [x1, y1, x2, y2]
    query_boxes: (K, 4) float32 array [x1, y1, x2, y2]

    Returns
    -------
    eiou: (N, K) float32 array of Efficient IoU values
    """
    cdef unsigned int N = boxes.shape[0]
    cdef unsigned int K = query_boxes.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] eiou_matrix = np.zeros((N, K), dtype=np.float32)

    cdef unsigned int n, k
    cdef DTYPE_t iw, ih, inter, ua, iou
    cdef DTYPE_t box_area, query_area
    cdef DTYPE_t box_cx, box_cy, query_cx, query_cy
    cdef DTYPE_t box_w, box_h, query_w, query_h
    cdef DTYPE_t center_distance_sq, diagonal_sq
    cdef DTYPE_t width_diff_sq, height_diff_sq
    cdef DTYPE_t enclosing_x1, enclosing_y1, enclosing_x2, enclosing_y2
    cdef DTYPE_t enclosing_w, enclosing_h, enclosing_w_sq, enclosing_h_sq
    cdef DTYPE_t eiou_val

    for k in range(K):
        query_area = (query_boxes[k, 2] - query_boxes[k, 0]) * \
                     (query_boxes[k, 3] - query_boxes[k, 1])
        query_cx = (query_boxes[k, 0] + query_boxes[k, 2]) / 2.0
        query_cy = (query_boxes[k, 1] + query_boxes[k, 3]) / 2.0
        query_w = query_boxes[k, 2] - query_boxes[k, 0]
        query_h = query_boxes[k, 3] - query_boxes[k, 1]

        for n in range(N):
            # Calculate intersection
            iw = min(boxes[n, 2], query_boxes[k, 2]) - \
                 max(boxes[n, 0], query_boxes[k, 0])
            if iw > 0:
                ih = min(boxes[n, 3], query_boxes[k, 3]) - \
                     max(boxes[n, 1], query_boxes[k, 1])
                if ih > 0:
                    inter = iw * ih
                else:
                    inter = 0
            else:
                inter = 0

            # Calculate union and IoU
            box_area = (boxes[n, 2] - boxes[n, 0]) * \
                       (boxes[n, 3] - boxes[n, 1])
            ua = box_area + query_area - inter
            
            if ua > 0:
                iou = inter / ua
            else:
                iou = 0

            # Calculate box centers and dimensions
            box_cx = (boxes[n, 0] + boxes[n, 2]) / 2.0
            box_cy = (boxes[n, 1] + boxes[n, 3]) / 2.0
            box_w = boxes[n, 2] - boxes[n, 0]
            box_h = boxes[n, 3] - boxes[n, 1]

            # Calculate center distance squared
            center_distance_sq = (box_cx - query_cx) * (box_cx - query_cx) + \
                                (box_cy - query_cy) * (box_cy - query_cy)

            # Calculate smallest enclosing box
            enclosing_x1 = min(boxes[n, 0], query_boxes[k, 0])
            enclosing_y1 = min(boxes[n, 1], query_boxes[k, 1])
            enclosing_x2 = max(boxes[n, 2], query_boxes[k, 2])
            enclosing_y2 = max(boxes[n, 3], query_boxes[k, 3])
            
            diagonal_sq = (enclosing_x2 - enclosing_x1) * (enclosing_x2 - enclosing_x1) + \
                         (enclosing_y2 - enclosing_y1) * (enclosing_y2 - enclosing_y1)
            
            enclosing_w = enclosing_x2 - enclosing_x1
            enclosing_h = enclosing_y2 - enclosing_y1
            enclosing_w_sq = enclosing_w * enclosing_w
            enclosing_h_sq = enclosing_h * enclosing_h

            # Calculate width and height differences squared
            width_diff_sq = (box_w - query_w) * (box_w - query_w)
            height_diff_sq = (box_h - query_h) * (box_h - query_h)

            # Calculate EIoU
            eiou_val = iou
            if diagonal_sq > 0:
                eiou_val -= center_distance_sq / diagonal_sq
            if enclosing_w_sq > 0:
                eiou_val -= width_diff_sq / enclosing_w_sq
            if enclosing_h_sq > 0:
                eiou_val -= height_diff_sq / enclosing_h_sq

            eiou_matrix[n, k] = eiou_val

    return eiou_matrix


def normalized_wasserstein_distance(np.ndarray[DTYPE_t, ndim=2] boxes,
                                   np.ndarray[DTYPE_t, ndim=2] query_boxes,
                                   DTYPE_t tau=1.0):
    """
    Compute Normalized Wasserstein Distance (NWD) between two sets of boxes.
    
    CORRECTED implementation based on:
    "A Normalized Gaussian Wasserstein Distance for Tiny Object Detection"
    https://arxiv.org/abs/2110.13389
    
    For a box [x1, y1, x2, y2], the corresponding 2D Gaussian has:
    - Mean: μ = (center_x, center_y)
    - Covariance: Σ = diag(w²/4, h²/4) where w,h are width and height
    
    W₂²(Na, Nb) = ||μa - μb||₂² + ||Σa^(1/2) - Σb^(1/2)||F²
                = (cx_a - cx_b)² + (cy_a - cy_b)² + (wa/2 - wb/2)² + (ha/2 - hb/2)²
    
    NWD(Na, Nb) = exp(-√(W₂²(Na, Nb)) / τ)

    Parameters
    ----------
    boxes: (N, 4) float32 array [x1, y1, x2, y2]
    query_boxes: (K, 4) float32 array [x1, y1, x2, y2]
    tau: float, normalization constant (default: 1.0)

    Returns
    -------
    nwd: (N, K) float32 array of NWD values in [0, 1] where 1 = identical, 0 = very different
    """
    cdef unsigned int N = boxes.shape[0]
    cdef unsigned int K = query_boxes.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] nwd_matrix = np.zeros((N, K), dtype=np.float32)

    cdef unsigned int n, k
    cdef DTYPE_t box_cx, box_cy, query_cx, query_cy
    cdef DTYPE_t box_w_half, box_h_half, query_w_half, query_h_half
    cdef DTYPE_t mean_diff_sq, cov_diff_sq, wasserstein_sq
    cdef DTYPE_t nwd_val
    cdef DTYPE_t tau_val = tau

    for k in range(K):
        query_cx = (query_boxes[k, 0] + query_boxes[k, 2]) / 2.0
        query_cy = (query_boxes[k, 1] + query_boxes[k, 3]) / 2.0
        query_w_half = (query_boxes[k, 2] - query_boxes[k, 0]) / 2.0
        query_h_half = (query_boxes[k, 3] - query_boxes[k, 1]) / 2.0

        for n in range(N):
            # Calculate box centers and half-dimensions
            box_cx = (boxes[n, 0] + boxes[n, 2]) / 2.0
            box_cy = (boxes[n, 1] + boxes[n, 3]) / 2.0
            box_w_half = (boxes[n, 2] - boxes[n, 0]) / 2.0
            box_h_half = (boxes[n, 3] - boxes[n, 1]) / 2.0

            # Mean difference squared: ||μ₁ - μ₂||²
            mean_diff_sq = (box_cx - query_cx) * (box_cx - query_cx) + \
                          (box_cy - query_cy) * (box_cy - query_cy)

            # Covariance difference squared: ||Σ₁^(1/2) - Σ₂^(1/2)||F²
            # For diagonal Σ = diag(w²/4, h²/4): ||Σ₁^(1/2) - Σ₂^(1/2)||F² = (w₁/2 - w₂/2)² + (h₁/2 - h₂/2)²
            cov_diff_sq = (box_w_half - query_w_half) * (box_w_half - query_w_half) + \
                         (box_h_half - query_h_half) * (box_h_half - query_h_half)

            # Wasserstein-2 distance squared
            wasserstein_sq = mean_diff_sq + cov_diff_sq

            # Apply exponential normalization: NWD = exp(-√(W₂²) / τ)
            if wasserstein_sq >= 0:
                nwd_val = exp(-sqrt(wasserstein_sq) / tau_val)
            else:
                nwd_val = 1.0  # Safety check

            nwd_matrix[n, k] = nwd_val

    return nwd_matrix


