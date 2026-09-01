"""
core/detectors/
────────────────
Detection backend implementations.

  RTDETRDetector       — wraps ultralytics.RTDETR (real-time transformer)
  UltralyticsDetector  — wraps ultralytics.YOLO   (AGPL-3.0)
  ONNXDetector         — pure onnxruntime          (MIT, AGPL-free)

Backend imports are deferred so that the ONNX path never pulls in
torch/ultralytics. Import directly from the submodules:

    from core.detectors.onnx_detector import ONNXDetector
    from core.detectors.rtdetr_detector import RTDETRDetector
    from core.detectors.ultralytics_detector import UltralyticsDetector
"""
