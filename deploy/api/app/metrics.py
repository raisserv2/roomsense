"""Prometheus metrics exposed on /metrics."""
from prometheus_client import Counter, Gauge, Histogram

LATENCY = Histogram("roomsense_inference_latency_seconds", "Model inference time per frame",
                    ["source"], buckets=(0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1, 2))
PREDICTIONS = Counter("roomsense_predictions_total", "Predictions by class", ["label", "source"])
CONFIDENCE = Histogram("roomsense_prediction_confidence", "Top-class probability",
                       buckets=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0))
BRIGHTNESS = Histogram("roomsense_input_brightness", "Mean grey level of input frames",
                       buckets=tuple(range(16, 257, 16)))
CURRENT = Gauge("roomsense_current_label", "1 for the class currently predicted from the camera", ["label"])

OOD = Gauge("roomsense_drift_ood_share", "Share of window frames in brightness ranges unseen in training")
PSI = Gauge("roomsense_drift_psi", "Population Stability Index of input brightness vs training data (informational)")
WINDOW = Gauge("roomsense_drift_window_size", "Frames in the drift window")
BRIGHT_MEAN = Gauge("roomsense_window_brightness_mean", "Mean brightness in the drift window")
REF_BRIGHT = Gauge("roomsense_reference_brightness_mean", "Mean brightness of the training frames")
CONF_MEAN = Gauge("roomsense_window_confidence_mean", "Mean top-class confidence in the drift window")
REF_CONF = Gauge("roomsense_reference_confidence_mean", "Mean top-class confidence on validation data")
LOW_CONF = Gauge("roomsense_window_low_confidence_ratio", "Share of window frames below the confidence threshold")
LIVE_ACC = Gauge("roomsense_live_accuracy", "Accuracy vs capture-session labels in the window (-1 = no labels)")
LABELLED = Counter("roomsense_labelled_frames_total", "Live frames with a ground-truth label", ["correct"])

MODEL_INFO = Gauge("roomsense_model_info", "Loaded model (value 1)", ["version", "run_id", "uri"])
MODEL_VERSION = Gauge("roomsense_model_version", "Registry version of the loaded model (0 = none)")
MODEL_LOADED = Gauge("roomsense_model_loaded", "1 when a model is loaded")
RELOADS = Counter("roomsense_model_reloads_total", "Model (re)loads", ["result"])

CAMERA_ERRORS = Counter("roomsense_camera_errors_total", "Failed camera fetches")
LAST_FRAME = Gauge("roomsense_last_frame_timestamp_seconds", "Unix time of the last camera frame")

RETRAIN_REQUESTS = Counter("roomsense_retrain_requests_total", "Retrain requests raised by drift alerts")
RETRAIN_PENDING = Gauge("roomsense_retrain_pending", "1 while a retrain request is waiting for the Legion")
RETRAINS_DONE = Counter("roomsense_retrains_completed_total", "Retrains acknowledged by the training machine")
