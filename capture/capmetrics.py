"""Prometheus metrics of the capture service (scraped at /metrics)."""
from prometheus_client import Counter, Gauge, Histogram

FRAMES = Counter("roomsense_capture_frames_total", "Frames saved by the capture loop", ["label", "lighting"])
CAMERA_ERRORS = Counter("roomsense_capture_camera_errors_total", "Failed camera fetches while recording")
SAVE_ERRORS = Counter("roomsense_capture_save_errors_total", "Frames fetched but not saved (disk or CSV error)")
SESSION_ACTIVE = Gauge("roomsense_capture_session_active", "1 while a labelled session is recording")
SESSIONS = Counter("roomsense_capture_sessions_started_total", "Recording sessions started", ["label"])
LAST_FRAME = Gauge("roomsense_capture_last_frame_timestamp_seconds", "Unix time of the last saved frame")
FETCH_SECONDS = Histogram("roomsense_capture_fetch_seconds", "Time to fetch one frame from the camera",
                          buckets=(0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 2, 5))
CURATION = Counter("roomsense_capture_curation_actions_total", "Review-tool actions", ["action"])
DATASET_FRAMES = Gauge("roomsense_capture_dataset_frames", "Frames in the raw dataset", ["label"])
DATASET_SESSIONS = Gauge("roomsense_capture_dataset_sessions", "Sessions in the raw dataset", ["label"])
DISK_FREE = Gauge("roomsense_capture_disk_free_bytes", "Free space on the dataset drive")
