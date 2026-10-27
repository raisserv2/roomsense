"""Shared RoomSense code used by both the training pipeline and the serving API.

Keeping frame preprocessing and drift statistics in one place guarantees that
training and serving treat a camera frame in exactly the same way.
"""
