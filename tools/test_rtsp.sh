#!/usr/bin/env bash
# Smoke-test a single RTSP URL through the exact GStreamer pipeline
# camera_reader.py uses. Verifies hw decode + nvvidconv resize end-to-end
# before plugging the URL into config.yaml.
#
# Usage:
#   bash tools/test_rtsp.sh                                    # default URL + 640
#   bash tools/test_rtsp.sh 'rtsp://10.0.11.153:8554/cctv09'   # custom URL
#   bash tools/test_rtsp.sh 'rtsp://.../cctv08' 416            # custom size
#
# Success: pipeline reaches PLAYING, "New clock: GstSystemClock", then sits
# there reading frames — Ctrl+C to exit. Failure: an ERROR line names the
# element that broke (rtph264depay if the stream isn't H.264, nvv4l2decoder
# if the profile/level is unsupported, etc.).

URL="${1:-rtsp://10.0.11.153:8554/cctv08}"
SIZE="${2:-640}"

echo "Testing: $URL  (resize to ${SIZE}x${SIZE})"
echo "Ctrl+C to stop once you see 'New clock'."
echo "---"

GST_DEBUG=3 gst-launch-1.0 -v \
  rtspsrc location="$URL" protocols=tcp latency=0 \
  ! rtph264depay \
  ! h264parse \
  ! nvv4l2decoder \
  ! nvvidconv \
  ! "video/x-raw,format=BGRx,width=${SIZE},height=${SIZE}" \
  ! videoconvert \
  ! "video/x-raw,format=BGR" \
  ! fakesink
