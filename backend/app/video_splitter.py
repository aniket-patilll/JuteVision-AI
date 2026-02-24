"""
Video Splitter Utility
======================
Splits a single multi-camera CCTV video (e.g. 2×2 grid) into separate
video files — one per camera quadrant. Uses OpenCV frame cropping.
"""

import cv2
import os


def split_video(video_path: str, rows: int, cols: int, output_dir: str) -> list:
    """
    Split a multi-camera grid video into individual camera videos.

    Args:
        video_path: Path to the input multi-camera video
        rows: Number of rows in the grid (e.g. 2 for a 2×2 layout)
        cols: Number of columns in the grid (e.g. 2 for a 2×2 layout)
        output_dir: Directory to write the split video files

    Returns:
        List of dicts: [{"label": "Camera 1", "video_path": "/path/to/split_cam1.mp4"}, ...]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25

    cell_width = total_width // cols
    cell_height = total_height // rows

    os.makedirs(output_dir, exist_ok=True)

    # Prepare writers for each cell
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    writers = []
    results = []

    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c + 1
            label = f"Camera {idx}"
            out_path = os.path.join(output_dir, f"{base_name}_cam{idx}.mp4")

            # Try H.264 first, fallback to mp4v
            try:
                fourcc = cv2.VideoWriter_fourcc(*"avc1")
                writer = cv2.VideoWriter(out_path, fourcc, fps, (cell_width, cell_height))
                if not writer.isOpened():
                    raise Exception("avc1 failed")
            except Exception:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_path, fourcc, fps, (cell_width, cell_height))

            writers.append({
                "writer": writer,
                "row": r,
                "col": c,
                "label": label,
                "path": out_path,
            })
            results.append({"label": label, "video_path": out_path})

    # Process frames
    while True:
        success, frame = cap.read()
        if not success:
            break

        for w in writers:
            r, c = w["row"], w["col"]
            y1 = r * cell_height
            y2 = y1 + cell_height
            x1 = c * cell_width
            x2 = x1 + cell_width
            cropped = frame[y1:y2, x1:x2]
            w["writer"].write(cropped)

    # Release everything
    cap.release()
    for w in writers:
        w["writer"].release()

    print(f"Split video into {len(results)} parts: {[r['label'] for r in results]}")
    return results
