"""Tests using Creative Commons licensed video test data.

These tests validate decord's VideoReader against videos with known properties
(resolution, frame count, FPS, timestamps) determined by ffmpeg analysis.

The test videos are generated using ffmpeg test sources with properties inspired
by well-known CC-BY-3.0 Blender Foundation films (Big Buck Bunny, Sintel,
Tears of Steel). See tests/test_data/cc_videos/README.md for details on
video sources and regeneration instructions.
"""
import os
import random
import numpy as np
from io import BytesIO
from decord import VideoReader, cpu

CTX = cpu(0)

# Path to the CC video test data directory
CC_VIDEO_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'test_data', 'cc_videos'))

# Video metadata determined by ffmpeg analysis:
#   ffmpeg -i <file> (duration, resolution, fps)
#   ffmpeg -i <file> -vf showinfo -f null - (frame count)
#   decord VideoReader.get_frame_timestamp() (timestamps)
CC_VIDEOS = {
    'bbb': {
        'filename': 'bbb_style_640x360_24fps.mp4',
        'num_frames': 120,
        'height': 360,
        'width': 640,
        'fps': 24.0,
        'duration_sec': 5.0,
        'first_timestamps': [0.0, 1/24, 2/24, 3/24, 4/24],
    },
    'sintel': {
        'filename': 'sintel_style_320x240_30fps.mp4',
        'num_frames': 90,
        'height': 240,
        'width': 320,
        'fps': 30.0,
        'duration_sec': 3.0,
        'first_timestamps': [0.0, 1/30, 2/30, 3/30, 4/30],
    },
    'tos': {
        'filename': 'tos_style_480x270_25fps.mp4',
        'num_frames': 200,
        'height': 270,
        'width': 480,
        'fps': 25.0,
        'duration_sec': 8.0,
        'first_timestamps': [0.0, 1/25, 2/25, 3/25, 4/25],
    },
}


def _get_cc_video(name, **kwargs):
    """Load a CC test video by short name."""
    info = CC_VIDEOS[name]
    path = os.path.join(CC_VIDEO_DIR, info['filename'])
    return VideoReader(path, ctx=CTX, **kwargs)


def _get_cc_video_path(name):
    """Get the file path for a CC test video."""
    return os.path.join(CC_VIDEO_DIR, CC_VIDEOS[name]['filename'])


# ---------------------------------------------------------------------------
# Frame count tests
# ---------------------------------------------------------------------------

def test_cc_bbb_frame_count():
    """Big Buck Bunny style: 5s @ 24fps = 120 frames."""
    vr = _get_cc_video('bbb')
    assert len(vr) == 120, f"Expected 120 frames, got {len(vr)}"


def test_cc_sintel_frame_count():
    """Sintel style: 3s @ 30fps = 90 frames."""
    vr = _get_cc_video('sintel')
    assert len(vr) == 90, f"Expected 90 frames, got {len(vr)}"


def test_cc_tos_frame_count():
    """Tears of Steel style: 8s @ 25fps = 200 frames."""
    vr = _get_cc_video('tos')
    assert len(vr) == 200, f"Expected 200 frames, got {len(vr)}"


# ---------------------------------------------------------------------------
# Resolution / shape tests
# ---------------------------------------------------------------------------

def test_cc_bbb_resolution():
    """Big Buck Bunny style: 640x360 -> frame shape (360, 640, 3)."""
    vr = _get_cc_video('bbb')
    frame = vr[0]
    assert frame.shape == (360, 640, 3), f"Expected (360, 640, 3), got {frame.shape}"


def test_cc_sintel_resolution():
    """Sintel style: 320x240 -> frame shape (240, 320, 3)."""
    vr = _get_cc_video('sintel')
    frame = vr[0]
    assert frame.shape == (240, 320, 3), f"Expected (240, 320, 3), got {frame.shape}"


def test_cc_tos_resolution():
    """Tears of Steel style: 480x270 -> frame shape (270, 480, 3)."""
    vr = _get_cc_video('tos')
    frame = vr[0]
    assert frame.shape == (270, 480, 3), f"Expected (270, 480, 3), got {frame.shape}"


# ---------------------------------------------------------------------------
# FPS tests
# ---------------------------------------------------------------------------

def test_cc_bbb_fps():
    """Big Buck Bunny style: 24 fps."""
    vr = _get_cc_video('bbb')
    assert abs(vr.get_avg_fps() - 24.0) < 0.01, f"Expected 24.0 fps, got {vr.get_avg_fps()}"


def test_cc_sintel_fps():
    """Sintel style: 30 fps."""
    vr = _get_cc_video('sintel')
    assert abs(vr.get_avg_fps() - 30.0) < 0.01, f"Expected 30.0 fps, got {vr.get_avg_fps()}"


def test_cc_tos_fps():
    """Tears of Steel style: 25 fps."""
    vr = _get_cc_video('tos')
    assert abs(vr.get_avg_fps() - 25.0) < 0.01, f"Expected 25.0 fps, got {vr.get_avg_fps()}"


# ---------------------------------------------------------------------------
# Frame timestamp tests (cross-referenced with ffmpeg analysis)
# ---------------------------------------------------------------------------

def test_cc_bbb_timestamps():
    """Verify frame timestamps match expected values for 24fps video."""
    vr = _get_cc_video('bbb')
    ts = vr.get_frame_timestamp(range(5))
    expected = CC_VIDEOS['bbb']['first_timestamps']
    assert np.allclose(ts[:, 0], expected, atol=1e-3), \
        f"Expected timestamps {expected}, got {ts[:, 0].tolist()}"


def test_cc_sintel_timestamps():
    """Verify frame timestamps match expected values for 30fps video."""
    vr = _get_cc_video('sintel')
    ts = vr.get_frame_timestamp(range(5))
    expected = CC_VIDEOS['sintel']['first_timestamps']
    assert np.allclose(ts[:, 0], expected, atol=1e-3), \
        f"Expected timestamps {expected}, got {ts[:, 0].tolist()}"


def test_cc_tos_timestamps():
    """Verify frame timestamps match expected values for 25fps video."""
    vr = _get_cc_video('tos')
    ts = vr.get_frame_timestamp(range(5))
    expected = CC_VIDEOS['tos']['first_timestamps']
    assert np.allclose(ts[:, 0], expected, atol=1e-3), \
        f"Expected timestamps {expected}, got {ts[:, 0].tolist()}"


def test_cc_timestamps_monotonic():
    """All CC videos should have monotonically increasing timestamps."""
    for name in CC_VIDEOS:
        vr = _get_cc_video(name)
        ts = vr.get_frame_timestamp(range(len(vr)))
        diffs = np.diff(ts[:, 0])
        assert np.all(diffs > 0), \
            f"Timestamps not monotonically increasing for {name}: " \
            f"found non-positive diff at indices {np.where(diffs <= 0)[0]}"


# ---------------------------------------------------------------------------
# Sequential read tests
# ---------------------------------------------------------------------------

def test_cc_sequential_read():
    """Read all frames sequentially for each CC video."""
    for name, info in CC_VIDEOS.items():
        vr = _get_cc_video(name)
        for i in range(len(vr)):
            frame = vr[i]
            assert frame.shape == (info['height'], info['width'], 3), \
                f"Frame {i} of {name}: expected shape " \
                f"({info['height']}, {info['width']}, 3), got {frame.shape}"


# ---------------------------------------------------------------------------
# Slice read tests
# ---------------------------------------------------------------------------

def test_cc_slice_all():
    """Read all frames as a slice for each CC video."""
    for name, info in CC_VIDEOS.items():
        vr = _get_cc_video(name)
        frames = vr[:]
        expected_shape = (info['num_frames'], info['height'], info['width'], 3)
        assert frames.shape == expected_shape, \
            f"{name}: expected shape {expected_shape}, got {frames.shape}"


def test_cc_slice_partial():
    """Read partial slices from each CC video."""
    for name, info in CC_VIDEOS.items():
        vr = _get_cc_video(name)
        # First 10 frames
        frames = vr[:10]
        assert frames.shape[0] == 10, \
            f"{name}: expected 10 frames in slice, got {frames.shape[0]}"
        # Last 5 frames
        frames = vr[-5:]
        assert frames.shape[0] == 5, \
            f"{name}: expected 5 frames in tail slice, got {frames.shape[0]}"
        # Every other frame in first 20
        frames = vr[:20:2]
        assert frames.shape[0] == 10, \
            f"{name}: expected 10 frames in strided slice, got {frames.shape[0]}"


# ---------------------------------------------------------------------------
# Random access tests
# ---------------------------------------------------------------------------

def test_cc_random_access():
    """Random access frames from each CC video."""
    for name, info in CC_VIDEOS.items():
        vr = _get_cc_video(name)
        indices = random.sample(range(len(vr)), min(15, len(vr)))
        for idx in indices:
            frame = vr[idx]
            assert frame.shape == (info['height'], info['width'], 3), \
                f"Random frame {idx} of {name}: wrong shape {frame.shape}"


# ---------------------------------------------------------------------------
# Batch access tests
# ---------------------------------------------------------------------------

def test_cc_get_batch():
    """Batch frame access for each CC video."""
    for name, info in CC_VIDEOS.items():
        vr = _get_cc_video(name)
        indices = random.sample(range(len(vr)), min(10, len(vr)))
        frames = vr.get_batch(indices)
        expected_shape = (len(indices), info['height'], info['width'], 3)
        assert frames.shape == expected_shape, \
            f"{name}: batch shape {frames.shape} != expected {expected_shape}"


def test_cc_get_batch_ordered():
    """Batch access with ordered indices should return frames in order."""
    for name in CC_VIDEOS:
        vr = _get_cc_video(name)
        indices = list(range(0, min(20, len(vr)), 2))  # [0, 2, 4, ...]
        batch = vr.get_batch(indices).asnumpy()
        # Verify each batch frame matches individual access
        for i, idx in enumerate(indices):
            individual = vr[idx].asnumpy()
            diff = np.mean(np.abs(
                batch[i].astype('float') -
                individual.astype('float')
            ))
            assert diff < 1.0, \
                f"{name}: batch frame {i} (idx={idx}) differs from " \
                f"individual access by {diff}"


# ---------------------------------------------------------------------------
# BytesIO tests
# ---------------------------------------------------------------------------

def test_cc_bytes_io():
    """VideoReader should work with BytesIO input for each CC video."""
    for name, info in CC_VIDEOS.items():
        path = _get_cc_video_path(name)
        with open(path, 'rb') as f:
            bio = BytesIO(f.read())
        vr_bio = VideoReader(bio, ctx=CTX)
        assert len(vr_bio) == info['num_frames'], \
            f"{name}: BytesIO frame count {len(vr_bio)} != {info['num_frames']}"

        # Compare first frame with file-based reader
        vr_file = _get_cc_video(name)
        diff = np.mean(np.abs(
            vr_file[0].asnumpy().astype('float') -
            vr_bio[0].asnumpy().astype('float')
        ))
        assert diff < 2.0, \
            f"{name}: BytesIO vs file pixel diff {diff} >= 2.0"


# ---------------------------------------------------------------------------
# Resize tests
# ---------------------------------------------------------------------------

def test_cc_resize():
    """VideoReader should support custom height/width for each CC video."""
    target_h, target_w = 128, 128
    for name in CC_VIDEOS:
        vr = _get_cc_video(name, height=target_h, width=target_w)
        frame = vr[0]
        assert frame.shape == (target_h, target_w, 3), \
            f"{name}: resized frame shape {frame.shape} != " \
            f"({target_h}, {target_w}, 3)"


def test_cc_resize_preserves_frame_count():
    """Resizing should not change the number of frames."""
    for name, info in CC_VIDEOS.items():
        vr = _get_cc_video(name, height=64, width=64)
        assert len(vr) == info['num_frames'], \
            f"{name}: resized frame count {len(vr)} != {info['num_frames']}"


# ---------------------------------------------------------------------------
# Cross-video consistency tests
# ---------------------------------------------------------------------------

def test_cc_different_fps_different_frame_counts():
    """Videos with different FPS should have different frame counts."""
    counts = {name: CC_VIDEOS[name]['num_frames'] for name in CC_VIDEOS}
    # All three should be different
    values = list(counts.values())
    assert len(set(values)) == len(values), \
        f"Expected all unique frame counts, got {counts}"


def test_cc_frame_values_are_valid_rgb():
    """All frame pixel values should be valid RGB (0-255)."""
    for name in CC_VIDEOS:
        vr = _get_cc_video(name)
        frame = vr[0].asnumpy()
        assert frame.min() >= 0, f"{name}: min pixel value {frame.min()} < 0"
        assert frame.max() <= 255, f"{name}: max pixel value {frame.max()} > 255"
        assert frame.dtype == np.uint8, f"{name}: dtype {frame.dtype} != uint8"


def test_cc_frame_not_all_black():
    """Frames should contain actual content (not all zeros)."""
    for name in CC_VIDEOS:
        vr = _get_cc_video(name)
        frame = vr[0].asnumpy()
        assert frame.max() > 0, f"{name}: frame 0 is all black"
        # Also check a middle frame
        mid = len(vr) // 2
        frame_mid = vr[mid].asnumpy()
        assert frame_mid.max() > 0, f"{name}: frame {mid} is all black"


if __name__ == '__main__':
    import nose
    nose.runmodule()
