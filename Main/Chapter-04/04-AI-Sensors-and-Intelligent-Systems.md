#!/usr/bin/env python3
"""
Module 04: AI Sensors & Intelligent Systems - code companion
=============================================================
Course : Hardware, Networks & Behaviourist AI
Files  : this single file (Python 3.8+, standard library only, no pip install)

Every worked example (1.A-5.A) and every numeric activity answer from the
module is implemented here as a small, reusable function. Each activity answer
is checked automatically against the Answer Key, so a run ends with PASS/FAIL.

Usage
-----
    python module04_ai_sensors.py          # run everything
    python module04_ai_sensors.py 3        # run only Section 3
    python module04_ai_sensors.py 1 4 5    # run Sections 1, 4 and 5

Note: all sensor numbers are illustrative teaching examples, not real
product specifications.
"""

import math
import sys

C = 3e8  # speed of light, m/s
_results = []  # (label, passed)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def check(label, got, expected, tol=0.01):
    """Compare a computed value with the Answer Key value and record it."""
    ok = abs(got - expected) <= tol
    _results.append((label, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: got {got:.4g}, expected {expected:.4g}")


def title(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


# --------------------------------------------------------------------------
# Section 1: Types of sensors
# --------------------------------------------------------------------------
def pixels_on_target(width_px, hfov_deg, size_m, dist_m):
    """Pinhole model: pixels covered by a target of width size_m at dist_m."""
    f = (width_px / 2) / math.tan(math.radians(hfov_deg / 2))
    return f * size_m / dist_m


def tof_distance(t_s):
    """LiDAR time of flight: d = c*t/2."""
    return C * t_s / 2


def speed_of_sound(temp_c):
    return 331.3 + 0.606 * temp_c


def ultrasonic_distance(t_s, v=343.0):
    """Ultrasonic echo: d = v*t/2."""
    return v * t_s / 2


def radar_speed(delta_f_hz, f0_hz):
    """Doppler radar: v = (df*c) / (2*f0)."""
    return delta_f_hz * C / (2 * f0_hz)


def lidar_data(beams, step_deg, hz, bytes_per_point=16):
    """Return (points per frame, MB per frame, MB per second)."""
    pts = beams * round(360 / step_deg)
    mb_frame = pts * bytes_per_point / 1e6
    return pts, mb_frame, mb_frame * hz


def section1():
    title("SECTION 1: Types of Sensors")

    print("\nWorked example 1.A - pixels on a 0.5 m child")
    for d in (50, 150):
        print(f"  {d} m: {pixels_on_target(1920, 60, 0.5, d):.1f} pixels")

    print("\nWorked example 1.B - LiDAR time of flight (400 ns)")
    print(f"  distance = {tof_distance(400e-9):.0f} m")

    print("\nWorked example 1.C - LiDAR data rate (128 beams, 0.2 deg, 10 Hz)")
    pts, mb_f, mb_s = lidar_data(128, 0.2, 10)
    print(f"  points/frame = {pts}, points/s = {pts * 10}")
    print(f"  {mb_f} MB per frame, {mb_s} MB/s")

    print("\nWorked example 1.D - ultrasonic echo and temperature error")
    print(f"  (a) 5.8 ms echo -> {ultrasonic_distance(5.8e-3):.2f} m")
    t = 2 * 1.00 / speed_of_sound(0)  # true 1.00 m at 0 C
    wrong = ultrasonic_distance(t, 343.0)
    print(f"  (b) echo time {t * 1e3:.2f} ms; sensor assuming 343 m/s reports "
          f"{wrong:.3f} m ({(wrong - 1) * 100:.1f}% error)")

    print("\nRadar example - 77 GHz, 10.27 kHz shift")
    print(f"  v = {radar_speed(10270, 77e9):.1f} m/s")

    print("\nActivities 1.A-1.E (checked against the Answer Key)")
    check("1.A pixels on 1.8 m car at 100 m", pixels_on_target(3840, 90, 1.8, 100), 34.56)
    check("1.B LiDAR 1.0 us (m)", tof_distance(1.0e-6), 150, 1e-6)
    check("1.C ultrasonic 11.7 ms (m)", ultrasonic_distance(11.7e-3), 2.0, 0.01)
    check("1.D radar 15.4 kHz (m/s)", radar_speed(15400, 77e9), 30, 1e-6)
    pts, mb_f, mb_s = lidar_data(64, 0.4, 10)
    check("1.E points per frame", pts, 57600, 0)
    check("1.E MB per frame", mb_f, 0.9216, 1e-4)


# --------------------------------------------------------------------------
# Section 2: Data acquisition and real-time processing
# --------------------------------------------------------------------------
def camera_rate_mb_s(w, h, bits, fps):
    return w * h * bits / 8 * fps / 1e6


def stopping(v_ms, latency_s, decel):
    """Return (blind distance, braking distance, total)."""
    blind = v_ms * latency_s
    brake = v_ms ** 2 / (2 * decel)
    return blind, brake, blind + brake


def compute_needed_tflops(gflop_per_frame, cams, fps, utilisation=1.0):
    return gflop_per_frame * 1e9 * cams * fps / 1e12 / utilisation


LATENCY_BUDGET_MS = {
    "Exposure and readout": 10, "ISP": 5, "Transfer to memory": 2,
    "Neural-network inference": 30, "Fusion and tracking": 10,
    "Planning": 15, "Actuation (brake build-up)": 30,
}


def section2():
    title("SECTION 2: Data Acquisition and Real-Time Processing")

    print("\nWorked example 2.A - 8 cameras, 1920x1080, 12-bit, 36 fps")
    per_cam = camera_rate_mb_s(1920, 1080, 12, 36)
    print(f"  frame = {1920 * 1080 * 12 / 8} bytes")
    print(f"  {per_cam:.1f} MB/s per camera, {per_cam * 8 / 1000:.2f} GB/s total")

    print("\nWorked example 2.B - distance travelled while thinking (100 km/h)")
    v = 100 / 3.6
    for lat in (0.1, 0.3):
        blind, brake, total = stopping(v, lat, 6.0)
        print(f"  latency {lat}: blind {blind:.2f} m, brake {brake:.1f} m, total {total:.1f} m")

    print("\nTable 2.1 - latency budget")
    for stage, ms in LATENCY_BUDGET_MS.items():
        print(f"  {stage:<28}{ms:>4} ms")
    print(f"  {'Total':<28}{sum(LATENCY_BUDGET_MS.values()):>4} ms (target 100 ms)")

    print("\nWorked example 2.C - compute for a 100 GFLOP network, 8 cams, 36 fps")
    need = compute_needed_tflops(100, 8, 36)
    print(f"  {need:.1f} TFLOP/s sustained, "
          f"{compute_needed_tflops(100, 8, 36, 0.4):.1f} TFLOP/s peak at 40% utilisation")

    print("\nActivities 2.A-2.E (checked against the Answer Key)")
    one = camera_rate_mb_s(1280, 960, 10, 30)
    check("2.A MB/s per camera", one, 46.08)
    check("2.A MB/s, six cameras", one * 6, 276.48)
    check("2.B stopping distance (m)", stopping(72 / 3.6, 0.15, 5.0)[2], 43.0, 1e-6)
    new_total = sum(LATENCY_BUDGET_MS.values()) - 30 + 60
    check("2.C new total (ms)", new_total, 132, 0)
    check("2.C miss vs 100 ms target (ms)", new_total - 100, 32, 0)
    check("2.D sustained TFLOP/s", compute_needed_tflops(60, 12, 30), 21.6, 1e-6)
    check("2.D peak at 30% (TFLOP/s)", compute_needed_tflops(60, 12, 30, 0.3), 72, 1e-6)
    check("2.E offset between views (m)", 0.015 * 30, 0.45, 1e-9)


# --------------------------------------------------------------------------
# Section 3: Sensor fusion and perception
# --------------------------------------------------------------------------
def inverse_variance_fuse(z, sigma):
    """Fuse measurements z with std-devs sigma. Returns (estimate, sigma)."""
    w = [1 / s ** 2 for s in sigma]
    est = sum(wi * zi for wi, zi in zip(w, z)) / sum(w)
    return est, (1 / sum(w)) ** 0.5


def kalman_predict(P, Q):
    return P + Q


def kalman_update(x, P, z, R):
    K = P / (P + R)
    return x + K * (z - x), (1 - K) * P


def stereo_depth(f_px, baseline_m, disparity_px):
    return f_px * baseline_m / disparity_px


def iou_shifted(length, width, overlap_length):
    """IoU of two equal boxes overlapping by overlap_length along the length."""
    area = length * width
    inter = overlap_length * width
    return inter / (2 * area - inter)


def precision_recall(tp, fp, fn):
    return tp / (tp + fp), tp / (tp + fn)


def section3():
    title("SECTION 3: Sensor Fusion and Environmental Perception")

    print("\nWorked example 3.A - radar + camera")
    est, sig = inverse_variance_fuse([50.0, 52.0], [0.5, 2.0])
    print(f"  fused = {est:.2f} m, sigma = {sig:.2f} m")

    print("\nWorked example 3.B - Kalman filter: predict + two updates")
    x, P = 50.0, kalman_predict(4.0, 1.0)
    print(f"  after predict: P = {P}")
    x, P = kalman_update(x, P, 52.0, 1.0)
    print(f"  update 1 (radar):  x = {x:.2f}, P = {P:.2f}")
    x, P = kalman_update(x, P, 50.5, 4.0)
    print(f"  update 2 (camera): x = {x:.2f}, P = {P:.2f}")

    print("\nWorked example 3.C - stereo depth sensitivity (f=1000, B=0.30)")
    for d in (5, 6, 7):
        print(f"  disparity {d} px -> {stereo_depth(1000, 0.3, d):.1f} m")
    print(f"  error per pixel at 50 m ~ {50 ** 2 / (1000 * 0.3):.1f} m")

    print("\nWorked example 3.D - IoU, precision, recall")
    print(f"  IoU = {iou_shifted(4, 2, 3):.1f}")
    p, r = precision_recall(90, 10, 30)
    print(f"  precision = {p:.2f}, recall = {r:.2f}")
    print(f"  1 deg yaw error at 50 m -> {50 * math.tan(math.radians(1)):.2f} m sideways")

    print("\nActivities 3.A-3.F (checked against the Answer Key)")
    est, sig = inverse_variance_fuse([30.0, 31.0], [0.1, 1.0])
    check("3.A fused estimate (m)", est, 30.01)
    check("3.A fused sigma (m)", sig, 0.10)
    x, P = kalman_update(20.0, kalman_predict(2.0, 0.5), 21.0, 1.5)
    check("3.B x", x, 20.63)
    check("3.B P", P, 0.94)
    check("3.C depth, 12 px (m)", stereo_depth(1200, 0.5, 12), 50.0, 1e-9)
    check("3.C depth, 11 px (m)", stereo_depth(1200, 0.5, 11), 54.5)
    check("3.D IoU", iou_shifted(5, 2, 4), 0.67)
    p, r = precision_recall(72, 8, 8)
    check("3.E precision", p, 0.90)
    check("3.E recall", r, 0.90)
    check("3.F sideways offset at 80 m (m)", 80 * math.tan(math.radians(0.5)), 0.70)


# --------------------------------------------------------------------------
# Section 4: Hardware + AI decision-making
# --------------------------------------------------------------------------
def impact_speed(detect_m, v_ms, latency_s, decel):
    """Speed at the obstacle (0.0 if the vehicle stops in time)."""
    remaining = detect_m - v_ms * latency_s
    v2 = v_ms ** 2 - 2 * decel * remaining
    return 0.0 if v2 <= 0 else v2 ** 0.5


def brake_threshold(cost_false_alarm, cost_miss):
    """Brake when p > cost_fp / (cost_fp + cost_fn)."""
    return cost_false_alarm / (cost_false_alarm + cost_miss)


def section4():
    title("SECTION 4: Integration of Hardware and AI Decision-Making")

    print("\nWorked example 4.A - can the car stop? (50 km/h, 0.15 s, 7 m/s^2)")
    v = 50 / 3.6
    blind, brake, total = stopping(v, 0.15, 7.0)
    print(f"  latency {blind:.2f} m + braking {brake:.2f} m = {total:.1f} m needed")
    for d in (25, 15):
        print(f"  detected at {d} m -> impact speed {impact_speed(d, v, 0.15, 7.0):.1f} m/s")

    print("\nWorked example 4.B - when should the car brake?")
    print(f"  threshold p > {brake_threshold(1, 1000):.6f}")
    sigma = 0.83
    print(f"  conservative planning distance = {51.47 - 2 * sigma:.1f} m")

    print("\nRedundancy: independent vs correlated failures (5% miss each)")
    print(f"  independent: {0.05 * 0.05:.2%}   correlated: {0.05:.0%}")

    print("\nActivities 4.A-4.D (checked against the Answer Key)")
    check("4.A minimum detection distance (m)", stopping(36 / 3.6, 0.2, 6.0)[2], 10.3, 0.05)
    check("4.B distance for zero impact (m)", stopping(v, 0.15, 7.0)[2], 15.9, 0.05)
    check("4.B impact speed at that distance", impact_speed(total, v, 0.15, 7.0), 0.0, 1e-6)
    check("4.C brake threshold", brake_threshold(2, 600), 0.0033, 0.0001)
    check("4.D both miss (independent)", 0.02 * 0.02, 0.0004, 1e-9)
    check("4.D both miss (correlated)", 0.02, 0.02, 1e-9)


# --------------------------------------------------------------------------
# Section 5: Case study - Tesla
# --------------------------------------------------------------------------
def npu_tops(rows, cols, clock_hz):
    """One MAC = 2 operations."""
    return rows * cols * 2 * clock_hz / 1e12


def section5():
    title("SECTION 5: Case Study - Tesla's Autonomous Driving Technology")

    print("\nWorked example 5.A - how many TOPS? (96x96 MACs at 2 GHz)")
    one = npu_tops(96, 96, 2e9)
    print(f"  one accelerator: {one:.1f} TOPS")
    print(f"  four accelerators: {one * 4:.1f} TOPS peak")
    print(f"  at 40% utilisation: {one * 4 * 0.4:.1f} TOPS usable "
          f"(needs {compute_needed_tflops(100, 8, 36):.1f} in Example 2.C)")

    print("\nCase-study question 3 (checked against the Answer Key)")
    check("Q3 TOPS for 128x128 at 1.5 GHz", npu_tops(128, 128, 1.5e9), 49.2, 0.05)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
SECTIONS = {"1": section1, "2": section2, "3": section3, "4": section4, "5": section5}


def main(argv):
    wanted = [a for a in argv if a in SECTIONS] or list(SECTIONS)
    for key in wanted:
        SECTIONS[key]()
    failed = [label for label, ok in _results if not ok]
    title("SUMMARY")
    print(f"  {len(_results) - len(failed)}/{len(_results)} checks passed")
    for label in failed:
        print(f"  FAILED: {label}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
