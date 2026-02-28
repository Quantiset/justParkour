import cv2
import numpy as np
from collections import deque

cap = cv2.VideoCapture("minecraft.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)

use_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
print("Using GPU" if use_cuda else "Using CPU")

if use_cuda:
    gpu_flow = cv2.cuda_FarnebackOpticalFlow.create(
        0.5, 3, False, 21, 5, 7, 1.5, 0  # larger window = more sensitive
    )

prev_gray = None
jump_times = []
frame_idx = 0
in_jump = False

# Rolling baseline to adapt to camera movement
vy_history = deque(maxlen=30)  # ~1 second of history

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_small = cv2.resize(frame, None, fx=0.5, fy=0.5)
    gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

    if prev_gray is not None:
        if use_cuda:
            gpu_prev = cv2.cuda_GpuMat()
            gpu_curr = cv2.cuda_GpuMat()
            gpu_prev.upload(prev_gray)
            gpu_curr.upload(gray)
            flow = gpu_flow.calc(gpu_prev, gpu_curr, None).download()
        else:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                0.5, 3, 21, 5, 7, 1.5, 0
            )

        vy = flow[..., 1]
        vx = flow[..., 0]

        h, w = vy.shape
        center_vy = vy[h//4:3*h//4, w//4:3*w//4]
        center_vx = vx[h//4:3*h//4, w//4:3*w//4]

        mean_vy = np.mean(center_vy)
        mean_vx = np.mean(center_vx)

        # Adaptive baseline: subtract recent average vertical motion
        # so slow camera tilt doesn't count as a jump
        baseline_vy = np.mean(vy_history) if vy_history else 0
        adjusted_vy = mean_vy - baseline_vy
        vy_history.append(mean_vy)

        # Much looser thresholds
        # adjusted_vy < -0.8 catches small jumps
        # horizontal check is also relaxed since players move while jumping
        if adjusted_vy < -0.8 and abs(mean_vx) < 3.5:
            if not in_jump:
                timestamp = round(frame_idx / fps, 3)
                jump_times.append(timestamp)
                in_jump = True
        else:
            in_jump = False

    prev_gray = gray
    frame_idx += 1

cap.release()

print(f"\nTotal jumps: {len(jump_times)}")
for t in jump_times:
    mins = int(t // 60)
    secs = t % 60
    print(f"  {mins}m {secs:.2f}s")