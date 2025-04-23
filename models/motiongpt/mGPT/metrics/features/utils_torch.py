import torch

def distance_between_points(a, b):
    return torch.norm(a - b)

def distance_from_plane(a, b, c, p, threshold):
    ba = b - a
    ca = c - a
    cross = torch.cross(ca, ba)
    pa = p - a
    return torch.dot(cross, pa) / torch.norm(cross) > threshold

def angle_within_range(j1, j2, k1, k2, range):
    j = j2 - j1
    k = k2 - k1
    angle = torch.acos(torch.dot(j, k) / (torch.norm(j) * torch.norm(k)))
    angle = torch.degrees(angle)
    return (angle > range[0]) and (angle < range[1])

def velocity_direction_above_threshold(j1, j1_prev, j2, j2_prev, p, p_prev, threshold, time_per_frame=1/120.0):
    velocity = (p - j1) - (p_prev - j1_prev)
    direction = j2 - j1
    velocity_along_direction = torch.dot(velocity, direction) / torch.norm(direction)
    velocity_along_direction /= time_per_frame
    return velocity_along_direction > threshold

def calc_average_velocity(positions, i, joint_idx, sliding_window, frame_time):
    current_window = 0
    average_velocity = torch.zeros(positions.size(2))
    for j in range(-sliding_window, sliding_window + 1):
        if i + j - 1 < 0 or i + j >= len(positions):
            continue
        average_velocity += (positions[i + j, joint_idx] - positions[i + j - 1, joint_idx])
        current_window += 1
    return torch.norm(average_velocity / (current_window * frame_time))

# 其他的函数依此类推，将 numpy 的操作改为 torch 的操作
