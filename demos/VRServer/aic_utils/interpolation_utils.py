import numpy as np
from scipy import signal as sig

def linear_interpolation(last_frame, next_frame, duration, slerp_func):
    """
    Perform linear interpolation between two animation frames.
    
    This function creates a smooth transition between two animation frames by
    linearly interpolating positions and using spherical linear interpolation (slerp)
    for rotations.
    
    Args:
        last_frame (dict): The starting frame with position and rotation values
        next_frame (dict): The target frame with position and rotation values
        duration (float): Duration of the transition in seconds
        slerp_func (callable): Function for spherical linear interpolation of quaternions
        
    Returns:
        list: A list of interpolated frames for the transition
    """
    num_blend_frames = int(duration * 30)
    blend_frames = []
    all_keys = set(last_frame.keys()).union(set(next_frame.keys()))
    for i in range(num_blend_frames):
        alpha = i / num_blend_frames
        blend_frame = {}
        for key in all_keys:
            if key.endswith('_position'):
                start_val = np.array(last_frame.get(key, [0, 0, 0]))
                end_val = np.array(next_frame.get(key, [0, 0, 0]))
                blend_val = (1 - alpha) * start_val + alpha * end_val
                blend_frame[key] = blend_val.tolist()
            elif key.endswith('_rotation'):
                start_val = np.array(last_frame.get(key, [0, 0, 0, 1]))
                end_val = np.array(next_frame.get(key, [0, 0, 0, 1]))
                blend_val = slerp_func(start_val, end_val, alpha)
                blend_frame[key] = blend_val.tolist()
        blend_frames.append(blend_frame)
    return blend_frames


def quintic_polynomial_1d(motion_seq_a, motion_seq_b, time=0.5, fps=30):
    """
    Create a quintic polynomial interpolation between two motion sequences.
    
    This advanced interpolation method creates a seamless transition between two
    motion sequences by solving a quintic polynomial system that preserves
    velocity and acceleration at the boundaries. It also applies filtering for
    smoother results.
    
    Args:
        motion_seq_a (numpy.ndarray): First motion sequence
        motion_seq_b (numpy.ndarray): Second motion sequence
        time (float): Duration of the transition in seconds
        fps (int): Frames per second
        
    Returns:
        numpy.ndarray: A combined motion sequence with a smooth transition
    """
    n_frames_a = motion_seq_a.shape[1]
    n_frames_b = motion_seq_b.shape[1]

    n_frames_transit = int(time * fps)
    n_frames_reference = n_frames_transit + 2
    time_reference = n_frames_transit / fps

    tf = time_reference / 4
    box_length = 4
    dt = 1 / fps
    x0 = motion_seq_a[-1]
    xt = motion_seq_b[0]

    # Calculate initial velocity from previous frames
    v0s = []
    for i in range(box_length):
        v0s.append((motion_seq_a[-box_length + i] - motion_seq_a[-box_length * 2 + i]) / (box_length * dt))
    v0s = np.array(v0s)
    v0 = v0s.mean(0)

    # Calculate initial acceleration from velocity changes
    a0s = []
    a_box_length = box_length // 2
    for i in range(a_box_length):
        a0s.append((v0s[-a_box_length + i] - v0s[-a_box_length * 2 + i]) / (a_box_length * dt))
    a0s = np.array(a0s)
    a0 = a0s.mean(0)

    # Calculate target velocity from following frames
    vts = []
    for i in range(box_length):
        vts.append((motion_seq_b[box_length * 2 - i] - motion_seq_b[box_length - i]) / (box_length * dt))
    vts = np.array(vts)
    vt = vts.mean(0)

    # Calculate target acceleration from velocity changes
    ats = []
    a_box_length = box_length // 2
    for i in range(a_box_length):
        ats.append((vts[a_box_length + i] - vts[i]) / (a_box_length * dt))
    ats = np.array(ats)
    at = ats.mean(0)

    # Precompute powers of the total transition time
    tf2 = tf * tf
    tf3 = tf2 * tf
    tf4 = tf3 * tf
    tf5 = tf4 * tf

    # Coefficients for the quintic polynomial
    F = x0
    E = v0
    D = 0.5 * a0

    # Solve the system of equations for the remaining coefficients
    L4 = xt - D * tf2 - E * tf - F
    L5 = vt - 2 * D * tf - E
    L6 = at - 2 * D
    factor = np.expand_dims(np.array([L4, L5, L6]).T, -1)

    matrix = np.array([[tf5, tf4, tf3], [5 * tf4, 4 * tf3, 3 * tf2], [20 * tf3, 12 * tf2, 6 * tf]])
    matrix = np.repeat(matrix[None, :, :], len(motion_seq_a), axis=0)
    matrix_inv = np.linalg.inv(matrix)
    result = np.matmul(matrix_inv, factor)

    A, B, C = result[:, 0:1, 0], result[:, 1:2, 0], result[:, 2:3, 0]

    # Evaluate the quintic polynomial at the transition points
    xs = np.expand_dims(np.linspace(0, tf, n_frames_reference), 0)
    xs2 = xs * xs
    xs3 = xs2 * xs
    xs4 = xs3 * xs
    xs5 = xs4 * xs
    intervals = A * xs5 + B * xs4 + C * xs3 + D[:, None] * xs2 + E[:, None] * xs + F[:, None]

    # Combine the sequences with the transition
    x = np.concatenate([motion_seq_a, intervals[:, 1:-1], motion_seq_b], 1)
    n_framesB = motion_seq_b.shape[1]

    # Apply low-pass filter for smoother transitions
    b, a = sig.butter(8, 0.075)
    y = sig.filtfilt(b, a, x, axis=1)
    
    # Create a mask for blending filtered and original sequences
    length = 10
    blend_point = (y.shape[1] - n_framesB) / y.shape[1]
    A_start = length * 2 * blend_point - length
    mask_x = np.linspace(-length, length, y.shape[1]) - A_start
    mask = np.exp(-(mask_x ** 2))
    result = mask * y + (1 - mask) * x
    result = result.reshape(-1, n_frames_a + n_frames_b + n_frames_transit)
    result = result.T
    return result


def quintic_interpolation(last_frame, next_frame, duration, slerp_func):
    """
    Perform quintic interpolation between two animation frames.
    
    This function creates a smoother transition than linear interpolation by using
    quintic Hermite spline interpolation for positions and slerp for rotations.
    
    Args:
        last_frame (dict): The starting frame with position and rotation values
        next_frame (dict): The target frame with position and rotation values
        duration (float): Duration of the transition in seconds
        slerp_func (callable): Function for spherical linear interpolation of quaternions
        
    Returns:
        list: A list of interpolated frames for the transition
    """
    num_blend_frames = int(duration * 30)
    blend_frames = []
    all_keys = set(last_frame.keys()).union(set(next_frame.keys()))

    for i in range(num_blend_frames):
        t = i / num_blend_frames

        # Quintic Hermite spline interpolation factor
        t2 = t * t
        t3 = t2 * t
        t4 = t3 * t
        t5 = t4 * t
        h00 = 6 * t5 - 15 * t4 + 10 * t3  # Quintic interpolation for smoother transitions

        blend_frame = {}
        for key in all_keys:
            if key.endswith('_position'):
                start_val = np.array(last_frame.get(key, [0, 0, 0]))
                end_val = np.array(next_frame.get(key, [0, 0, 0]))

                blend_val = (1 - h00) * start_val + h00 * end_val
                blend_frame[key] = blend_val.tolist()
            elif key.endswith('_rotation'):
                start_val = np.array(last_frame.get(key, [0, 0, 0, 1]))
                end_val = np.array(next_frame.get(key, [0, 0, 0, 1]))

                # Handle quaternion double-cover problem
                if np.dot(start_val, end_val) < 0.0:
                    end_val = -end_val

                blend_val = slerp_func(start_val, end_val, h00)
                blend_frame[key] = blend_val.tolist()

        blend_frames.append(blend_frame)

    return blend_frames


def slerp(q1, q2, t):
    """
    Perform spherical linear interpolation (slerp) between two quaternions.
    
    This function interpolates between two quaternions along the shortest arc
    on the unit sphere, preserving the rotation properties.
    
    Args:
        q1 (numpy.ndarray): First quaternion [x, y, z, w]
        q2 (numpy.ndarray): Second quaternion [x, y, z, w]
        t (float): Interpolation parameter in range [0, 1]
        
    Returns:
        numpy.ndarray: Interpolated quaternion
    """
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot_product = np.dot(q1, q2)
    
    # Handle quaternion double-cover problem
    if dot_product < 0.0:
        q2 = -q2
        dot_product = -dot_product
    
    # If quaternions are very close, use linear interpolation
    dot_threshold = 0.995
    if dot_product > dot_threshold:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)

    # Standard spherical linear interpolation
    theta_0 = np.arccos(dot_product)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    s1 = np.cos(theta) - dot_product * sin_theta / sin_theta_0
    s2 = sin_theta / sin_theta_0
    return (s1 * q1) + (s2 * q2)