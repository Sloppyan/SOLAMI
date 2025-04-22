import numpy as np
from scipy import signal as sig
from typing import Union

from xrmodama.utils.log_utils import get_logger, logging


class QuinticPolynomialBlendOps:
    """Quintic polynomial blending for rotation matrix or position vectors.

    Input motion cannot be shorter than
    QuinticPolynomialBlendOps.NFRAMES_LOWERBOUND.
    """
    NFRAMES_LOWERBOUND = 9

    def __init__(self,
                 transit_time: float,
                 fps: int,
                 threshold: float = 0.5,
                 compromise_method: Union[str, None] = None,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """
        Args:
            transit_time (float):
                Time for transition, in seconds.
            fps (int):
                Frame per second, for motion a, b and output.
            threshold (float):
                The threshold for allowable cutting.
                When n_frames_deleted > n_frames_src * threshold,
                the cutting will be warned.
            compromise_method (Union[str, None], optional):
                The method for compromise, delete, keep or None.
                Delete: Delete the short motion and repeat the
                    long one. Do not select this if the short motion
                    contains important information.
                Keep: Keep the short motion and repeat it to
                    meet the blending requirement.
                If None, no compromising and raise error.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        self.transit_time = transit_time
        self.fps = fps
        self.threshold = threshold
        self.compromise_method = compromise_method \
            if compromise_method is None \
            else compromise_method.lower()
        self.logger = get_logger(logger)

    def blend_quaternions(
        self,
        motion_quat_a: np.ndarray,
        motion_quat_b: np.ndarray,
        cut_a: bool = True,
        cut_b: bool = True,
        transit_time_overwrite: Union[float, None] = None,
        inplace: bool = False,
    ) -> np.ndarray:
        """Run QuinticPolynomialBlend between 2 motion quaternion sequences.

        Args:
            motion_quat_a (np.ndarray):
                Multi-frame quaternion for motion a,
                in shape [n_frames_a, n_joints, 4].
            motion_quat_b (np.ndarray):
                Multi-frame quaternion for motion b,
                in shape [n_frames_b, n_joints, 4].
            cut_a (bool, optional):
                If True, motion a will be shortened by
                transition motion.
                Defaults to True.
            cut_b (bool, optional):
                If True, motion b will be shortened by
                transition motion.
                Defaults to True.
            transit_time_overwrite (Union[float, None], optional):
                If not None, self.transit_time will be overwritten
                by this value.
                Defaults to None.
            inplace (bool, optional):
                If True, input motion a and b will be overwritten by
                transition motion.
                Defaults to False.

        Raises:
            ValueError:
                Input motion is too short and no
                compromise method is selected.

        Returns:
            np.ndarray:
                Blended motion,
                in shape [n_frames_a + n_frames_b, n_joints, 4].
        """
        n_frames_a, n_joints_a = motion_quat_a.shape[:2]
        n_frames_b, n_joints_b = motion_quat_b.shape[:2]
        if n_joints_a != n_joints_b:
            self.logger.error('Numbers of joints are different: ' +
                              f'n_joints_a={n_joints_a}, ' +
                              f'n_joints_b={n_joints_b}')
            raise ValueError

        blend_quat = self.blend_vector(
            motion_vec_a=motion_quat_a,
            motion_vec_b=motion_quat_b,
            cut_a=cut_a,
            cut_b=cut_b,
            transit_time_overwrite=transit_time_overwrite,
            inplace=False)

        if inplace:
            motion_quat_a[:] = blend_quat[:n_frames_a]
            motion_quat_b[:] = blend_quat[n_frames_a:]
        return blend_quat

    def blend_vector(
        self,
        motion_vec_a: np.ndarray,
        motion_vec_b: np.ndarray,
        cut_a: bool = True,
        cut_b: bool = True,
        transit_time_overwrite: Union[float, None] = None,
        inplace: bool = False,
    ) -> np.ndarray:
        """Run QuinticPolynomialBlend between 2 motion vectors like
        translation or quaternion.

        Args:
            motion_vec_a (np.ndarray):
                Multi-frame vector for motion a,
                in shape [n_frames, n_dim].
            motion_vec_b (np.ndarray):
                Multi-frame vector for motion b,
                in shape [n_frames, n_dim].
            cut_a (bool, optional):
                If True, motion a will be shortened by
                transition motion.
                Defaults to True.
            cut_b (bool, optional):
                If True, motion b will be shortened by
                transition motion.
                Defaults to True.
            transit_time_overwrite (Union[float, None], optional):
                If not None, self.transit_time will be overwritten
                by this value.
                Defaults to None.
            inplace (bool, optional):
                If True, input motion a and b will be overwritten by
                transition motion.
                Defaults to False.

        Raises:
            ValueError:
                Input motion is too short and no
                compromise method is selected.

        Returns:
            np.ndarray:
                Merged motion, in shape [n_frames, n_dim].
        """
        # check shape
        n_frames_a, n_dim_a = motion_vec_a.shape
        n_frames_b, n_dim_b = motion_vec_b.shape
        assert n_frames_a > 0 and n_frames_b > 0
        if n_dim_a != n_dim_b:
            self.logger.error('Numbers of dimensions are different: ' +
                              f'n_dim_a={n_dim_a}, n_dim_b={n_dim_b}')
            raise ValueError
        transit_time = self.transit_time \
            if transit_time_overwrite is None \
            else transit_time_overwrite
        n_frames_transit = int(transit_time * self.fps)
        # cut both a and b for transit motion
        if cut_a and cut_b:
            n_frames_cut_a = int(n_frames_transit / 2)
            n_frames_cut_b = n_frames_transit - n_frames_cut_a
        # replace a's tail with transit motion
        elif cut_a and not cut_b:
            n_frames_cut_a = n_frames_transit
            n_frames_cut_b = 0
        # replace b's head with transit motion
        elif not cut_a and cut_b:
            n_frames_cut_a = 0
            n_frames_cut_b = n_frames_transit
        else:
            n_frames_cut_a = 0
            n_frames_cut_b = 0
        cut_dict = dict(
            a=(n_frames_cut_a, n_frames_a), b=(n_frames_cut_b, n_frames_b))
        need_compromise = False
        short_motion_key = None
        for key, (n_frames_cut, n_frames_src) in cut_dict.items():
            if self.__class__.NFRAMES_LOWERBOUND > \
                    n_frames_src - n_frames_cut:
                if self.compromise_method is not None:
                    need_compromise = True
                    short_motion_key = key
                    break
                else:
                    self.logger.error(
                        f'Motion_{key} is too short to cut. ' +
                        f'n_frames_cut={n_frames_cut}, ' +
                        f'n_frames_src={n_frames_src}'
                    )
                    raise ValueError
            # 如果 n_frames_a * self.threshold 小于生成帧的一半，就认为当前帧数过少。
            if n_frames_cut > n_frames_src * self.threshold:
                self.logger.warning(
                    f'Motion_{key} is short,' +
                    ' n_frames_cut reaching threshold. ' +
                    f'n_frames_cut={n_frames_cut}, ' +
                    f'n_frames_src={n_frames_src},' +
                    f' threshold={self.threshold}')
        if need_compromise:
            if self.compromise_method == 'delete':
                blend_seq = self._compromise_del_short(
                    motion_seq_a=motion_vec_a,
                    motion_seq_b=motion_vec_b,
                    short_key=short_motion_key,
                    cut_dict=cut_dict)
            elif self.compromise_method == 'keep':
                blend_seq = self._compromise_keep_short(
                    motion_seq_a=motion_vec_a,
                    motion_seq_b=motion_vec_b,
                    short_key=short_motion_key,
                    transit_time=transit_time,
                    cut_dict=cut_dict)
            else:
                self.logger.error(
                    f'Unknown compromise method: {self.compromise_method}')
                raise ValueError
        else:
            blend_seq = self._cut_and_blend(
                motion_seq_a=motion_vec_a,
                motion_seq_b=motion_vec_b,
                n_frames_cut_a=n_frames_cut_a,
                n_frames_cut_b=n_frames_cut_b,
                transit_time=transit_time)
        if inplace:
            if n_frames_cut_a > 0:
                motion_vec_a[-n_frames_cut_a:, :] = \
                    blend_seq[n_frames_a-n_frames_cut_a: n_frames_a, :]
            if n_frames_cut_b > 0:
                motion_vec_b[:n_frames_cut_b, :] = \
                    blend_seq[n_frames_a:n_frames_a+n_frames_cut_b, :]
        return blend_seq

    def _compromise_del_short(self,
                              motion_seq_a: np.ndarray,
                              motion_seq_b: np.ndarray,
                              cut_dict: dict,
                              short_key: str) -> np.ndarray:
        n_frames_a = cut_dict['a'][1]
        n_frames_b = cut_dict['b'][1]
        self.logger.warning('Compromise deleting short: delete' +
                            f' motion_{short_key} and abort blending.')
        # if a is too short
        # delete a and repeat b's head
        if short_key == 'a':
            n_frames_diff = motion_seq_a.shape[0]
            repeated_seq = motion_seq_b[:1, :].repeat(
                n_frames_diff, axis=0)
            merged_motion_seq = np.concatenate(
                (repeated_seq, motion_seq_b), axis=0)
        # if b is too short
        # delete b repeat a's tail
        else:
            n_frames_diff = motion_seq_b.shape[0]
            repeated_seq = motion_seq_a[-1:, :].repeat(
                n_frames_diff, axis=0)
            merged_motion_seq = np.concatenate(
                (motion_seq_a, repeated_seq), axis=0)
        self.logger.warning(
            'Motion\'s n_frames before compromise: ' +
            f'motion_a={n_frames_a}, motion_b={n_frames_b}\n' +
            'Motion\'s n_frames after compromise: ' +
            f'merged_motion_seq={merged_motion_seq.shape[0]}')
        return merged_motion_seq

    def _compromise_keep_short(self,
                               motion_seq_a: np.ndarray,
                               motion_seq_b: np.ndarray,
                               transit_time: int,
                               cut_dict: dict,
                               short_key: str) -> np.ndarray:
        n_frames_a = cut_dict['a'][1]
        n_frames_b = cut_dict['b'][1]
        n_frames_interval = int(transit_time * self.fps)
        if n_frames_a + n_frames_b < \
                2 * \
                self.__class__.NFRAMES_LOWERBOUND + \
                n_frames_interval:
            self.logger.error('No potential to compromise.')
            raise ValueError
        self.logger.warning('Compromise keeping short: repeat' +
                            f' motion_{short_key} to make it longer.')
        n_frames_short = cut_dict[short_key][1]
        n_frames_diff = \
            self.__class__.NFRAMES_LOWERBOUND - \
            n_frames_short
        # if a is too short
        # repeat a's tail and cut b's head
        if short_key == 'a':
            repeat_seq = motion_seq_a[-1:, :].repeat(
                n_frames_diff, axis=0)
            motion_seq_a = np.concatenate(
                (motion_seq_a, repeat_seq), axis=0)
            motion_seq_b = motion_seq_b[n_frames_diff:, :]
            n_frames_cut_a = 0
            n_frames_cut_b = n_frames_interval
        # if b is too short
        # repeat b's head and cut a's tail
        else:
            repeat_seq = motion_seq_b[:1, :].repeat(
                n_frames_diff, axis=0)
            motion_seq_b = np.concatenate(
                (repeat_seq, motion_seq_b), axis=0)
            motion_seq_a = motion_seq_a[:-n_frames_diff, :]
            n_frames_cut_a = n_frames_interval
            n_frames_cut_b = 0
        self.logger.warning('Motion\'s n_frames' +
                            ' before compromise: ' +
                            f'motion_a={n_frames_a}, ' +
                            f'motion_b={n_frames_b}\n' + 'Motion\'s n_frames' +
                            ' after compromise: ' +
                            f'motion_a={motion_seq_a.shape[0]}, ' +
                            f'motion_b={motion_seq_b.shape[0]}')
        return self._cut_and_blend(
            motion_seq_a=motion_seq_a,
            motion_seq_b=motion_seq_b,
            n_frames_cut_a=n_frames_cut_a,
            n_frames_cut_b=n_frames_cut_b,
            transit_time=transit_time)

    def _cut_and_blend(self,
                       motion_seq_a: np.ndarray,
                       motion_seq_b: np.ndarray,
                       n_frames_cut_a: int,
                       n_frames_cut_b: int,
                       transit_time: float) -> np.ndarray:
        # preventing error caused by motion_rotmat_a[:-0, ...]
        if n_frames_cut_a > 0:
            motion_seq_a = motion_seq_a[:-n_frames_cut_a, :]
        motion_seq_b = motion_seq_b[n_frames_cut_b:, :]
        blend_result = quintic_polynomial_1d(
            motion_seq_a=motion_seq_a,
            motion_seq_b=motion_seq_b,
            time=transit_time,
            fps=self.fps)
        return blend_result


def quintic_polynomial_1d(motion_seq_a: np.ndarray,
                          motion_seq_b: np.ndarray,
                          time: float = 1.0,
                          fps: int = 30) -> np.ndarray:
    """Generate transit motion between motion_seq_a and motion_seq_b,
    concatenate them and return the total motion.

    Args:
        motion_seq_a (np.ndarray):
            The first motion array, in shape [n_frames_a, data_dim].
            Cannot be shorter than 9 frames.
        motion_seq_b (np.ndarray):
            The second motion array, in shape [n_frames_b, data_dim].
            Cannot be shorter than 9 frames.
        time (float, optional):
            Time length of the transit motion inserted
            between a and b, in seconds.
            Defaults to 1.0.
        fps (int, optional):
            Number of frames per second. Defaults to 30.

    Returns:
        np.ndarray:
            Total motion(a+transit+b),
            in shape [n_frames_total, data_dim],
            where n_frames_total = n_frames_a + time*fps + n_frames_b.
    """
    motion_seq_a = motion_seq_a.T
    motion_seq_b = motion_seq_b.T
    n_frames_a = motion_seq_a.shape[1]
    n_frames_b = motion_seq_b.shape[1]
    # n_frames_transit is the real transit frame number
    n_frames_transit = int(time * fps)
    # n_frames_reference is the frame number of intervals array
    # the first and last frame in intervals array
    # will be dropped
    n_frames_reference = n_frames_transit + 2
    time_reference = n_frames_transit / fps

    tf = time_reference / 4
    box_length = 4
    dt = 1 / fps
    x0 = motion_seq_a[:, -1]
    xt = motion_seq_b[:, 0]

    v0s = []
    for i in range(box_length):
        v0s.append((motion_seq_a[:, -box_length + i] -
                    motion_seq_a[:, -box_length * 2 + i]) / (box_length * dt))
    v0s = np.array(v0s)
    v0 = v0s.mean(0)

    a0s = []
    a_box_length = box_length // 2
    for i in range(a_box_length):
        a0s.append((v0s[-a_box_length + i] - v0s[-a_box_length * 2 + i]) /
                   (a_box_length * dt))
    a0s = np.array(a0s)
    a0 = a0s.mean(0)

    vts = []
    for i in range(box_length):
        vts.append((motion_seq_b[:, box_length * 2 - i] -
                    motion_seq_b[:, box_length - i]) / (box_length * dt))
    vts = np.array(vts)
    vt = vts.mean(0)

    ats = []
    a_box_length = box_length // 2
    for i in range(a_box_length):
        ats.append((vts[a_box_length + i] - vts[i]) / (a_box_length * dt))
    ats = np.array(ats)
    at = ats.mean(0)

    tf2 = tf * tf
    tf3 = tf2 * tf
    tf4 = tf3 * tf
    tf5 = tf4 * tf

    F = x0
    E = v0
    D = 0.5 * a0

    L4 = xt - D * tf2 - E * tf - F
    L5 = vt - 2 * D * tf - E
    L6 = at - 2 * D
    factor = np.expand_dims(np.array([L4, L5, L6]).T, -1)

    matrix = np.array([[tf5, tf4, tf3], [5 * tf4, 4 * tf3, 3 * tf2],
                       [20 * tf3, 12 * tf2, 6 * tf]])

    matrix = np.repeat(matrix[None, :, :], len(motion_seq_a), axis=0)
    matrix_inv = np.linalg.inv(matrix)
    result = np.matmul(matrix_inv, factor)

    A, B, C = result[:, 0:1, 0], result[:, 1:2, 0], result[:, 2:3, 0]

    xs = np.expand_dims(np.linspace(0, tf, n_frames_reference), 0)

    xs2 = xs * xs
    xs3 = xs2 * xs
    xs4 = xs3 * xs
    xs5 = xs4 * xs
    intervals = A * xs5 + B * xs4 + C * xs3 + \
        D[:, None] * xs2 + \
        E[:, None] * xs + F[:, None]

    x = np.concatenate([motion_seq_a, intervals[:, 1:-1], motion_seq_b], 1)

    n_framesB = motion_seq_b.shape[1]

    # filtering
    b, a = sig.butter(8, 0.075)
    y = sig.filtfilt(b, a, x, axis=1)
    length = 10
    blend_point = (y.shape[1] - n_framesB) / y.shape[1]
    A_start = length * 2 * blend_point - length
    mask_x = np.linspace(-length, length, y.shape[1]) - A_start
    mask = np.exp(-(mask_x**2))
    result = mask * y + (1 - mask) * x
    result = result.reshape(-1,
                            n_frames_a + n_frames_b + n_frames_transit)
    result = result.T
    return result
