# %%
from ahrs.filters import Madgwick
from ahrs.common.orientation import q_prod, q_conj
import numpy.linalg as la
from sympy import lambdify
from sympy.algebras.quaternion import Quaternion
from sympy import symbols
import ahrs
import numpy as np
import matplotlib.pyplot as plt
import sympy

# Basic parameters
NUM_SAMPLES = 1000
SAMPLING_FREQUENCY = 100.0
THRESHOLD = 0.5

# Geomagnetic values
wmm = ahrs.utils.WMM(latitude=ahrs.MUNICH_LATITUDE,
                     longitude=ahrs.MUNICH_LONGITUDE, height=ahrs.MUNICH_HEIGHT)
REFERENCE_MAGNETIC_VECTOR = wmm.geodetic_vector
MAG_NOISE_STD_DEVIATION = np.linalg.norm(REFERENCE_MAGNETIC_VECTOR) * 0.005

# Gravitational values
NORMAL_GRAVITY = ahrs.utils.WGS().normal_gravity(
    ahrs.MUNICH_LATITUDE, ahrs.MUNICH_HEIGHT*1000)
REFERENCE_GRAVITY_VECTOR = np.array([0.0, 0.0, NORMAL_GRAVITY])
ACC_NOISE_STD_DEVIATION = np.linalg.norm(REFERENCE_GRAVITY_VECTOR) * 0.01

NOISE_SIGMA = abs(np.random.standard_normal(3) * 0.1) * ahrs.RAD2DEG


def __gaussian_filter(in_array: np.ndarray, size: int = 10, sigma: float = 1.0) -> np.ndarray:
    """
    Gaussian filter over an array

    This implementation tries to mimic the behavior of Scipy's function
    ``gaussian_filter1d``, in order to avoid the dependency on Scipy.

    Parameters
    ----------
    in_array : np.ndarray
        Input array to be filtered.
    size : int, default: 10
        Size of Kernel used over the input array.
    sigma : float, default: 1.0
        Standard deviation of Gaussian Kernel.

    Returns
    -------
    y : np.ndarray
        Filtered array.
    """
    x = np.linspace(-sigma*4, sigma*4, size)
    phi_x = np.exp(-0.5*x**2/sigma**2)
    phi_x /= phi_x.sum()
    if in_array.ndim < 2:
        return np.correlate(in_array, phi_x, mode='same')
    return np.array([np.correlate(col, phi_x, mode='same') for col in in_array.T]).T


def random_angpos(num_samples: int = 500, max_positions: int = 4, num_axes: int = 3, span: list = None, **kwargs) -> np.ndarray:
    """
    Random angular positions

    Create an array of synthetic random angular positions with reference to a
    local sensor coordinate frame.

    These angular positions are "simulated" by creating a random number of
    positions per axis, extend them for several samples, and then smoothing
    them with a gaussian filter.

    This creates smooth transitions between the different angular positions.

    Parameters
    ----------
    num_samples : int, default: 500
        Number of samples to generate. Set it to minimum 50, so that the
        gaussian filter can be applied.
    max_positions : int, default: 4
        Maximum number of rotations per axis.
    num_axes : int, default: 3
        Number of axes required.
    span : list or tuple, default: [-pi/2, pi/2]
        Span (minimum to maximum) of the random values.

    Returns
    -------
    angular_positions: np.ndarray
        M-by-3 Array of angular positions.
    """
    span = span if isinstance(span, (list, tuple)) else [-0.5*np.pi, 0.5*np.pi]
    all_angs = [np.random.uniform(span[0], span[1], np.random.randint(
        1, max_positions)) for _ in np.arange(num_axes)]
    angular_positions = np.zeros((num_samples, num_axes))
    for j, angs in enumerate(all_angs):
        # Create angular positions per axis
        num_angs = len(angs)
        idxs = np.sort(np.random.randint(
            0, num_samples, 2*num_angs)).reshape((num_angs, 2))
        for i, idx in enumerate(idxs):
            # Extend each angular position for several samples
            angular_positions[idx[0]:idx[1], j] = angs[i]
    smoothed_angular_positions = __gaussian_filter(angular_positions, size=kwargs.pop(
        'gauss_size', 50 if num_samples > 50 else num_samples//5), sigma=5)
    return smoothed_angular_positions


class Sensors:
    """
    Generate synthetic sensor data of a hypothetical strapdown inertial
    navigation system.

    It generates data of a 9-DOF IMU (3-axes gyroscope, 3-axes accelerometer,
    and 3-axes magnetometer) from a given array of orientations as quaternions.

    The accelerometer data is given as m/s^2, the gyroscope data as deg/s, and
    the magnetometer data as nT.

    If no quaternions are provided, it generates random angular positions and
    computes the corresponding quaternions.

    The sensor data can be accessed as attributes of the object. For example,
    the gyroscope data can be accessed as ``sensors.gyroscopes``.

    Parameters
    ----------
    quaternions : ahrs.QuaternionArray, default: None
        Array of orientations as quaternions.
    num_samples : int, default: 500
        Number of samples to generate.
    freq : float, default: 100.0
        Sampling frequency, in Hz, of the data.
    in_degrees : bool, default: True
        If True, the gyroscope data is generated in degrees per second.
        Otherwise in radians per second.
    normalized_mag : bool, default: False
        If True, the magnetometer data is normalized to unit norm.
    reference_gravitational_vector : np.ndarray, default: None
        Reference gravitational vector. If None, it uses the default reference
        gravitational vector of ``ahrs.utils.WGS()``.
    reference_magnetic_vector : np.ndarray, default: None
        Reference magnetic vector. If None, it uses the default reference
        magnetic vector of ``ahrs.utils.WMM()``.
    gyr_noise : float
        Standard deviation of the gyroscope noise. If None given, it is
        generated from a normal distribution with zero mean. It is then scaled
        to be in the same units as the gyroscope data.
    acc_noise : float
        Standard deviation of the accelerometer noise. If None given, it is
        generated from a normal distribution with zero mean. It is then scaled
        to be in the same units as the accelerometer data.
    mag_noise : float
        Standard deviation of the magnetometer noise. If None given, it is
        generated from a normal distribution with zero mean. It is then scaled
        to be in the same units as the magnetometer data.

    Examples
    --------
    >>> sensors = Sensors(num_samples=1000)
    >>> sensors.gyroscopes.shape
    (1000, 3)
    >>> sensors.accelerometers.shape
    (1000, 3)
    >>> sensors.magnetometers.shape
    (1000, 3)
    >>> sensors.quaternions.shape
    (1000, 4)

    """

    def __init__(self, quaternions: ahrs.QuaternionArray = None, num_samples: int = 500, freq: float = SAMPLING_FREQUENCY, **kwargs):
        self.frequency = freq
        self.in_degrees = kwargs.get('in_degrees', True)
        self.normalized_mag = kwargs.get('normalized_mag', False)

        # Reference earth frames
        self.reference_gravitational_vector = kwargs.get(
            'reference_gravitational_vector', REFERENCE_GRAVITY_VECTOR)
        self.reference_magnetic_vector = kwargs.get(
            'reference_magnetic_vector', REFERENCE_MAGNETIC_VECTOR)

        # Spectral noise density
        self.gyr_noise = kwargs.get('gyr_noise', NOISE_SIGMA)
        self.acc_noise = kwargs.get('acc_noise', ACC_NOISE_STD_DEVIATION)
        self.mag_noise = kwargs.get('mag_noise', MAG_NOISE_STD_DEVIATION)

        # Orientations as quaternions
        if quaternions is None:
            self.num_samples = num_samples
            # Generate orientations (angular positions)
            self.ang_pos = random_angpos(
                num_samples=self.num_samples, span=(-np.pi, np.pi), max_positions=20)
            self.quaternions = ahrs.QuaternionArray(rpy=self.ang_pos)
            # Estimate angular velocities
            self.ang_vel = self.angular_velocities(
                self.ang_pos, self.frequency)
        else:
            # Define angular positions and velocities
            self.quaternions = ahrs.QuaternionArray(quaternions)
            self.num_samples = self.quaternions.shape[0]
            self.ang_pos = self.quaternions.to_angles()
            self.ang_vel = np.r_[
                np.zeros((1, 3)), self.quaternions.angular_velocities(1/self.frequency)]

        # Rotation Matrices
        self.rotations = self.quaternions.to_DCM()

        # Set empty arrays
        self.gyroscopes = None
        self.accelerometers = np.zeros((self.num_samples, 3))
        self.magnetometers = np.zeros((self.num_samples, 3))
        self.magnetometers_nd = np.zeros((self.num_samples, 3))
        self.magnetometers_enu = np.zeros((self.num_samples, 3))

        # Generate MARG data
        self.generate(self.rotations)

    def angular_velocities(self, angular_positions: np.ndarray, frequency: float) -> np.ndarray:
        """Compute angular velocities"""
        Qts = angular_positions if isinstance(
            angular_positions, ahrs.QuaternionArray) else ahrs.QuaternionArray(rpy=angular_positions)
        angvels = Qts.angular_velocities(1/frequency)
        return np.vstack((angvels[0], angvels))

    def generate(self, rotations: np.ndarray) -> None:
        """Compute synthetic data"""
        # Angular velocities measured in the local frame
        self.gyroscopes = np.copy(self.ang_vel) * ahrs.RAD2DEG
        # Add gyro biases: uniform random constant biases within 1/200th of the full range of the gyroscopes
        self.biases_gyroscopes = (np.random.default_rng().random(
            3)-0.5) * np.ptp(self.gyroscopes)/200
        self.gyroscopes += self.biases_gyroscopes

        # Accelerometers and magnetometers are measured w.r.t. global frame (inverse of the local frame)
        self.reference_magnetic_vector_nd = np.array(
            [np.cos(wmm.I * ahrs.DEG2RAD), 0.0, np.sin(wmm.I * ahrs.DEG2RAD)])
        self.reference_magnetic_vector_enu = ahrs.common.frames.ned2enu(
            self.reference_magnetic_vector)
        for i in np.arange(self.num_samples):
            self.accelerometers[i] = rotations[i].T @ self.reference_gravitational_vector
            self.magnetometers[i] = rotations[i].T @ self.reference_magnetic_vector
            self.magnetometers_nd[i] = rotations[i].T @ self.reference_magnetic_vector_nd
            self.magnetometers_enu[i] = rotations[i].T @ self.reference_magnetic_vector_enu

        # # Add centrifugal force based on cross product of angular velocities
        # self.accelerometers -= __centrifugal_force(self.ang_vel)

        # Add noise
        if self.mag_noise < np.ptp(self.magnetometers):
            self.mag_noise = np.linalg.norm(REFERENCE_MAGNETIC_VECTOR) * 0.005
        self.gyroscopes += np.random.standard_normal(
            (self.num_samples, 3)) * self.gyr_noise
        self.accelerometers += np.random.standard_normal(
            (self.num_samples, 3)) * self.acc_noise
        self.magnetometers += np.random.standard_normal(
            (self.num_samples, 3)) * self.mag_noise
        self.magnetometers_nd += np.random.standard_normal(
            (self.num_samples, 3)) * self.mag_noise
        self.magnetometers_enu += np.random.standard_normal(
            (self.num_samples, 3)) * self.mag_noise

        if not self.in_degrees:
            self.gyroscopes *= ahrs.DEG2RAD
            self.biases_gyroscopes *= ahrs.DEG2RAD
        if self.normalized_mag:
            self.magnetometers /= np.linalg.norm(
                self.magnetometers, axis=1, keepdims=True)
            self.magnetometers_nd /= np.linalg.norm(
                self.magnetometers_nd, axis=1, keepdims=True)
            self.magnetometers_enu /= np.linalg.norm(
                self.magnetometers_enu, axis=1, keepdims=True)


SENSOR_DATA = Sensors(num_samples=1000, in_degrees=False)
REFERENCE_QUATERNIONS = SENSOR_DATA.quaternions
REFERENCE_ROTATIONS = SENSOR_DATA.rotations

# %%

qx, qy, qz, qw = symbols('q_x q_y q_z q_w')
dx, dy, dz, dw = symbols('d_x d_y d_z d_w')
sx, sy, sz, sw = symbols('s_x s_y s_z s_w')

q = Quaternion(qw, qx, qy, qz)
ref_d = Quaternion(0, dx, dy, dz)
sensor = Quaternion(0, sx, sy, sz)

min_obj = (q.conjugate() * ref_d * q - sensor)
min_obj = min_obj.expand().simplify()
min_objM = min_obj.to_Matrix()
min_obj_lamb = lambdify((qw, qx, qy, qz, dx, dy, dz, sx, sy, sz), min_objM)

obj_J = min_objM.jacobian([qw, qx, qy, qz])
obj_J_lamb = lambdify((qw, qx, qy, qz, dx, dy, dz), obj_J)
obj_J_lamb(*np.random.rand(7))

sympy.print_latex(obj_J.T)

g = Quaternion(0, 0, 0, 1)
ax, ay, az = symbols('a_x a_y a_z')
a = Quaternion(0, ax, ay, az)

min_obj = q.conjugate() * g * q - a
min_obj = min_obj.expand().simplify()
min_objM = min_obj.to_Matrix()
sympy.print_latex(min_objM)
obj_J = min_objM.jacobian([qw, qx, qy, qz])
sympy.print_latex(obj_J.T)

bx, bz = symbols('b_x b_z')
b = Quaternion(0, bx, 0, bz)
mx, my, mz = symbols('m_x m_y m_z')
m = Quaternion(0, mx, my, mz)

min_obj = q.conjugate() * b * q - m
min_obj = min_obj.expand().simplify()
min_objM = min_obj.to_Matrix()
sympy.print_latex(min_objM)

obj_J = min_objM.jacobian([qw, qx, qy, qz])
sympy.print_latex(obj_J.T)
obj_J

# %%


def objfun(q, d, s): return min_obj_lamb(*q, *d, *s)
def ojbJ(q, d): return obj_J_lamb(*q, *d)


q = ahrs.Quaternion().to_array()
gyroscopes = np.copy(SENSOR_DATA.gyroscopes)
accelerometers = np.copy(SENSOR_DATA.accelerometers)
mags = SENSOR_DATA.magnetometers

Qs = np.zeros((len(gyroscopes), 4))
beta = 0.1
dt = 1/SAMPLING_FREQUENCY
for i in range(len(gyroscopes)):
    acc = ahrs.Quaternion(accelerometers[i]).to_array()[1:]
    g = ahrs.Quaternion([0, 0, 0, 1]).to_array()[1:]
    m = ahrs.Quaternion(mags[i]).to_array()[1:]

    qDot = 0.5 * q_prod(q, [0, *gyroscopes[i]])

    h = q_prod(q, q_prod([0, *m], q_conj(q)))
    bx = np.linalg.norm([h[1], h[2]])
    bz = h[3]
    b = np.array([bx, 0, bz])

    f_mag = objfun(q, b, m)[1:]
    J_mag = ojbJ(q, b)[1:]
    f_acc = objfun(q, g, acc)[1:]
    J_acc = ojbJ(q, g)[1:]
    f = np.vstack((f_mag, f_acc))
    J = np.vstack((J_mag, J_acc))

    assert la.norm(f) > 1e-6
    grad = J.T @ f
    grad /= np.linalg.norm(grad)
    grad = grad.reshape(4)
    qDot -= beta * grad

    q += qDot * dt
    q /= np.linalg.norm(q)
    Qs[i] = q

Qs = ahrs.QuaternionArray(Qs)

mean_err = np.nanmean(ahrs.utils.metrics.qad(
    REFERENCE_QUATERNIONS, Qs))
assert mean_err < THRESHOLD, mean_err

madgwick = Madgwick()
Q_ahrs = np.tile([1., 0., 0., 0.], (len(gyroscopes), 1)
                 )  # Allocate for quaternions
for t in range(1, len(gyroscopes)):
    Q_ahrs[t] = madgwick.updateMARG(
        Q_ahrs[t-1], gyr=gyroscopes[t], acc=accelerometers[t],
        mag=mags[t], dt=1/SAMPLING_FREQUENCY)

ahrs_q = ahrs.QuaternionArray(Q_ahrs)

mean_err_ahrs = np.nanmean(ahrs.utils.metrics.qad(
    REFERENCE_QUATERNIONS, ahrs_q))
print(mean_err_ahrs, mean_err)  # outperforms

plt.figure(figsize=(20, 20))
plt.plot(Qs.to_angles(), label='Qs', linestyle='--')
plt.plot(REFERENCE_QUATERNIONS.to_angles(), label='REF')
plt.plot(ahrs_q.to_angles(), label='AHRS', linestyle='-.')
plt.legend()
plt.show()

# %%
