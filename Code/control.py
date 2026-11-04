"""The position controller: position error in, velocity command out.

In Project 2a this file held a cascade:

    position PID -> velocity PID -> attitude -> body rates -> motor mixing

On the Crazyflie the firmware does everything after the first stage, so only
that first stage is left: three PIDs (x, y, z), nine gains. The `pid` class is
the one from 2a, with the time step passed in on every call instead of fixed.
"""
import math

import numpy as np


class pid:
    def __init__(self, kp, ki, kd, filter_tau, dt, minVal=-1.0, maxVal=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.filter_tau = filter_tau    # low-pass on the derivative, seconds
        self.dt = dt                    # default time step, seconds
        self.minVal = minVal
        self.maxVal = maxVal
        self.reset()

    def reset(self):
        self.prev_err = None
        self.prev_filter_val = 0.0
        self.prev_integral = 0.0

    def step(self, dsOrErr, current_state=None, dt=None):
        dt = self.dt if dt is None or dt <= 0 else dt

        # Error
        err = dsOrErr if current_state is None else dsOrErr - current_state

        # Derivative of the error, low-pass filtered. No derivative on the
        # very first call: there is no previous error to difference against.
        err_der = 0.0 if self.prev_err is None else (err - self.prev_err) / dt
        alpha = min(1.0, dt / self.filter_tau)
        err_der_filtered = err_der * alpha + self.prev_filter_val * (1 - alpha)

        # Integral, clamped so it cannot wind up
        err_integral = float(np.clip(err * dt + self.prev_integral,
                                     self.minVal, self.maxVal))

        out = self.kp * err + self.kd * err_der_filtered + self.ki * err_integral
        if math.isnan(out) or math.isinf(out):
            raise Exception('PID output is not finite (err=%r)' % (err,))

        self.prev_err = err
        self.prev_filter_val = err_der_filtered
        self.prev_integral = err_integral
        return float(np.clip(out, self.minVal, self.maxVal))


class PositionController:
    """v_cmd = feed-forward velocity + PID(target position - position).

    All vectors are in one world frame (whichever the caller uses; the policy
    uses scene metres). The output is limited to `max_speed` in the horizontal
    plane and `max_climb` vertically.
    """

    ###########################################################################
    # THE NINE GAINS. kp is 1/seconds: a 0.1 m error asks for kp*0.1 m/s.
    #            kp    ki    kd
    GAINS = {'x': (1.5, 0.0, 0.05),
             'y': (1.5, 0.0, 0.05),
             'z': (1.5, 0.0, 0.05)}
    ###########################################################################

    def __init__(self, dt=1.0 / 15.0, max_speed=0.6, max_climb=0.4,
                 filter_tau=0.15, gains=None):
        g = dict(self.GAINS if gains is None else gains)
        self.max_speed = float(max_speed)
        self.max_climb = float(max_climb)
        # Each PID is allowed the full speed; the sum is limited in step().
        self.x_pid = pid(*g['x'], filter_tau, dt, -max_speed, max_speed)
        self.y_pid = pid(*g['y'], filter_tau, dt, -max_speed, max_speed)
        self.z_pid = pid(*g['z'], filter_tau, dt, -max_climb, max_climb)

    def reset(self):
        for c in (self.x_pid, self.y_pid, self.z_pid):
            c.reset()

    def step(self, position, target, velocity_ff=(0.0, 0.0, 0.0), dt=None):
        """One control tick. Returns the world-frame velocity command."""
        p = np.asarray(position, dtype=float)
        w = np.asarray(target, dtype=float)
        ff = np.asarray(velocity_ff, dtype=float)
        v = np.array([ff[0] + self.x_pid.step(w[0], p[0], dt),
                      ff[1] + self.y_pid.step(w[1], p[1], dt),
                      ff[2] + self.z_pid.step(w[2], p[2], dt)])
        # Limit the horizontal SPEED (the norm), not each axis, so the
        # direction of the command is kept.
        h = float(math.hypot(v[0], v[1]))
        if h > self.max_speed:
            v[:2] *= self.max_speed / h
        v[2] = float(np.clip(v[2], -self.max_climb, self.max_climb))
        return v


# =============================================================================
# Added for P3: yaw control and the 1-D quintic move. Nothing above is changed.
# =============================================================================
def wrap_pi(a):
    """Angle -> (-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def heading_of(vec):
    """Heading (rad, counter-clockwise from +x) of the horizontal part of a vector."""
    return math.atan2(float(vec[1]), float(vec[0]))


class YawController:
    """yaw_rate = clip(kp * wrap(target - yaw) + rate_ff, +-max_rate).

    The drone only takes a yaw RATE, so this turns a heading error into one.
    kp is 1/s: a 0.2 rad error asks for kp*0.2 rad/s. The runtime caps any
    yaw rate at 90 deg/s; max_rate here is our own, lower cap.
    """

    def __init__(self, kp=2.0, max_rate=math.radians(45.0)):
        self.kp = float(kp)
        self.max_rate = float(max_rate)

    def error(self, yaw, target_yaw):
        return wrap_pi(float(target_yaw) - float(yaw))

    def step(self, yaw, target_yaw, rate_ff=0.0, max_rate=None):
        cap = self.max_rate if max_rate is None else float(max_rate)
        w = self.kp * self.error(yaw, target_yaw) + float(rate_ff)
        return float(np.clip(w, -cap, cap))

    def arrived(self, yaw, target_yaw, tol=math.radians(2.0)):
        return abs(self.error(yaw, target_yaw)) < tol


class Quintic:
    """Straight-line move p0 -> p1 that starts and ends at rest.

        p(t) = p0 + (p1 - p0) s(tau),  s = 10 tau^3 - 15 tau^4 + 6 tau^5,  tau = t / T

    Peak speed is 1.875 L / T. The speed rule we chose: v_peak = min(v_gain * L, v_cap),
    so every move shorter than v_cap / v_gain takes the same time, 1.875 / v_gain,
    and longer ones fly at v_cap. The vertical part is kept under max_climb.
    tau is clamped to [0, 1]: after T the sample stays on p1 (the polynomial
    is never evaluated past its end).
    """

    def __init__(self, p0, p1, v_gain=0.75, v_cap=0.8, max_climb=0.6,
                 arrive_m=0.04, late_s=0.5):
        self.p0 = np.asarray(p0, float).reshape(3)
        self.p1 = np.asarray(p1, float).reshape(3)
        self.d = self.p1 - self.p0
        self.L = float(np.linalg.norm(self.d))
        v_peak = min(v_gain * self.L, v_cap)
        if self.L > 1e-9 and v_peak > 0:
            vz = abs(self.d[2]) / self.L * v_peak       # vertical share of the peak speed
            if vz > max_climb:
                v_peak *= max_climb / vz
            self.T = 1.875 * self.L / v_peak
        else:
            self.T = 0.0
        self.arrive_m = float(arrive_m)
        self.late_s = float(late_s)

    def sample(self, t):
        """(position, velocity) at t seconds after the start."""
        if self.T <= 0.0:
            return self.p1.copy(), np.zeros(3)
        tau = min(max(t / self.T, 0.0), 1.0)
        s = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
        ds = (30 * tau ** 2 - 60 * tau ** 3 + 30 * tau ** 4) / self.T
        return self.p0 + self.d * s, self.d * ds

    def done(self, t, position):
        """Time is up and we are within arrive_m of the end, or T + late_s has passed."""
        if t < self.T:
            return False
        close = np.linalg.norm(np.asarray(position, float) - self.p1) < self.arrive_m
        return close or t > self.T + self.late_s
