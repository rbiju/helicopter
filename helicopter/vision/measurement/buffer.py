import bisect
import threading

import quaternion


class StateBuffer:
    def __init__(self, maxlen=2000):
        self.maxlen = maxlen
        self.times = []
        self.states = []
        self.lock = threading.Lock()

    def add(self, timestamp, state):
        with self.lock:
            self.times.append(timestamp)
            self.states.append(state)
            if len(self.times) > self.maxlen:
                self.times.pop(0)
                self.states.pop(0)

    def get_interpolated_state(self, query_time):
        with self.lock:
            if not self.times or query_time < self.times[0] or query_time > self.times[-1]:
                return None

            idx = bisect.bisect_right(self.times, query_time)

            t0 = self.times[idx - 1]
            t1 = self.times[idx]
            ratio = (query_time - t0) / (t1 - t0)

            state0 = self.states[idx - 1]
            state1 = self.states[idx]

            interp_state = state0 + ratio * (state1 - state0)

            q0 = quaternion.quaternion(*state0[0:4])
            q1 = quaternion.quaternion(*state1[0:4])
            q_interp = quaternion.slerp(q0, q1, t0, t1, query_time)

            interp_state[0:4] = quaternion.as_float_array(q_interp)

            return interp_state
