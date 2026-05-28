#!/usr/bin/env python3
"""
Web visualizer for the yifu_banshen ESP32 reader.

订阅 2.py 发布的以下话题：
- /sensor/capacitance
- /sensor/acceleration_raw
- /sensor/gyroscope_raw
- /sensor/magnetic_raw
"""

import copy
import threading
import time

import rclpy
from flask import Flask, render_template, request
from flask_socketio import SocketIO
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


IMU_CHANNELS = 8
CAP_CHANNELS = 10
AXES_PER_CHANNEL = 3
UI_STREAM_INTERVALS = {
    'group_data': 1.0 / 24.0,
    'cap_data': 1.0 / 18.0,
}


app = Flask(__name__)
app.config['SECRET_KEY'] = 'yifu_2py_visualizer'
socketio = SocketIO(app, cors_allowed_origins='*', manage_session=False)


class IMUWebNode(Node):
    def __init__(self, socketio_instance):
        super().__init__('imu_2py_web_node')
        self.socketio = socketio_instance
        self._ui_stream_lock = threading.Lock()
        self._pending_ui_events = {
            'group_data': {},
            'cap_data': {},
        }
        self._last_ui_emit_at = {
            'group_data': {},
            'cap_data': {},
        }
        self._latest_ui_events = {
            'group_data': {},
            'cap_data': {},
        }

        self.cap_sub = self.create_subscription(
            Float32MultiArray,
            'sensor/capacitance',
            self.capacitance_callback,
            10,
        )
        self.acc_sub = self.create_subscription(
            Float32MultiArray,
            'sensor/acceleration_raw',
            lambda msg: self.group_callback(msg, 'acc'),
            10,
        )
        self.gyro_sub = self.create_subscription(
            Float32MultiArray,
            'sensor/gyroscope_raw',
            lambda msg: self.group_callback(msg, 'gyro'),
            10,
        )
        self.mag_sub = self.create_subscription(
            Float32MultiArray,
            'sensor/magnetic_raw',
            lambda msg: self.group_callback(msg, 'mag'),
            10,
        )

        self.ui_emit_timer = self.create_timer(0.02, self.flush_ui_streams)
        self.get_logger().info('2.py web visualizer node started')

    def cache_ui_event(self, event_name: str, stream_key: str, payload: dict):
        with self._ui_stream_lock:
            self._pending_ui_events[event_name][stream_key] = payload
            self._latest_ui_events[event_name][stream_key] = copy.deepcopy(payload)

    def get_latest_ui_events(self):
        with self._ui_stream_lock:
            return {
                event_name: {
                    stream_key: copy.deepcopy(payload)
                    for stream_key, payload in stream_payloads.items()
                }
                for event_name, stream_payloads in self._latest_ui_events.items()
            }

    def flush_ui_streams(self):
        now = time.time()
        events_to_emit = []

        with self._ui_stream_lock:
            for event_name, interval in UI_STREAM_INTERVALS.items():
                pending_streams = self._pending_ui_events[event_name]
                last_emit_map = self._last_ui_emit_at[event_name]

                for stream_key, payload in list(pending_streams.items()):
                    last_emit_at = last_emit_map.get(stream_key, 0.0)
                    if now - last_emit_at < interval:
                        continue
                    pending_streams.pop(stream_key, None)
                    last_emit_map[stream_key] = now
                    events_to_emit.append((event_name, payload))

        for event_name, payload in events_to_emit:
            self.socketio.emit(event_name, payload)

    def group_callback(self, msg, kind: str):
        values = list(msg.data)
        expected_len = IMU_CHANNELS * AXES_PER_CHANNEL
        if len(values) < expected_len:
            return

        channels = []
        for i in range(IMU_CHANNELS):
            base = i * AXES_PER_CHANNEL
            channels.append([
                float(values[base]),
                float(values[base + 1]),
                float(values[base + 2]),
            ])

        self.cache_ui_event('group_data', kind, {
            'kind': kind,
            'timestamp': time.time(),
            'channels': channels,
        })

    def capacitance_callback(self, msg: Float32MultiArray):
        if len(msg.data) < CAP_CHANNELS:
            return

        self.cache_ui_event('cap_data', 'capacitance', {
            'timestamp': time.time(),
            'values': [float(v) for v in list(msg.data[:CAP_CHANNELS])],
        })


@app.route('/')
def index():
    return render_template('index.html', imu_channels=IMU_CHANNELS, cap_channels=CAP_CHANNELS)


@socketio.on('connect')
def handle_connect(auth=None):
    socketio.emit('server_meta', {
        'imuChannels': IMU_CHANNELS,
        'capChannels': CAP_CHANNELS,
    }, to=request.sid)

    imu_node = app.config.get('IMU_NODE')
    if imu_node is None:
        return

    latest_events = imu_node.get_latest_ui_events()
    for payload in latest_events.get('group_data', {}).values():
        socketio.emit('group_data', payload, to=request.sid)
    for payload in latest_events.get('cap_data', {}).values():
        socketio.emit('cap_data', payload, to=request.sid)


def ros_spin(node: Node):
    rclpy.spin(node)


def main():
    rclpy.init()
    imu_node = IMUWebNode(socketio)
    app.config['IMU_NODE'] = imu_node

    ros_thread = threading.Thread(target=ros_spin, args=(imu_node,), daemon=True)
    ros_thread.start()

    print('\n' + '=' * 60)
    print('yifu_banshen Web Visualizer')
    print('=' * 60)
    print('Open your browser and navigate to: http://localhost:5016')
    print('=' * 60 + '\n')

    try:
        import sys

        run_kwargs = {
            'host': '0.0.0.0',
            'port': 5016,
            'debug': False,
        }
        if (not sys.stdin) or (not sys.stdin.isatty()):
            run_kwargs['allow_unsafe_werkzeug'] = True

        socketio.run(app, **run_kwargs)
    except KeyboardInterrupt:
        pass
    finally:
        imu_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
