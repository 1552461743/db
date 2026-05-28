#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROS2 Serial Reader for 8-Channel IMU Sensor Data
增强版：
1. 使用流式缓冲解析
2. 解析设备当前输出的 ASCII 调试文本
3. 支持 MAG/CAP/ACC/GYRO 任意起步顺序
4. 只有 ACC/GYRO/MAG 三帧齐全后，才统一发布一次 8 路 IMU
"""

import re
import struct
import serial
import rclpy

from typing import Optional, List
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Imu


# =========================
# 协议定义
# =========================
FRAME_DEFS = [
    {"type": 0x0D, "len_byte": 0x30, "frame_len": 52,  "name": "MAG"},
    {"type": 0x0A, "len_byte": 0x14, "frame_len": 24,  "name": "CAP"},
    {"type": 0x0B, "len_byte": 0x30, "frame_len": 52,  "name": "ACC"},
    {"type": 0x0C, "len_byte": 0x60, "frame_len": 100, "name": "GYRO"},
]

FRAME_MAP = {(f["type"], f["len_byte"]): f for f in FRAME_DEFS}
SEQ = [(f["type"], f["len_byte"]) for f in FRAME_DEFS]

STREAM_MAX = 8192
IMU_CHANNELS = 8
CAP_CHANNELS = 10


def crc16_modbus(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc & 0xFFFF


class SerialMotionParser:
    def __init__(self):
        self.stream_buf = bytearray()
        self.crc_ok_count = 0
        self.crc_fail_count = 0
        self.frame_ok_count = 0

    def stream_feed(self, data: bytes):
        self.stream_buf.extend(data)
        if len(self.stream_buf) > STREAM_MAX:
            self.stream_buf = self.stream_buf[-STREAM_MAX:]

    def parse_imu_triplets(self, line: str) -> Optional[List[List[float]]]:
        matches = re.findall(
            r'IMU\d+:\((-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)\)',
            line
        )
        if len(matches) != IMU_CHANNELS:
            return None
        return [[float(x), float(y), float(z)] for x, y, z in matches]

    def parse_text_line(self, line: str):
        if line.startswith('[0A 14]'):
            match = re.search(r'CAP\(10\)\s*=\s*\[([^\]]+)\]', line)
            if not match:
                return None
            values = [float(v.strip()) for v in match.group(1).split(',')]
            if len(values) != CAP_CHANNELS:
                return None
            return 'CAP', values

        if line.startswith('[0B 30]'):
            data = self.parse_imu_triplets(line)
            return ('ACC', data) if data is not None else None

        if line.startswith('[0C 60]'):
            data = self.parse_imu_triplets(line)
            return ('GYRO', data) if data is not None else None

        if line.startswith('[0D 30]'):
            data = self.parse_imu_triplets(line)
            return ('MAG', data) if data is not None else None

        return None

    def parse_from_stream_strict_with_next_header_check(self):
        """按行解析设备输出的 ASCII 文本帧。"""
        while True:
            line_end = self.stream_buf.find(b'\n')
            if line_end < 0:
                return None

            raw_line = bytes(self.stream_buf[:line_end + 1])
            del self.stream_buf[:line_end + 1]

            line = raw_line.decode('ascii', errors='ignore').strip()
            parsed = self.parse_text_line(line)
            if parsed is None:
                continue

            self.crc_ok_count += 1
            self.frame_ok_count += 1
            return parsed


class IMU8ChannelReader(Node):
    def __init__(self):
        super().__init__('imu_8channel_reader_group_publish')

        # 参数
        self.declare_parameter('port', '/dev/ttyACM0')
        self.declare_parameter('baudrate', 921600)
        self.declare_parameter('timeout', 0.02)
        self.declare_parameter('command_interval', 0.05)

        port = self.get_parameter('port').value
        baudrate = self.get_parameter('baudrate').value
        timeout = self.get_parameter('timeout').value
        self.command_interval = self.get_parameter('command_interval').value

        # 发布器
        self.imu_pubs = [
            self.create_publisher(Imu, f'imu/channel_{i}', 10)
            for i in range(IMU_CHANNELS)
        ]

        self.capacitance_pub = self.create_publisher(Float32MultiArray, 'sensor/capacitance', 10)
        self.acc_raw_pub = self.create_publisher(Float32MultiArray, 'sensor/acceleration_raw', 10)
        self.gyro_raw_pub = self.create_publisher(Float32MultiArray, 'sensor/gyroscope_raw', 10)
        self.mag_raw_pub = self.create_publisher(Float32MultiArray, 'sensor/magnetic_raw', 10)

        # 串口
        try:
            self.serial = serial.Serial(port, baudrate, timeout=timeout)
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()
            self.send_data_command()
            self.get_logger().info(f'串口已打开: {port}, baudrate={baudrate}')
            self.get_logger().info('已发送 DATA 指令')
        except serial.SerialException as e:
            self.get_logger().error(f'无法打开串口: {e}')
            self.serial = None
            return

        self.parser = SerialMotionParser()

        # 最新已发布缓存
        self.latest_acc = [[0.0, 0.0, 0.0] for _ in range(IMU_CHANNELS)]
        self.latest_gyro = [[0.0, 0.0, 0.0] for _ in range(IMU_CHANNELS)]
        self.latest_mag = [[0.0, 0.0, 0.0] for _ in range(IMU_CHANNELS)]
        self.latest_cap = [0.0 for _ in range(CAP_CHANNELS)]

        # 当前轮临时缓存
        self.pending_acc = None
        self.pending_gyro = None
        self.pending_mag = None
        self.pending_cap = None

        # 当前轮是否收到
        self.round_acc_ready = False
        self.round_gyro_ready = False
        self.round_mag_ready = False

        # 统计
        self.last_stat_time = self.get_clock().now()
        self.last_command_time = self.get_clock().now()
        self.frame_count_1s = 0
        self.publish_count_1s = 0
        self.command_count_1s = 1

        # 定时器
        self.timer = self.create_timer(0.005, self.read_serial_data)

    # =========================
    # 解析函数
    # =========================
    def parse_capacitance(self, full_data: bytes) -> Optional[List[float]]:
        if len(full_data) != 22:
            return None

        payload = full_data[2:]
        values = []
        for i in range(CAP_CHANNELS):
            value = struct.unpack('>H', payload[i * 2:i * 2 + 2])[0]
            values.append(float(value))
        return values

    def parse_acceleration_8ch(self, full_data: bytes) -> Optional[List[List[float]]]:
        if len(full_data) != 50:
            return None

        payload = full_data[2:]
        acc_data = []
        for ch in range(IMU_CHANNELS):
            offset = ch * 6
            acc_x = struct.unpack('>h', payload[offset:offset + 2])[0] / 1000.0
            acc_y = struct.unpack('>h', payload[offset + 2:offset + 4])[0] / 1000.0
            acc_z = struct.unpack('>h', payload[offset + 4:offset + 6])[0] / 1000.0
            acc_data.append([acc_x, acc_y, acc_z])
        return acc_data

    def parse_gyroscope_8ch(self, full_data: bytes) -> Optional[List[List[float]]]:
        if len(full_data) != 98:
            return None

        payload = full_data[2:]
        gyro_data = []
        for ch in range(IMU_CHANNELS):
            offset = ch * 12
            gyro_x = struct.unpack('>i', payload[offset:offset + 4])[0] / 1000.0
            gyro_y = struct.unpack('>i', payload[offset + 4:offset + 8])[0] / 1000.0
            gyro_z = struct.unpack('>i', payload[offset + 8:offset + 12])[0] / 1000.0
            gyro_data.append([gyro_x, gyro_y, gyro_z])
        return gyro_data

    def parse_magnetic_8ch(self, full_data: bytes) -> Optional[List[List[float]]]:
        if len(full_data) != 50:
            return None

        payload = full_data[2:]
        mag_data = []
        for ch in range(IMU_CHANNELS):
            offset = ch * 6
            mag_x = struct.unpack('>h', payload[offset:offset + 2])[0]
            mag_y = struct.unpack('>h', payload[offset + 2:offset + 4])[0]
            mag_z = struct.unpack('>h', payload[offset + 4:offset + 6])[0]
            mag_data.append([float(mag_x), float(mag_y), float(mag_z)])
        return mag_data

    # =========================
    # 发布函数
    # =========================
    def publish_float_array(self, pub, nested_data):
        msg = Float32MultiArray()
        if len(nested_data) == 0:
            msg.data = []
        elif isinstance(nested_data[0], list):
            msg.data = [v for row in nested_data for v in row]
        else:
            msg.data = nested_data
        pub.publish(msg)

    def publish_imu_data(self):
        stamp = self.get_clock().now().to_msg()

        for i in range(IMU_CHANNELS):
            imu_msg = Imu()
            imu_msg.header.stamp = stamp
            imu_msg.header.frame_id = f'imu_{i}'

            imu_msg.linear_acceleration.x = self.latest_acc[i][0]
            imu_msg.linear_acceleration.y = self.latest_acc[i][1]
            imu_msg.linear_acceleration.z = self.latest_acc[i][2]

            imu_msg.angular_velocity.x = self.latest_gyro[i][0]
            imu_msg.angular_velocity.y = self.latest_gyro[i][1]
            imu_msg.angular_velocity.z = self.latest_gyro[i][2]

            self.imu_pubs[i].publish(imu_msg)

    def try_publish_group(self):
        """
        只有当前轮 ACC / GYRO / MAG 都到齐，才统一发布。
        """
        if not (self.round_acc_ready and self.round_gyro_ready and self.round_mag_ready):
            return

        self.latest_acc = self.pending_acc
        self.latest_gyro = self.pending_gyro
        self.latest_mag = self.pending_mag

        if self.pending_cap is not None:
            self.latest_cap = self.pending_cap
            self.publish_float_array(self.capacitance_pub, self.latest_cap)

        self.publish_float_array(self.acc_raw_pub, self.latest_acc)
        self.publish_float_array(self.gyro_raw_pub, self.latest_gyro)
        self.publish_float_array(self.mag_raw_pub, self.latest_mag)
        self.publish_imu_data()

        self.publish_count_1s += 1

        # 清空当前轮状态，等待下一轮
        self.pending_acc = None
        self.pending_gyro = None
        self.pending_mag = None
        self.pending_cap = None

        self.round_acc_ready = False
        self.round_gyro_ready = False
        self.round_mag_ready = False

    # =========================
    # 主读取逻辑
    # =========================
    def send_data_command(self):
        if self.serial is not None and self.serial.is_open:
            self.serial.write(b'DATA')

    def read_serial_data(self):
        if self.serial is None:
            return

        try:
            now = self.get_clock().now()
            if (now - self.last_command_time).nanoseconds >= int(self.command_interval * 1_000_000_000):
                self.send_data_command()
                self.last_command_time = now
                self.command_count_1s += 1

            waiting = self.serial.in_waiting
            if waiting > 0:
                data = self.serial.read(min(waiting, 2048))
                self.parser.stream_feed(data)

            while True:
                parsed_frame = self.parser.parse_from_stream_strict_with_next_header_check()
                if parsed_frame is None:
                    break

                self.frame_count_1s += 1
                frame_type, frame_data = parsed_frame

                if frame_type == 'CAP':
                    self.pending_cap = frame_data

                elif frame_type == 'ACC':
                    self.pending_acc = frame_data
                    self.round_acc_ready = True

                elif frame_type == 'GYRO':
                    self.pending_gyro = frame_data
                    self.round_gyro_ready = True

                elif frame_type == 'MAG':
                    self.pending_mag = frame_data
                    self.round_mag_ready = True

                # 每拿到一帧都尝试一次，只有三帧齐全才真正发布
                self.try_publish_group()

            now = self.get_clock().now()
            if (now - self.last_stat_time).nanoseconds >= 1_000_000_000:
                self.get_logger().info(
                    f'解析帧率: {self.frame_count_1s} frame/s | '
                    f'统一发布率: {self.publish_count_1s} pub/s | '
                    f'DATA发送: {self.command_count_1s}/s | '
                    f'文本帧: {self.parser.frame_ok_count} | '
                    f'缓冲区剩余: {len(self.parser.stream_buf)} bytes'
                )
                self.frame_count_1s = 0
                self.publish_count_1s = 0
                self.command_count_1s = 0
                self.last_stat_time = now

        except serial.SerialException as e:
            self.get_logger().error(f'串口读取错误: {e}')
        except Exception as e:
            self.get_logger().error(f'数据解析错误: {e}')

    def destroy_node(self):
        if hasattr(self, 'serial') and self.serial is not None and self.serial.is_open:
            self.serial.close()
            if rclpy.ok():
                self.get_logger().info('串口已关闭')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = IMU8ChannelReader()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
