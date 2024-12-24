
import sys
import cv2
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLineEdit, QGridLayout, QGroupBox, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QImage, QColor, QFont
from infer import Infer

categories = ["crane","worker"]


class CameraWindow(QWidget):
    """摄像头画面显示窗口"""
    def __init__(self, rtsp1, rtsp2, pt_file1, pt_file2):
        super().__init__()

        self.setWindowTitle("摄像头画面")
        self.resize(1200, 800)

        # 保存 RTSP 和模型文件路径
        self.rtsp1 = rtsp1
        self.rtsp2 = rtsp2
        self.pt_file1 = pt_file1
        self.pt_file2 = pt_file2

        # 初始化摄像头流
        self.cap1 = cv2.VideoCapture(rtsp1)
        self.cap2 = cv2.VideoCapture(rtsp2)

        # 主布局
        main_layout = QHBoxLayout()

        # 推理模型实例
        self.model_left = Infer(pt_file1)
        self.model_right = Infer(pt_file2)

        # 左侧摄像头 1 布局
        camera1_layout = QVBoxLayout()
        self.camera1_label = QLabel("摄像头 1")
        self.camera1_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera1_label.setStyleSheet("background-color: black; color: white; font-size: 16px; border-radius: 10px;")
        self.camera1_label.setFixedHeight(500)
        self.camera1_info = QLabel(f"模型文件: {pt_file1}")
        self.camera1_info.setStyleSheet("background-color: #f0f0f0; padding: 5px; border-radius: 5px;")
        
        # 检测结果显示
        self.camera1_detection_info = QLabel("检测结果：无目标")
        self.camera1_detection_info.setStyleSheet("background-color: #f0f0f0; padding: 5px; border-radius: 5px;")
        
        # 告警信息
        self.camera1_alert = QLabel("无警报")
        self.camera1_alert.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera1_alert.setStyleSheet("background-color: #4CAF50; color: white; font-size: 14px; font-weight: bold; padding: 5px; border-radius: 5px;")
        
        # 按顺序添加：检测结果 -> 告警信息
        camera1_layout.addWidget(self.camera1_label)
        camera1_layout.addWidget(self.camera1_info)
        camera1_layout.addWidget(self.camera1_detection_info)
        camera1_layout.addWidget(self.camera1_alert)

        # 右侧摄像头 2 布局
        camera2_layout = QVBoxLayout()
        self.camera2_label = QLabel("摄像头 2")
        self.camera2_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera2_label.setStyleSheet("background-color: black; color: white; font-size: 16px; border-radius: 10px;")
        self.camera2_label.setFixedHeight(500)
        self.camera2_info = QLabel(f"模型文件: {pt_file2}")
        self.camera2_info.setStyleSheet("background-color: #f0f0f0; padding: 5px; border-radius: 5px;")
        
        # 检测结果显示
        self.camera2_detection_info = QLabel("检测结果：无目标")
        self.camera2_detection_info.setStyleSheet("background-color: #f0f0f0; padding: 5px; border-radius: 5px;")
        
        # 告警信息
        self.camera2_alert = QLabel("无警报")
        self.camera2_alert.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera2_alert.setStyleSheet("background-color: #4CAF50; color: white; font-size: 14px; font-weight: bold; padding: 5px; border-radius: 5px;")
        
        # 按顺序添加：检测结果 -> 告警信息
        camera2_layout.addWidget(self.camera2_label)
        camera2_layout.addWidget(self.camera2_info)
        camera2_layout.addWidget(self.camera2_detection_info)
        camera2_layout.addWidget(self.camera2_alert)

        # 将左右布局加入主布局
        main_layout.addLayout(camera1_layout)
        main_layout.addLayout(camera2_layout)
        self.setLayout(main_layout)

        # 定时器用于更新摄像头画面
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frames)
        self.timer.start(30)


    def update_frames(self):
        """更新摄像头画面"""
        if self.cap1 and self.cap1.isOpened():
            ret1, frame1 = self.cap1.read()
            if ret1:
                self.model_left(frame1)
                detection_info1 = self.model_left.parse_result()  # 假设有检测函数
                alert1 = self.model_left.get_alert_info()
                annotated_frame = self.model_left.plot_pred()
                self.display_frame(annotated_frame, self.camera1_label)
                # 此处添加检测目标并更新检测信息
                self.update_detection_info(self.camera1_detection_info, detection_info1)
                self.update_alert_info(self.camera1_alert, alert1)

        if self.cap2 and self.cap2.isOpened():
            ret2, frame2 = self.cap2.read()
            if ret2:
                self.model_right(frame2)
                detection_info2 = self.model_right.parse_result()  # 假设有检测函数
                alert2 = self.model_right.get_alert_info()
                annotated_frame = self.model_right.plot_pred()
                self.display_frame(annotated_frame, self.camera2_label)
                # 此处添加检测目标并更新检测信息
                self.update_detection_info(self.camera2_detection_info, detection_info2)
                self.update_alert_info(self.camera2_alert, alert2)

    def update_alert_info(self, alert_label, alert_info):
        """根据告警信息更新告警标签"""
        if alert_info:
            alert_label.setText(f"警报：{alert_info}")  # 可以根据实际情况修改
            alert_label.setStyleSheet("background-color: #F44336; color: white; font-size: 14px; font-weight: bold; padding: 5px; border-radius: 5px;")  # 红色背景表示警报
        else:
            alert_label.setText("无警报")
            alert_label.setStyleSheet("background-color: #4CAF50; color: white; font-size: 14px; font-weight: bold; padding: 5px; border-radius: 5px;")  # 绿色背景表示正常

    def display_frame(self, frame, label):
        """将帧显示在指定的 QLabel 上"""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape
        qimg = QImage(frame.data, width, height, channel * width, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        label.setPixmap(pixmap.scaled(label.width(), label.height(), Qt.AspectRatioMode.KeepAspectRatio))
    
    def update_detection_info(self, label, detection_info):
        """更新检测信息标签"""
        if detection_info:
            label.setText("检测结果：" + ", ".join(detection_info))
        else:
            label.setText("检测结果：无目标")

    def closeEvent(self, event):
        """窗口关闭事件，释放资源"""
        self.release_resources()
        event.accept()

    def release_resources(self):
        """释放资源"""
        if self.cap1:
            self.cap1.release()
        if self.cap2:
            self.cap2.release()


class MainWindow(QMainWindow):
    """主界面"""
    def __init__(self):
        super().__init__()

        self.setWindowTitle("智慧园区多分系统 APP")
        self.resize(900, 600)

        # 保存模型文件路径
        self.pt_file1 = ""
        self.pt_file2 = ""

        # 主布局
        main_layout = QVBoxLayout()

        # RTSP 地址部分
        rtsp_group = QGroupBox("RTSP 摄像头地址")
        rtsp_group.setStyleSheet("background-color: #f4f4f4; border-radius: 10px; padding: 10px;")
        rtsp_layout = QGridLayout()
        rtsp_group.setLayout(rtsp_layout)
        self.rtsp1_input = QLineEdit()
        self.rtsp1_input.setPlaceholderText("输入摄像头 1 的 RTSP 地址")
        self.rtsp2_input = QLineEdit()
        self.rtsp2_input.setPlaceholderText("输入摄像头 2 的 RTSP 地址")
        self.rtsp1_input.setStyleSheet("padding: 10px; font-size: 14px; border-radius: 5px; border: 1px solid #ccc;")
        self.rtsp2_input.setStyleSheet("padding: 10px; font-size: 14px; border-radius: 5px; border: 1px solid #ccc;")
        rtsp_layout.addWidget(QLabel("摄像头 1:"), 0, 0)
        rtsp_layout.addWidget(self.rtsp1_input, 0, 1)
        rtsp_layout.addWidget(QLabel("摄像头 2:"), 1, 0)
        rtsp_layout.addWidget(self.rtsp2_input, 1, 1)

        # 模型文件选择部分
        pt_group = QGroupBox("模型文件选择")
        pt_group.setStyleSheet("background-color: #f4f4f4; border-radius: 10px; padding: 10px;")
        pt_layout = QGridLayout()
        pt_group.setLayout(pt_layout)
        self.pt1_button = QPushButton("选择模型 1")
        self.pt1_button.clicked.connect(self.select_pt1)
        self.pt1_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px; border-radius: 5px; font-size: 14px;")
        self.pt1_label = QLabel("未选择模型")
        self.pt2_button = QPushButton("选择模型 2")
        self.pt2_button.clicked.connect(self.select_pt2)
        self.pt2_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px; border-radius: 5px; font-size: 14px;")
        self.pt2_label = QLabel("未选择模型")
        pt_layout.addWidget(self.pt1_button, 0, 0)
        pt_layout.addWidget(self.pt1_label, 0, 1)
        pt_layout.addWidget(self.pt2_button, 1, 0)
        pt_layout.addWidget(self.pt2_label, 1, 1)

        # 启动按钮
        self.start_button = QPushButton("启动摄像头画面")
        self.start_button.setStyleSheet("background-color: #008CBA; color: white; padding: 15px; border-radius: 10px; font-size: 16px;")
        self.start_button.clicked.connect(self.start_action)

        # 主界面布局
        main_layout.addWidget(rtsp_group)
        main_layout.addWidget(pt_group)
        main_layout.addWidget(self.start_button, alignment=Qt.AlignmentFlag.AlignCenter)

        # 设置主控界面
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def select_pt1(self):
        """选择第一个模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择模型文件 1", "", "PyTorch 模型文件 (*.pt)")
        if file_path:
            self.pt_file1 = file_path
            self.pt1_label.setText(file_path)

    def select_pt2(self):
        """选择第二个模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择模型文件 2", "", "PyTorch 模型文件 (*.pt)")
        if file_path:
            self.pt_file2 = file_path
            self.pt2_label.setText(file_path)

    def start_action(self):
        """启动按钮的逻辑处理"""
        rtsp1 = self.rtsp1_input.text()
        rtsp2 = self.rtsp2_input.text()

        # 检查 RTSP 地址和模型文件是否输入
        if not rtsp1 or not rtsp2:
            self.show_error_message("RTSP 地址不能为空", "请填写两个 RTSP 地址")
            return
        if not self.pt_file1 or not self.pt_file2:
            self.show_error_message("模型文件不能为空", "请为两个摄像头选择模型文件")
            return

        # 打开摄像头显示窗口，并传递模型文件路径
        self.camera_window = CameraWindow(rtsp1, rtsp2, self.pt_file1, self.pt_file2)
        self.camera_window.show()

    def show_error_message(self, title, message):
        """显示错误信息弹窗"""
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.exec()


# 主程序运行
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
