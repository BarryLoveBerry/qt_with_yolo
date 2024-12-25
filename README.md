# qt_with_yolo

业余时间接的小项目，基于qt，开发智能园区系统，获取rtsp视频流，并将yolo检测结果显示和分析，并根据吊车和工人的IOU输出对应的告警逻辑
实现功能：（gui.py,推理部分在infer.py）
1. 设置RTSP流
2. 为相机选取模型文件
3. 启动视频画面
4. 到相机显示窗口，主要分为左右俩栏，分别包含相机显示画面，检测结果显示和告警检测

![image](https://github.com/BarryLoveBerry/qt_with_yolo/blob/main/qt_interface)

   
