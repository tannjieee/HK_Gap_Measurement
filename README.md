# HIKROBOT MVS SDK Python examples

本工程在 Linux 下直接调用海康机器人 MVS Python SDK，包含命令行抓图、
实时控制 GUI、棋盘格相机内参标定工具和 AprilTag 6D 位姿识别 GUI。

## GEMINI 335L RGB 内参标定

[gemini335l_calibration.py](gemini335l_calibration.py) 订阅 ROS 2 原始彩色图像进行棋盘格标定，
可以与 RViz 同时运行。启动时若相机驱动已经运行，不要重复启动驱动。

```bash
# 终端 1：尚未启动驱动时执行
bash /home/tan/HK/scripts/run_gemini335l_ros2.sh

# 终端 2：打开标定窗口（脚本自动使用系统 ROS / Python）
bash /home/tan/HK/scripts/calibrate_gemini335l_rgb.sh
```

默认使用 **12 × 9 个方格、15 mm 边长**的棋盘格，对应 **11 × 8 个内角点**。
可在窗口右侧修改规格，开始采集后锁定；清空样本后可再次修改。例如一块有
9 × 6 个内角点、25 mm 方格的棋盘，应设置 10 × 7 个方格：

```bash
bash scripts/calibrate_gemini335l_rgb.sh \
  --squares-x 10 --squares-y 7 --square-size-mm 25
```

1. 让完整、平整的棋盘入镜，确认角点叠加显示；按空格或点击“采集当前视图”。
2. 也可勾选“自动采集”，程序会筛选清晰度、棋盘大小和姿态差异，默认收集 30 张。
   移动后短暂停稳，覆盖画面中心、四角和边缘，同时改变距离及前后/左右倾角。
3. 至少收集 12 张后点击“标定并保存内参”。后台计算完成后显示 fx、fy、cx、cy、
   k1、k2、p1、p2、k3，以及总体 RMS 和各视图重投影误差（像素）。
4. 默认保存到 `calibration_results/gemini335l_rgb/`，包含 ROS CameraInfo 格式 YAML、
   JSON 报告、原始灰度样本目录及 `.session.json` 来源记录。来源记录中的
   `source_camera_info` 是采集时驱动发布的参数，计算出的新内参在 YAML / 主 JSON 中。

默认订阅 `/gemini335l/color/image_raw` 和 `/gemini335l/color/camera_info`，采用
Best Effort QoS；可通过 `--topic` 和 `--camera-info-topic` 改名。
当前 RGB 驱动配置为最大分辨率 1280 × 800 / 30 Hz，输出会记录实际输入分辨率。
此前 640 × 480 的内参不能直接用于该模式，需要重新采样标定。请使用原始图像，保持分辨率、
ROI、镜像、旋转及其他成像设置不变；检测到 CameraInfo 几何信息变化后会要求清空重采。
未反映在 CameraInfo 中的设置变化需要自行清空重采。参数不能直接套用到不同分辨率或
另一台相机。程序将结果保存为独立文件，不自动替换驱动参数。

重投影误差低不保证标定准确，仍需确保棋盘规格正确、平整、姿态和图像区域覆盖充分。
内参标定只使用棋盘角点，不使用先前 AprilTag 世界坐标或坐标轴变换。

## GEMINI 335L AprilTag 识别

启动已有的 335L ROS 驱动后执行（驱动已运行时无需重复启动）：

```bash
bash /home/tan/HK/scripts/run_gemini335l_apriltag.sh
```

[gemini335l_apriltag.py](gemini335l_apriltag.py) 直接订阅 RGB 原始图像，使用
[config/gemini335l_apriltag.yaml](config/gemini335l_apriltag.yaml) 中指定的新标定内参。
标签配置为 tag36h11、ID 0、80 mm 黑色外边框边长（不含外围白边）。
RGB 驱动与 AprilTag 内参均已切换到 1280 × 800。当前使用
`gemini335l_rgb_20260911_142253_747809.yaml`：189 张标定样本，RMS 0.1036 px。
后续重新标定时更新 `camera.intrinsics_file`，或在窗口中选择新文件。
尺寸不匹配时不会输出位姿。
窗口显示检测边框、XYZ 三色坐标轴、相机坐标下的 XYZ 毫米位置、XYZ 欧拉角及重投影误差。
标签轴与海康程序一致：新 +X = 旧 +Z，新 +Y = 旧 −X，新 +Z = 旧 −Y。
相机坐标轴保持 X 向右、Y 向下、Z 向前。

右侧可修改边长、ID 列表和相对位姿参考 ID，点击“应用标签参数”生效；ID 留空检测全部。
同时检测多个标签时，所有标签须使用所填写的边长，否则距离估计比例会出错。
参考标签与其他标签同时可见时显示相对位姿。当前配置以 ID 0 为参考。
“选择 RGB 内参文件”可以切换内参；图像尺寸不匹配时程序会拒绝估计位姿。
窗口内修改只作用于本次运行，持久设置请修改 YAML。

可通过命令行指定其他内参或话题：

```bash
bash scripts/run_gemini335l_apriltag.sh --intrinsics /path/to/gemini_rgb.yaml
```

ROS 输出：

- `/tf`：`gemini335l_color_optical_frame → gemini335l_tag_0`（其他 ID 类推），
  平移按 ROS 使用米，旋转为重新定义后的标签轴，时间戳使用原始 RGB 图像时间。
  仅当前检测到标签时更新；TF 缓存中的旧变换不能作为标签仍然可见的依据。
- `/gemini335l/apriltag/poses_json`（`std_msgs/msg/String`）：包含图像时间、ID、
  XYZ 毫米位置、旋转矩阵、四元数、角点、误差及相对位姿；无检测时 `tags` 为空。
  图像断流时不再发布，下游应检查消息新鲜度。

RViz 中启用 TF 显示即可查看标签坐标系；根坐标可用 `gemini335l_link`。
点击“保存当前画面和位姿 JSON”会写入 `captures/gemini335l_apriltag/`。
这里输出的是相机坐标及标签间相对坐标；335L 的世界外参需要独立标定，不能套用海康相机的外参。

## 双相机世界坐标与 RViz 差异比较

```bash
bash /home/tan/HK/scripts/run_dual_camera_world.sh
```

[dual_camera_world.py](dual_camera_world.py) 在一个窗口同时打开海康 MVS 图像和
335L ROS RGB 图像，各自使用本机已标定内参识别所有 tag36h11 标签；同时打开 RViz。
335L 驱动已运行时直接复用，否则程序启动自己的驱动并在退出时关闭它。
335L 红外结构光投射器默认关闭；比较程序启动时也会调用服务关闭并读取硬件状态，
确认关闭后再打开双画面，避免红外投射干扰海康图像。
海康相机需要独占连接，先关闭其他占用它的 MVS/海康预览程序。

1. 固定两台相机，将 **ID 0、80 mm** 标签固定到世界坐标
   **(126.1, −12.5, 142) mm**，重定义后的标签轴与世界轴同向。
   坐标和方向在 [config/dual_camera_world.yaml](config/dual_camera_world.yaml) 中设置。
2. 让两台相机都看到完整 ID 0，点击“ID 0 已就位：标定两台相机”。
   程序采集 60 对有效观测，检查位置/旋转稳定性，分别计算
   `T_world_camera = T_world_reference @ inverse(mean(T_camera_reference))`。
   仅位置已知不足以求完整外参，此处使用上述已知方向约束。
3. 标定成功后**锁定外参**，自动保存至 `calibration_results/world/dual_时间戳/`。
   后续世界坐标为 `T_world_camera @ T_camera_tag`，不会用每帧 ID 0 重新对齐。
   移动相机、修改镜头或成像几何设置后须重新标定。
4. 用另一个相同边长的标签（如 ID 1）在共同视野中检查不同位置。
   RViz 中海康为青色、335L 为橙色，显示两套世界位姿、三轴及差值连线。
   差值定义为 **海康 − 335L**，显示 ΔXYZ、三维距离和旋转夹角。
   窗口还显示样本对数、距离差 RMS 和最大值；换测试位置后可清零统计。
5. 点击“保存当前比较报告”保存该时刻世界位姿、时间信息和累计差异统计。
   退出程序时也会保存已标定会话的比较报告。当前配置的 `saved_session` 指向
   `dual_20260911_144419_741510`，下次启动默认恢复这组固定外参，不需要 ID 0 留在场景中。

**差异的含义**：ID 0 本身是共同标定基准，在原位置比较主要反映重复性，不能作为
独立准确度验证；不同标签可检验相机之间的一致性，也不等同于相对于外部真值的误差。
两相机没有硬件同步，程序按主机接收时间近似配对（默认上限 80 ms），一帧只用一次，
标定帧不计入比较统计。请先静止目标再读数，动态运动会混入传输/曝光时间差。
超过 0.5 秒的数据不参与实时差值，丢失标签时清除对应 RViz 标记。

ROS 接口：

- `/dual_camera_world/markers`：`visualization_msgs/msg/MarkerArray`，RViz 固定坐标系为 `world`。
- `/dual_camera_world/comparison_json`：`std_msgs/msg/String`，世界坐标和差异统计 JSON。
- `/dual_camera_world/calibrate`：`std_srvs/srv/Trigger`，与界面标定按钮相同。
- TF：`world → comparison/{hik,gemini}_camera` 和
  `world → comparison/{hik,gemini}_tag_ID`。使用独立名称显示估计结果，
  不覆盖 Orbbec 驱动的内部 TF；此视图没有把深度点云变换到世界系。

如果 ID 0 已在已知位置与方向上固定，可用 `--calibrate-on-start` 启动自动采样。
该选项会忽略默认保存的外参；重新标定后如需默认使用新结果，更新配置中的 `saved_session`。
如果两台相机和成像设置都没有变化，可用 `--load-session calibration_results/world/dual_时间戳`
加载保存的外参；程序会核对序列号、内参文件哈希、标签规格和已知世界位姿。
所有参与测距的标签须符合配置中的 `tag_size_mm`，目前统一为 80 mm。
拆除所有标签后，相机在世界系中的固定位置仍有效；程序不会凭空输出目标的位置或两机差值。

## 棋盘格内参标定

[hikrobot_calibration.py](hikrobot_calibration.py) 默认按以下标定板规格运行：

- 方格数：12 × 9；
- OpenCV 检测内角点数：11 × 8，共 88 点；
- 方格边长：15 mm。

这里的 **12 × 9 指方格数，不是内角点数**。启动实时标定工具：

```bash
conda activate revo3_ros
python hikrobot_calibration.py
```

程序会自动连接第一台相机并检测棋盘格。绿色角点完整出现后，可以按空格或
“采集当前视图”；默认也会自动采集清晰且姿态不同的视图。至少需要 12 张，
推荐采集 20–30 张。应让棋盘覆盖画面中心、四角和边缘，并改变距离、旋转以及
前后/左右倾斜角度，但所有姿态都应位于镜头清晰的工作距离内。

右侧的“相机参数”页会根据当前相机的 GenICam 能力动态列出可用参数；不同型号、
彩色/黑白相机看到的项目可能不同，不支持的白平衡、光学控制等项目不会作为可调
参数提供。曝光、增益等允许在线修改的参数会直接更新。ROI、分辨率、像素格式、
合并/抽样、翻转或旋转等节点通常只有暂停取流后才会出现；点击窗口顶部“暂停取流”
即可刷新并修改。如果某型号允许这些节点在线写入，工具仍会在写入前安全停流，成功
后再恢复。

内参和采集时的成像几何配置是一一对应的。改变 `Width`、`Height`、`OffsetX`、
`OffsetY`、Binning、Decimation、翻转、旋转、像素格式或电子光学参数后，工具会清空
已经采集的标定样本和旧结果，必须在新配置下重新采集。即使 ROI 的宽高没有变化，
只改变偏移量，旧样本也不能继续混用。加载相机 User Set 也可能同时改变这些配置，
因此同样需要重新采样。标定期间还应保持镜头焦距、对焦和光圈不变。

### 6 mm 定焦镜头与景深

“6 mm 定焦”表示焦距固定，并不表示软件可以改变光学景深。景深主要取决于光圈、
对焦距离、传感器和允许的清晰度范围；相机曝光、增益或数字锐化不能把失焦图像真正
变清晰。开始采样前建议：

- 将镜头对焦环调到实际使用距离的中间位置，并在整个预期工作范围内检查棋盘角点；
- 如果镜头有手动光圈环，适当收小光圈（更大的 f 数）来增加景深，然后锁紧对焦和
  光圈环；
- 收小光圈后优先增加稳定、均匀且无频闪的照明，再调整曝光时间；尽量保持低增益，
  避免噪声影响亚像素角点；
- 移动标定板时使用足够短的曝光以避免运动模糊，画面稳定后再采集；
- 标定完成后不要重新调焦或改变光圈。若必须调整，应清空样本并重新标定。

参数页和标定状态会显示当前画面/棋盘的清晰度，可用于辅助物理调焦；它是相对质量
指标，不是自动对焦。若镜头本身没有可调光圈或对焦机构，只能通过补光、改变工作距离
或更换更合适的镜头来改善清晰范围。

点击“标定并保存内参”后会生成：

- ROS `CameraInfo` 兼容的 YAML，包含相机矩阵和 `plumb_bob` 畸变参数；
- 同名 JSON 完整报告，包含 RMS、各视图误差、外参和异常视图记录；
- `<输出名>_samples/`，保存本次标定使用的原始灰度样本。

如果厂家所说的“12 × 9”实际指 **内角点数**，请改用：

```bash
python hikrobot_calibration.py --inner-corners 12x9 --square-size-mm 15
```

也可以从已有图片离线重算，不需要连接相机：

```bash
python hikrobot_calibration.py \
  --images calibration_images/ \
  --squares 12x9 \
  --square-size-mm 15 \
  --output calibration_results/hikrobot_intrinsics.yaml
```

输出中的主要内参为：

```text
camera_matrix = [[fx,  0, cx],
                 [ 0, fy, cy],
                 [ 0,  0,  1]]
distortion    = [k1, k2, p1, p2, k3]
```

## AprilTag 实时 6D 位姿 GUI

[apriltag_pose_gui.py](apriltag_pose_gui.py) 从 Hikrobot 相机取流，在视频窗口中实时检测
AprilTag，并为每个标签绘制四个角点、XYZ 坐标轴和 6D 姿态文字。程序同时显示标签相对
相机的平移/旋转；指定参考标签后会显示其他标签相对于该标签的相对位姿，未指定时多
标签帧自动以最小 ID 作为参考。
坐标轴端点标有红色 X、绿色 Y、蓝色 Z；画面文字和实时位姿面板明确显示
标签中心在相机坐标系中的 X、Y、Z 位置（毫米），相对位姿也使用相同的分量标注。

项目提供了可直接使用的配置模板
[config/apriltag_pose.yaml](config/apriltag_pose.yaml)：

```bash
source /opt/ros/jazzy/setup.bash
conda activate revo3_ros
python apriltag_pose_gui.py --config config/apriltag_pose.yaml
```

模板中的先验参数是：

- AprilTag 字典 `tag36h11`；
- 标签外边长 `48.35 mm`（PnP 计算时为 `0.04835 m`）；
- 内参文件 `calibration_results/hikrobot_intrinsics_20260905_124817.yaml`；
- 标定图像尺寸 `1440 × 1080`，相机矩阵和 `plumb_bob` 畸变系数直接从该文件读取。

### NVIDIA CUDA 加速后端

配置中的 `detector.backend: auto` 会优先尝试 NVIDIA Isaac ROS AprilTag CUDA
后端；如果当前系统没有 Isaac ROS 接口，程序自动回退到 OpenCV CPU 后端。GPU
后端通过 [NVIDIA 官方 Isaac ROS AprilTag](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_apriltag/isaac_ros_apriltag/index.html)
组件完成 `tag36h11` 检测和位姿估计，GUI 仍负责 HIKROBOT 取流、显示和相对位姿计算。

当前电脑已检测到 NVIDIA GeForce RTX 5070 Laptop GPU（驱动 580.173.02）。该驱动
适合使用 Isaac ROS release 4.0；要实际启用 CUDA 后端，需要把官方组件安装到系统
ROS Jazzy 环境（这一步需要 sudo）：

```bash
cd /home/tan/HK
bash scripts/setup_isaac_ros_apriltag.sh
source /opt/ros/jazzy/setup.bash
# 可选检查
bash scripts/setup_isaac_ros_apriltag.sh --check
# 自动检查 GPU、CUDA/VPI、ROS 接口并隔离启动一次 cuAprilTag
bash scripts/test_isaac_ros_apriltag.sh
# 使用真实相机验证 CPU AprilTag 是否达到 30 Hz（需要相机在线）
bash scripts/test_cpu_apriltag_30hz.sh
```

脚本会同时检查并安装 CUDA 13 runtime（`libcudart.so.13`）和 NVIDIA VPI 4 runtime
（`libnvvpi.so.4`）；缺少任一库时，Isaac ROS 组件无法加载，GUI 会显示“CUDA 回退”
并使用 CPU。脚本默认使用 NVIDIA 中国镜像；
网络不通时可改用官方主站：
`ISAAC_ROS_MIRROR=https://isaac.download.nvidia.com bash scripts/setup_isaac_ros_apriltag.sh`。
如果系统没有 sudo 权限，需由管理员执行该脚本；Conda 环境本身不需要安装 Isaac ROS
二进制包。

然后直接按前面的 GUI 命令启动即可。`nvidia.auto_start: true` 会自动启动项目内的
`launch/isaac_ros_apriltag_bridge.launch.py`，并传入 `0.04835 m`、`tag36h11`、
`CUDA` 和当前检测参数。若 Isaac ROS 节点由其他 launch 文件管理，将
`nvidia.auto_start` 改为 `false`，并保证它订阅 `image`/`camera_info`、发布
`tag_detections`。CUDA 后端目前只支持 `tag36h11`；其他字典会自动使用 CPU 或在
强制 `backend: nvidia` 时给出明确错误。

GPU 后端会先对畸变图像做校正，再将 RGB 帧传给 Isaac ROS；返回的位姿会重新投影到
原始 HIKROBOT 图像上，因此视频叠加仍与原始画面坐标一致。

CPU 回退路径默认使用 `detector.quad_decimate: 2.0`，在当前 1440×1080 相机上可达到
30 Hz 以上；若标签在画面中很小、需要最高角点精度，可改为 `1.0`，代价是检测速度明显下降。

桥接程序默认将 Fast DDS 的本地传输设为 UDPv4，避开被异常中断的组件容器遗留的
`fastrtps_portXXXX` 共享内存锁；退出 GUI 时也会连同它启动的组件进程组一起清理。

内参文件不是按文件名硬编码的，修改 `camera.intrinsics_file` 即可切换另一组标定结果。
路径默认相对于配置文件所在目录解析（模板因此使用 `../calibration_results/...`）；也支持把同样的参数写成根目录的
[apriltag_pose_config.yaml](apriltag_pose_config.yaml)。

### 坐标系和相对位姿约定

标签坐标系原点位于标签中心，轴方向按原坐标轴重新定义：
**新 +X = 旧 +Z，新 +Y = 旧 −X，新 +Z = 旧 −Y**。
画面坐标轴、旋转向量、欧拉角和标签间相对位姿均使用新定义。
各检测后端先在原生坐标系中完成求解和重投影，再统一应用这组轴映射。
只改变标签坐标系的朝向，不移动标签中心，因此相机系中的平移数值不变；
以标签为参考的相对平移分量则随新坐标轴转换。
相机坐标系遵循 OpenCV 约定：X 向右、Y 向下、Z 从镜头指向场景。GUI 输出的
`tvec` 在核心接口中使用米，界面默认将平移显示为毫米；旋转同时显示 Rodrigues
向量（弧度）和 XYZ 欧拉角（度）。

要计算标签之间的相对位姿，把配置中的 `relative.reference_tag_id` 改为固定标签的
ID，例如：

```yaml
relative:
  enabled: true
  reference_tag_id: 0
```

程序采用
`T_reference_tag = inverse(T_camera_reference) @ T_camera_tag`，因此输出的平移和旋转
表示“参考标签坐标系中的目标标签”。指定 `reference_tag_id` 时使用固定参考；留空时，
若当前帧有多个标签，GUI 自动选择 ID 最小的标签作为参考，只有一个标签时不臆造相对坐标系。
无论是否有相对位姿，每个检测到的标签都会显示相对于相机的完整 6D 位姿。`tag.ids: []`
表示检测该字典的全部 ID；填写整数列表可降低场景中无关标签的误检和计算量。

## GEMINI 335L 双目相机与 ROS 2

项目新增 Orbbec Gemini 335L 的 ROS 2 Jazzy 配置，使用
[官方 Orbbec ROS 2 驱动](https://github.com/orbbec/OrbbecSDK_ROS2)。本机设备序列号为
`CP2AB53000BE`，USB ID 为 `2bc5:0804`，已检测为 USB 3.0（5000 Mbps）。

系统驱动安装（需要在本机终端输入 sudo 密码）：

```bash
sudo bash /home/tan/HK/scripts/setup_gemini335l_system.sh
```

安装脚本使用 `.cache/orbbec-packages/` 中已下载的 Orbbec **2.9.3** 驱动和依赖，
通过 `dpkg --install` 直接安装本地文件，避免镜像下载故障。官方原包重复包含了
现有 `ros-jazzy-magic-enum` 所拥有的 `magic_enum.hpp`，因此本机使用版本后缀为
`+hk1` 的修复包：仅移除重复头文件并声明该依赖，驱动二进制保持原样。
原包的来源和 APT 元数据 SHA256 记录在 `manifest.json` / `SHA256SUMS` 中；
实际安装包的修改说明和哈希记录在 `install-manifest.json` / `INSTALL_SHA256SUMS` 中。
安装官方 udev 规则后，只触发 Orbbec 设备的权限刷新。这里不执行相机固件升级。

启动相机：

```bash
bash /home/tan/HK/scripts/run_gemini335l_ros2.sh
```

在 RViz2 中显示（保持相机驱动运行，在另一个终端执行）：

```bash
bash /home/tan/HK/scripts/view_gemini335l_rviz2.sh
```

RViz2 自动加载 [config/gemini335l.rviz](config/gemini335l.rviz)，以 `gemini335l_link`
为固定坐标系，显示彩色图像、深度图像、按距离着色的三维点云及相机 TF。
图像和点云订阅采用 Best Effort，与驱动一致。需要左右红外画面时，在 Displays 中
勾选 `Left IR` / `Right IR`。鼠标拖动可旋转三维视角，滚轮可缩放。

脚本使用系统 ROS 2 / Python 环境，可从 Conda 终端启动，保留当前 `ROS_DOMAIN_ID`。
也可先 `source /opt/ros/jazzy/setup.bash`，再直接使用项目 launch：

```bash
ros2 launch /home/tan/HK/launch/gemini335l.launch.py
```

配置文件为 [config/gemini335l.yaml](config/gemini335l.yaml)。默认启用彩色、深度、
左右红外、深度点云、相机内参和 TF。RGB 使用本机最大分辨率 **1280×800 / 30 Hz**，
深度和左右红外使用 **640×480 / 30 Hz**。RGB 支持列表见
[logs/gemini335l/supported_profiles.log](logs/gemini335l/supported_profiles.log)。
图像/点云使用 `sensor_data` QoS（Best Effort），CameraInfo 使用默认可靠传输。
启动脚本和 launch 默认使用 Fast DDS 的
`LARGE_DATA?max_msg_size=8MB&sockets_size=16MB&non_blocking=true` 配置，以避免本机
默认传输配置下大消息的接收掉帧。配置只作用于启动的进程及其子进程，已显式设置的
`FASTDDS_BUILTIN_TRANSPORTS` 会被保留。原理参见
[Fast DDS 大数据传输文档](https://fast-dds.docs.eprosima.com/en/v2.14.6/fastdds/use_cases/tcp/tcp_large_data_with_options.html)。
其他订阅程序也可先 `source /home/tan/HK/scripts/gemini335l_ros_env.sh` 使用相同配置；
跨机器传输时，发送端和接收端都应配置兼容的大数据传输模式。
主题路径如下：

| 数据 | ROS 2 话题 | 类型 |
| --- | --- | --- |
| 彩色图像 | `/gemini335l/color/image_raw` | `sensor_msgs/msg/Image` |
| 深度图像 | `/gemini335l/depth/image_raw` | `sensor_msgs/msg/Image` |
| 左红外图像 | `/gemini335l/left_ir/image_raw` | `sensor_msgs/msg/Image` |
| 右红外图像 | `/gemini335l/right_ir/image_raw` | `sensor_msgs/msg/Image` |
| 各流相机内参 | `/gemini335l/{color,depth,left_ir,right_ir}/camera_info` | `sensor_msgs/msg/CameraInfo` |
| 深度点云 | `/gemini335l/depth/points` | `sensor_msgs/msg/PointCloud2` |
| 相机内部坐标关系 | `/tf_static` | `tf2_msgs/msg/TFMessage` |

深度保持原生深度坐标系，未开启彩色对齐。`enable_depth_scale=true` 时深度图的
`16UC1` 值以毫米表示，点云 XYZ 使用米。各流自身的内参和坐标系以 `CameraInfo`
和消息 `header.frame_id` 为准，不能直接套用 HIKROBOT 相机的内参或世界外参。
驱动按需发布部分数据，观察某个流时需先订阅相应话题。

实机验证（先停止正在运行的 Gemini 驱动，避免同一相机被两个进程占用）：

```bash
bash /home/tan/HK/scripts/test_gemini335l_ros2.sh --seconds 20
```

验证脚本在独立 ROS domain 221 启动驱动，检查四路图像及各自的 CameraInfo、有效深度、
点云 XYZ、递增时间戳和光学帧的 TF 连通性，最后自动关闭它启动的驱动。
JSON 报告和驱动日志保存在 `logs/gemini335l/`。报告只有 `passed: true` 才表示实机数据验证通过。
2026-09-11 已在本机 GEMINI 335L（固件 1.4.60）完成实测：启用上述大数据传输配置，
同时订阅四路 640×480 图像和深度点云，实收均约 29.97 Hz；四组 CameraInfo、
有效深度、递增时间戳和全部光学帧的 TF 检查通过。结果见
[logs/gemini335l/verification.json](logs/gemini335l/verification.json)。
随后已切换 RGB 至 1280×800 / 30 Hz，并完成四路图像、CameraInfo、点云和 TF 联合验证；
结果见 [logs/gemini335l/max_rgb_verification.json](logs/gemini335l/max_rgb_verification.json)。
若需要检查已经运行的节点，可在同一 ROS domain 下执行：

```bash
bash -c 'source /home/tan/HK/scripts/gemini335l_ros_env.sh; \
  /usr/bin/python3 /home/tan/HK/scripts/check_gemini335l_topics.py --seconds 10'
```

其他 ROS 2 节点在相同 ROS domain 中订阅上述标准消息即可；图像订阅端使用
`rclpy.qos.qos_profile_sensor_data` 可匹配此配置。

## 世界坐标系标定

[world_calibration_gui.py](world_calibration_gui.py) 使用 **tag36h11、ID 0、80 mm**
标签标定固定相机到世界坐标系的变换，复用现有相机取流和 AprilTag 检测。

```bash
conda run -n revo3_ros python world_calibration_gui.py
```

默认配置为 [config/world_calibration.yaml](config/world_calibration.yaml)，可以用
`--config` 指定其他文件。默认 ID 0 原点的世界坐标为 **(126.1, −12.5, 142) mm**；
经过上述轴变换后，标签与世界系三个正轴同向，因此标签在世界系中的 R/P/Y 均为零。
前面的轴变换负责方向定义，这里的已知坐标负责原点位置；只给一个原点不能确定额外的
安装旋转。如果三个正轴仍有偏差，需要在界面输入已知的 R/P/Y（度），采用
`R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`，程序不会凭原点坐标推断这些角度。

1. 固定相机和 ID 0，让标签原点位于上述世界坐标、轴方向与设置一致。
2. 启动程序，在“世界标定”页确认世界原点和 R/P/Y，点击“开始标定并保存”。
3. 默认采集 60 个不同帧中的有效 ID 0，要求重投影误差不超过 1.5 px；
   位置/旋转的采样 RMS 分别不超过 2 mm / 1°，超时或不稳定时需重新采样。
4. 完成后自动保存到 `calibration_results/world/world_calibration_时间戳.json`，
   页面显示相机原点的世界坐标、变换矩阵和检测到的标签的实时世界 XYZ/RPY。

核心计算为：

```text
T_world_reference = [I, (0.1261, -0.0125, 0.142); 0, 0, 0, 1]  # 默认方向对齐
T_world_camera = T_world_reference @ inverse(T_camera_reference)
T_world_tag = T_world_camera @ T_camera_tag
```

输入的 `T_camera_reference` 已包含标签轴变换，世界标定不会再次变换标签轴。
程序平均多帧相机系平移，并将旋转矩阵的均值投影回合法旋转矩阵；RMS 反映采样重复性，
不代表绝对测量精度。固定标定后即使 ID 0 离开画面，其他标签仍可输出世界位姿。
默认检测全部 ID，所有被测标签使用同一个 80 mm 边长先验。

保存文件中的 `T_world_camera`、`T_camera_world`、`T_world_reference` 和
`T_camera_reference` 均为 4×4 矩阵，**平移单位为米**；界面位置和
`camera_origin_world_mm` 使用毫米。文件还记录相机序列号、内参、图像尺寸、检测后端、
标签尺寸、轴定义和采样统计信息。“加载标定”会检查这些参数是否匹配。
相机移动后必须重新标定；重启加载旧结果时应保证相机仍位于原来的固定位置。

其他 Python 程序可以通过 [world_calibration_core.py](world_calibration_core.py) 使用结果：

```python
import numpy as np
from world_calibration_core import WorldCalibration

calibration = WorldCalibration.load("calibration_results/world/你的标定文件.json")
point_world_m = calibration.world_points(np.array([0.02, -0.01, 0.5]))
# camera_from_tag 为检测得到的 4×4 位姿，平移单位 m，已使用新的标签轴定义。
# world_from_tag = calibration.world_pose(camera_from_tag)
```

## 实时视频 GUI

[hikrobot_gui.py](hikrobot_gui.py) 提供：

- 实时视频显示和实时 FPS、分辨率状态；
- 运行中调整曝光时间、增益和采集帧率；
- 曝光、增益的自动/单次/连续模式，以及相机支持时的自动白平衡模式；
- 相机连接/断开、开始/停止取流、重新读取参数；
- 保存当前显示帧为 PNG；
- 根据相机返回值设置常用数值范围。

先按“环境配置”一节激活 `revo3_ros`，再启动 GUI：

```bash
python hikrobot_gui.py
```

程序启动后会自动连接第一台相机并开始取流。手动曝光或增益控制只有在对应自动模式
为 `Off` 时才会启用。完整的动态 GenICam 参数浏览以及 ROI、触发、I/O、高级命令
控制位于 [hikrobot_calibration.py](hikrobot_calibration.py) 的“相机参数”页；该页
会按设备运行时能力过滤，并对危险操作进行确认。

普通实时 GUI 使用 Conda 环境 `revo3_ros` 中的 Python 和 PyQt5；其图像颜色转换由
MVS SDK 完成。AprilTag GUI 另外使用 OpenCV/NumPy 完成灰度化、检测和 PnP 位姿计算，
因此应按 `requirements.txt` 安装完整依赖。启动时若提示缺少 `cv2.aruco`，请将环境中
相互冲突的 `opencv-python` 替换为同版本的 `opencv-contrib-python`（两者不要同时安装）。
NVIDIA CUDA 后端属于系统 ROS/Isaac ROS 组件，不放入 Conda `requirements.txt`；未安装
时不会影响 CPU 模式。

## 命令行抓图

[hikrobot_camera.py](hikrobot_camera.py) 支持：

- 枚举 USB3 Vision 和 GigE Vision 相机；
- 按设备序号或序列号选择相机；
- 打开相机并关闭触发模式；
- 可选设置曝光时间、增益；
- 获取一帧并通过 MVS SDK 保存为 PNG；
- 无论成功或失败，都按顺序释放图像缓存、取流、设备和 SDK 资源。

## 环境配置

系统前置条件：

- 已安装 HIKROBOT MVS Linux SDK，默认位置为 `/opt/MVS`；
- 相机已连接，并且当前用户拥有 USB 设备访问权限；
- 已创建 Conda 环境 `revo3_ros`（当前验证版本为 Python 3.12）。

首次使用时，在项目目录安装已记录的 Python 依赖：

```bash
conda activate revo3_ros
python -m pip install -r requirements.txt
```

确认当前终端确实使用目标环境：

```bash
python -c "import sys, PyQt5; print(sys.executable); print(PyQt5.__file__)"
```

第一行应为 `/home/tan/miniconda3/envs/revo3_ros/bin/python`。VS Code 工作区也已
将该解释器设为默认值，并提供内参标定、枚举相机、实时 GUI 和 AprilTag 6D 位姿四个
运行/调试配置；AprilTag 配置项会自动指向 `config/apriltag_pose.yaml`。

程序默认从 `/opt/MVS/Samples/64/Python/MvImport` 加载官方 Python
绑定。如果 MVS 安装在其他位置，可以设置 `MVCAM_SDK_PATH`。

不想激活环境时，也可以显式运行：

```bash
conda run -n revo3_ros python hikrobot_camera.py --list
conda run -n revo3_ros python hikrobot_gui.py
conda run -n revo3_ros python hikrobot_calibration.py
conda run -n revo3_ros python apriltag_pose_gui.py --config config/apriltag_pose.yaml
```

### 使用

枚举相机：

```bash
python hikrobot_camera.py --list
```

使用第 0 台相机抓取一帧：

```bash
python hikrobot_camera.py
```

图像默认保存到 `captures/hikrobot_时间戳.png`。也可以指定文件名：

```bash
python hikrobot_camera.py --output captures/test.png
```

按序列号选择相机，并设置曝光和增益：

```bash
python hikrobot_camera.py \
  --serial DB0447679 \
  --exposure-us 20000 \
  --gain 5 \
  --output captures/manual.png
```

如果提示设备被占用，请先在 MVS 客户端里断开该相机，或者关闭 MVS
客户端，再重新运行示例。

查看全部参数：

```bash
python hikrobot_camera.py --help
```
