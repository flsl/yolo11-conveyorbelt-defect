# 基于yolov11的传送带异物检测系统python源码+pytorch模型+评估指标曲线+精美GUI界面

【算法介绍】

## 基于 YOLOv11 的传送带异物检测系统：为工业生产秩序保驾护航

在工业生产朝着自动化、智能化方向迅猛发展的关键时期，精准且高效地监测传送带上的异物情况，已然成为保障生产流程顺畅、提升产品质量以及确保生产安全的核心挑战之一。传送带作为工业生产中物料传输的关键设备，其运行环境复杂多样，物料形态千变万化。不同批次物料的大小、形状、颜色差异显著，传送带运行速度也因生产工艺需求而各有不同，而且生产现场还可能存在光线变化、灰尘干扰等因素。这些状况不仅直观反映了当前生产的稳定性和规范性，更与整个生产线的效率、产品的合格率以及设备的使用寿命等紧密相连。一旦传送带上出现异物，如金属碎片混入食品生产传送带、石块混入建材生产传送带，或者因物料堆积异常导致传送带卡顿、跑偏等情况，若未能及时察觉并采取相应措施，极易引发设备损坏、产品质量下降甚至生产安全事故，给企业带来巨大的经济损失和声誉损害。

传统传送带异物检测方式主要依赖人工目视检查。然而，受生产规模不断扩大、传送带长度日益增加以及生产环境复杂等因素限制，人工检查很难全面覆盖传送带的各个位置，尤其是那些处于设备内部、光线昏暗或者高速运行区域的传送带情况，往往难以被及时观察到。而且，早期基于简单人工判断的监测方法，由于物料形态多样、传送带运行速度变化大以及各种设备、管道遮挡等因素干扰，误判率高达 35%以上，根本无法满足企业“精准化、全方位”的生产管理需求。因此，开发一套具备高精度、强适应性且能实时监测的传送带异物智能检测系统，成为提升工业生产管理水平和治理效能的关键技术突破点。

目前现有技术存在诸多明显瓶颈：人工检查不仅效率极其低下（单人单日仅能完成有限长度传送带的检查），而且检查人员还面临着被高速运转设备夹伤、接触有害物料等风险；基于简单图像特征提取的传统算法，难以准确区分正常物料与异物（例如，颜色相近的杂质混入物料中），在光线昏暗、物料表面反光等低能见度环境下，算法性能会急剧下降；传统目标检测模型对物料和异物的多变性（如不同生产阶段物料的形态变化、异物的种类和大小差异）和尺度变化（从小颗粒杂质到大型障碍物）适应性较差，对于小目标（如微小金属屑）的漏检率超过 45%，难以满足实际工业生产的复杂需求。

基于 YOLOv11 的传送带异物检测系统为工业生产管理带来了革命性的变革。YOLOv11 作为先进的目标检测算法，具备强大的特征提取和实时检测能力。该系统充分发挥 YOLOv11 的端到端实时检测优势，并针对传送带复杂环境进行了深度优化。

此系统能够精准识别“异物”“传送带”和“物料”这三个关键类别。它可以准确捕捉异物的各种形态特征，无论是微小的金属颗粒、细长的纤维杂质，还是较大的石块、木块；无论是与物料颜色相近的隐蔽异物，还是形状奇特的特殊异物，系统都能敏锐识别。同时，对于传送带本身的运行状态，如是否跑偏、有无破损等情况也能准确判断。对于物料，系统可以识别其正常堆积形态和流动状态。通过对大量传送带异物图像数据的学习和训练，系统能够保持较高的检测准确率，即便在光线变化大、物料流动速度快、存在各种灰尘和反光等复杂场景下，也能稳定发挥检测性能。

同时，系统具备强大的抗干扰能力，能够有效应对生产现场的车辆、设备、管道等干扰因素，不会因这些干扰而出现误判或漏判的情况。这为新型智能化工业生产建设提供了坚实的技术支撑，助力企业实现高效、精准、科学的生产管理，及时发现并处理传送带异物问题，保障生产流程的正常进行和产品质量的稳定可靠，提升企业的核心竞争力和市场形象。

【效果展示】

<div style="text-align:center;"><img alt="" src="https://i-blog.csdnimg.cn/direct/4a0059b1b8814b0a99b5dbcb30485111.jpeg"></div>

<div style="text-align:center;"><img alt="" src="https://i-blog.csdnimg.cn/direct/e102dbb869af4274ae3c2a75b1c5a2c7.jpeg"></div>

【测试环境】

windows10
anaconda3+python3.8
torch==2.3.1
ultralytics==8.3.81

【模型可以检测出类别】

chuansongdai(皮带跑偏)
wuliao(物料)
yiwu(异物)

【训练数据集介绍】

数据集格式：Pascal VOC格式+YOLO格式(不包含分割路径的txt文件，仅仅包含jpg图片以及对应的VOC格式xml文件和yolo格式txt文件)
图片数量(jpg文件个数)：437
标注数量(xml文件个数)：437
标注数量(txt文件个数)：437
标注类别数：3
所在仓库：firc-dataset
标注类别名称(注意yolo格式类别顺序不和这个对应，而以labels文件夹classes.txt为准):["chuansongdai","wuliao","yiwu"]=>[皮带,物料,异物]
每个类别标注的框数：
chuansongdai 框数 = 244
wuliao 框数 = 161
yiwu 框数 = 244
总框数：649
使用标注工具：labelImg
标注规则：对类别进行画矩形框
重要说明：暂无
特别声明：本数据集不对训练的模型或者权重文件精度作任何保证，数据集只提供准确且合理标注

图片预览：

 ![](./assets/335_3.jpeg)

标注例子：

 ![](./assets/335_4.jpeg)

【训练信息】

| 参数 | 值 |
|:---:|:---:|
| 训练集图片数 | 415 |
| 验证集图片数 | 22 |
| 训练map | 94.6% |
| 训练精度(Precision) | 92.8% |
| 训练召回率(Recall) | 87.4% |

验证集精度统计：

| Class | Images | Instances | P | R | mAP50 | mAP50-95 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| all | 22 | 25 | 0.928 | 0.874 | 0.946 | 0.748 |
| chuansongdai | 3 | 6 | 0.934 | 1 | 0.995 | 0.931 |
| wuliao | 7 | 7 | 1 | 0.954 | 0.995 | 0.728 |
| yiwu | 12 | 12 | 0.849 | 0.667 | 0.849 | 0.586 |

【界面设计】

```
class Ui_MainWindow(QtWidgets.QMainWindow):
    signal = QtCore.pyqtSignal(str, str)
 
    def setupUi(self):
        self.setObjectName("MainWindow")
        self.resize(1280, 728)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
 
        self.weights_dir = './weights'
 
        self.picture = QtWidgets.QLabel(self.centralwidget)
        self.picture.setGeometry(QtCore.QRect(260, 10, 1010, 630))
        self.picture.setStyleSheet("background:black")
        self.picture.setObjectName("picture")
        self.picture.setScaledContents(True)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(10, 10, 81, 21))
        self.label_2.setObjectName("label_2")
        self.cb_weights = QtWidgets.QComboBox(self.centralwidget)
        self.cb_weights.setGeometry(QtCore.QRect(10, 40, 241, 21))
        self.cb_weights.setObjectName("cb_weights")
        self.cb_weights.currentIndexChanged.connect(self.cb_weights_changed)
 
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(10, 70, 72, 21))
        self.label_3.setObjectName("label_3")
        self.hs_conf = QtWidgets.QSlider(self.centralwidget)
        self.hs_conf.setGeometry(QtCore.QRect(10, 100, 181, 22))
        self.hs_conf.setProperty("value", 25)
        self.hs_conf.setOrientation(QtCore.Qt.Horizontal)
        self.hs_conf.setObjectName("hs_conf")
        self.hs_conf.valueChanged.connect(self.conf_change)
        self.dsb_conf = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.dsb_conf.setGeometry(QtCore.QRect(200, 100, 51, 22))
        self.dsb_conf.setMaximum(1.0)
        self.dsb_conf.setSingleStep(0.01)
        self.dsb_conf.setProperty("value", 0.25)
        self.dsb_conf.setObjectName("dsb_conf")
        self.dsb_conf.valueChanged.connect(self.dsb_conf_change)
        self.dsb_iou = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.dsb_iou.setGeometry(QtCore.QRect(200, 160, 51, 22))
        self.dsb_iou.setMaximum(1.0)
        self.dsb_iou.setSingleStep(0.01)
        self.dsb_iou.setProperty("value", 0.45)
        self.dsb_iou.setObjectName("dsb_iou")
        self.dsb_iou.valueChanged.connect(self.dsb_iou_change)
        self.hs_iou = QtWidgets.QSlider(self.centralwidget)
        self.hs_iou.setGeometry(QtCore.QRect(10, 160, 181, 22))
        self.hs_iou.setProperty("value", 45)
        self.hs_iou.setOrientation(QtCore.Qt.Horizontal)
        self.hs_iou.setObjectName("hs_iou")
        self.hs_iou.valueChanged.connect(self.iou_change)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(10, 130, 72, 21))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(10, 210, 72, 21))
        self.label_5.setObjectName("label_5")
        self.le_res = QtWidgets.QTextEdit(self.centralwidget)
        self.le_res.setGeometry(QtCore.QRect(10, 240, 241, 400))
        self.le_res.setObjectName("le_res")
        self.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1110, 30))
        self.menubar.setObjectName("menubar")
        self.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)
        self.toolBar = QtWidgets.QToolBar(self)
        self.toolBar.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.toolBar.setObjectName("toolBar")
        self.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.actionopenpic = QtWidgets.QAction(self)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/images/1.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionopenpic.setIcon(icon)
        self.actionopenpic.setObjectName("actionopenpic")
        self.actionopenpic.triggered.connect(self.open_image)
        self.action = QtWidgets.QAction(self)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/images/2.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action.setIcon(icon1)
        self.action.setObjectName("action")
        self.action.triggered.connect(self.open_video)
        self.action_2 = QtWidgets.QAction(self)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/images/3.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_2.setIcon(icon2)
        self.action_2.setObjectName("action_2")
        self.action_2.triggered.connect(self.open_camera)
 
        self.actionexit = QtWidgets.QAction(self)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/images/4.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionexit.setIcon(icon3)
        self.actionexit.setObjectName("actionexit")
        self.actionexit.triggered.connect(self.exit)
 
        self.toolBar.addAction(self.actionopenpic)
        self.toolBar.addAction(self.action)
        self.toolBar.addAction(self.action_2)
        self.toolBar.addAction(self.actionexit)
 
        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)
        self.init_all()
```

【常用评估参数介绍】

在目标检测任务中，评估模型的性能是至关重要的。你提到的几个术语是评估模型性能的常用指标。下面是对这些术语的详细解释：

Class：
这通常指的是模型被设计用来检测的目标类别。例如，一个模型可能被训练来检测车辆、行人或动物等不同类别的对象。
Images：
表示验证集中的图片数量。验证集是用来评估模型性能的数据集，与训练集分开，以确保评估结果的公正性。
Instances：
在所有图片中目标对象的总数。这包括了所有类别对象的总和，例如，如果验证集包含100张图片，每张图片平均有5个目标对象，则Instances为500。
P（精确度Precision）：
精确度是模型预测为正样本的实例中，真正为正样本的比例。计算公式为：Precision = TP / (TP + FP)，其中TP表示真正例（True Positives），FP表示假正例（False Positives）。
R（召回率Recall）：
召回率是所有真正的正样本中被模型正确预测为正样本的比例。计算公式为：Recall = TP / (TP + FN)，其中FN表示假负例（False Negatives）。
mAP50：
表示在IoU（交并比）阈值为0.5时的平均精度（mean Average Precision）。IoU是衡量预测框和真实框重叠程度的指标。mAP是一个综合指标，考虑了精确度和召回率，用于评估模型在不同召回率水平上的性能。在IoU=0.5时，如果预测框与真实框的重叠程度达到或超过50%，则认为该预测是正确的。
mAP50-95：
表示在IoU从0.5到0.95（间隔0.05）的范围内，模型的平均精度。这是一个更严格的评估标准，要求预测框与真实框的重叠程度更高。在目标检测任务中，更高的IoU阈值意味着模型需要更准确地定位目标对象。mAP50-95的计算考虑了从宽松到严格的多个IoU阈值，因此能够更全面地评估模型的性能。
这些指标共同构成了评估目标检测模型性能的重要框架。通过比较不同模型在这些指标上的表现，可以判断哪个模型在实际应用中可能更有效。

【使用步骤】

使用步骤：
（1）首先根据官方框架ultralytics安装教程安装好yolov11环境，并安装好pyqt5
（2）切换到自己安装的yolo11环境后，并切换到源码目录，执行python main.py即可运行启动界面，进行相应的操作即可

【提供文件】

python源码
yolo11n.pt模型
训练的map,P,R曲线图(在weights\results.png)
测试图片（在test_img文件夹下面）

注意提供训练的数据集，请到mytxt.txt文件中找到地址
