# PointPillars 训练测试流程学习指南

## 目录
1. [概述](#概述)
2. [整体架构](#整体架构)
3. [训练流程详解](#训练流程详解)
4. [测试/推理流程](#测试推理流程)
5. [Loss 计算详解](#loss-计算详解)
6. [配置文件说明](#配置文件说明)

---

## 概述

PointPillars 是基于点云的 3D 目标检测算法，核心思想是将点云转换为 Pseudo-image，然后用 2D CNN 进行检测。

**参考论文**: PointPillars: Fast Encoders for Object Detection from Point Clouds

---

## 整体架构

```
输入: 点云 (N, 4) - N个点，每点 x, y, z, intensity
         ↓
┌─────────────────────────────────────────────────────────────┐
│  Pillar Feature Encoder (VFE)                                │
│  - 将点云体素化，提取每个 Pillar 的特征                       │
│  - 输出: pillar_features (num_pillars, C)                    │
│         voxel_coords (num_pillars, 4) - batch, z, y, x      │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│  PointPillarScatter                                          │
│  - 将 Pillar 特征散布到 BEV (Bird's Eye View) 伪图像          │
│  - 输出: spatial_features (B, C, H, W)                       │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│  Backbone2D (Conv2D)                                         │
│  - 2D 卷积网络提取 BEV 特征                                   │
│  - 输出多尺度特征图                                           │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│  DenseHead (AnchorHeadTemplate)                             │
│  - 生成 Anchor                                               │
│  - 分类: 判断每个位置是否有目标                               │
│  - 回归: 预测目标的 box 参数 (x, y, z, w, h, l, rot)         │
└─────────────────────────────────────────────────────────────┘
```

**关键文件路径:**
- 模型定义: [pcdet/models/detectors/pointpillar.py](pcdet/models/detectors/pointpillar.py)
- 网络基类: [pcdet/models/detectors/detector3d_template.py](pcdet/models/detectors/detector3d_template.py)
- Pillar Scatter: [pcdet/models/backbones_2d/map_to_bev/pointpillar_scatter.py](pcdet/models/backbones_2d/map_to_bev/pointpillar_scatter.py)
- Loss 计算: [pcdet/utils/loss_utils.py](pcdet/utils/loss_utils.py)
- Head: [pcdet/models/dense_heads/anchor_head_template.py](pcdet/models/dense_heads/anchor_head_template.py)

---

## 训练流程详解

### 1. 入口: tools/train.py

```python
# main() 函数流程:
1. parse_config()          # 解析命令行参数和 YAML 配置
2. build_dataloader()      # 构建训练数据加载器
3. build_network()         # 构建 PointPillars 模型
4. build_optimizer()       # 构建优化器 (Adam 或 SGD)
5. build_scheduler()       # 构建学习率调度器
6. train_model()           # 开始训练循环
```

**启动命令示例:**
```bash
python tools/train.py --cfg_file cfgs/pointpillar_models/pointpillar.yaml --batch_size 4
```

### 2. 数据加载: batch_dict 的构建

```python
# batch_dict 包含:
{
    'points': np.ndarray,           # 点云 (N, 4) - x, y, z, intensity
    'frame_id': str,                # 帧 ID
    'calib': Calibration,           # 标定信息
    'gt_boxes': np.ndarray,         # GT boxes (N, 8) - x,y,z,w,h,l,rot,class
    'batch_size': int,              # batch 大小
    ...
}
```

### 3. 模型前向: pcdet/models/__init__.py

```python
# model_fn_decorator() 包装前向传播
def model_func(model, batch_dict):
    load_data_to_gpu(batch_dict)   # 将数据移到 GPU
    ret_dict, tb_dict, disp_dict = model(batch_dict)  # 前向传播
    loss = ret_dict['loss'].mean()
    return ModelReturn(loss, tb_dict, disp_dict)
```

### 4. PointPillars forward: pcdet/models/detectors/pointpillar.py

```python
class PointPillar(Detector3DTemplate):
    def forward(self, batch_dict):
        # 依次通过所有模块
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            return {'loss': loss}, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts
```

### 5. 单步训练: tools/train_utils/train_utils.py

```python
def train_one_epoch(...):
    for cur_it in range(total_it_each_epoch):
        batch = next(dataloader_iter)        # 获取数据
        lr_scheduler.step(accumulated_iter)  # 更新学习率
        optimizer.zero_grad()                # 清零梯度
        with torch.cuda.amp.autocast():       # 混合精度
            loss, tb_dict, disp_dict = model_func(model, batch)
        scaler.scale(loss).backward()         # 反向传播
        scaler.step(optimizer)               # 更新参数
        accumulated_iter += 1
```

### 6. 数据流图 (batch_dict)

```
DataLoader
    ↓ (batch_dict: points, gt_boxes, calib, ...)
    ↓
VFE (PillarFeatureEncoder)
    ↓ (batch_dict: pillar_features, voxel_coords)
    ↓
PointPillarScatter
    ↓ (batch_dict: spatial_features [B, C, H, W])
    ↓
Backbone2D
    ↓ (batch_dict: multi_scale_features)
    ↓
DenseHead
    ↓ (batch_dict: cls_preds, box_preds, ...)
    ↓
get_loss() → total_loss
```

---

## 测试/推理流程

### 1. 入口: tools/test.py

```python
def main():
    # 构建模型和数据加载器 (同训练)
    model = build_network(...)
    test_loader = build_dataloader(training=False)

    with torch.no_grad():
        repeat_eval_ckpt(model, test_loader, ...)  # 评估
```

### 2. 推理流程

```python
# 在 model(batch_dict) 中 (训练模式=False):
pred_dicts, recall_dicts = self.post_processing(batch_dict)

# post_processing 包含:
1. 解码预测的 box 参数 (相对于 anchors)
2. NMS (Non-Maximum Suppression) 过滤重叠框
3. 阈值过滤 (score > 0.1)
4. 输出最终检测结果
```

### 3. 输出格式

```python
# 每帧的输出:
{
    'pred_boxes': np.ndarray,    # (M, 7) - x, y, z, w, h, l, rot
    'pred_scores': np.ndarray,   # (M,) - 置信度
    'pred_labels': np.ndarray,    # (M,) - 类别 1=Car, 2=Pedestrian, 3=Cyclist
}
```

---

## Loss 计算详解

### 1. 总Loss: rpn_loss

位于: [pcdet/models/dense_heads/anchor_head_template.py](pcdet/models/dense_heads/anchor_head_template.py)

```python
def get_loss(self):
    cls_loss, tb_dict = self.get_cls_layer_loss()      # 分类 loss
    box_loss, tb_dict_box = self.get_box_reg_layer_loss()  # 回归 loss
    rpn_loss = cls_loss + box_loss
    return rpn_loss, tb_dict
```

### 2. 分类 Loss: SigmoidFocalClassificationLoss

位于: [pcdet/utils/loss_utils.py](pcdet/utils/loss_utils.py)

```python
class SigmoidFocalClassificationLoss(nn.Module):
    # Focal Loss for 分类
    # 公式: focal_weight * BCE_loss
    # 其中: focal_weight = alpha * (1-pt)^gamma  (for positive)
    #              = (1-alpha) * pt^gamma        (for negative)
    def forward(self, input, target, weights):
        # input: (B, num_anchors, num_classes) 预测 logits
        # target: (B, num_anchors, num_classes) one-hot 标签
        # weights: (B, num_anchors) 每个 anchor 的权重
```

**参数:**
- `alpha=0.25`: 正负样本平衡因子
- `gamma=2.0`: 难例挖掘因子

### 3. 回归 Loss: WeightedSmoothL1Loss

位于: [pcdet/utils/loss_utils.py](pcdet/utils/loss_utils.py)

```python
class WeightedSmoothL1Loss(nn.Module):
    # Smooth L1 Loss (也称 Huber Loss)
    # |x| - 0.5*beta, when |x| >= beta
    # 0.5*x^2/beta,   when |x| < beta
    def forward(self, input, target, weights):
        # input: (B, num_anchors, 7) 预测的 box 编码
        # target: (B, num_anchors, 7) 目标的 box 编码
```

**回归目标 (7个参数):**
```
[x, y, z, w, h, l, rot]
 |  |  |  |  |  |   |
 |  |  |  |  |  |   └── 航向角
 |  |  |  |  └── l --- 长度
 |  |  |  └──── h --- 高度
 |  |  └────── w --- 宽度
 |  └──────── y --- Y 方向偏移
 └────────── x --- X 方向偏移
```

### 4. Anchor 匹配策略

位于: [pcdet/models/dense_heads/target_assigner/](pcdet/models/dense_heads/target_assigner/)

每个特征图位置生成多个 anchors，根据 IoU 匹配正负样本:
- **正样本**: 与任意 GT box IoU > 阈值 (通常 0.6)
- **负样本**: 与所有 GT box IoU < 阈值 (通常 0.3)
- **忽略样本**: 其他情况

---

## 配置文件说明

### PointPillars 配置示例: cfgs/pointpillar_models/pointpillar.yaml

```yaml
MODEL:
  NAME: PointPillars
  VFE:
    NAME: PillarFeatureEncoder
    MAX_POINTS_PER_PILLAR: 32
    MAX_PILLARS: 12000
    NUM_POINT_FEATURES: 4

  MAP_TO_BEV:
    NAME: PointPillarScatter
    NUM_BEV_FEATURES: 64

  BACKBONE_2D:
    NAME: BaseBEVBackbone
    LAYER_NUMS: [3, 5, 5]
    LAYER_STRIDES: [2, 2, 2]
    NUM_FILTERS: [64, 128, 256]
    UPSAMPLE_STRIDES: [1, 2, 4]
    NUM_UPSAMPLE_FILTERS: [128, 128, 128]

  DENSE_HEAD:
    NAME: AnchorHeadSingle
    CLASS_NAMES: ['Car', 'Pedestrian', 'Cyclist']
    ANCHOR_GENERATOR_CONFIG:
      - class_name: Car
        anchor_sizes: [[3.9, 1.6, 1.56]]
        anchor_rotations: [0, 1.57]
        anchor_bottom_heights: [-1.78]
        align_center: False
      ...
    TARGET_ASSIGNER_CONFIG:
      NAME: AxisAlignedTargetAssigner
      BOX_CODER: ResidualCoder
      BOX_CODER_CONFIG: ...
    LOSS_CONFIG:
      LOSS_WEIGHTS:
        cls_weight: 1.0
        loc_weight: 2.0
        dir_weight: 0.2

DATA_CONFIG:
  DATASET: KittiDataset
  POINT_CLOUD_RANGE: [0, -39.68, -3, 70.4, 39.68, 1]
  MAX_POINTS_IN_ALL_BOXES: ...

OPTIMIZATION:
  BATCH_SIZE_PER_GPU: 4
  NUM_EPOCHS: 80
  OPTIMIZER: Adam
  LR: 0.003
```

---

## 快速运行指南

### 训练
```bash
cd /mnt/tj/projects/OpenPCDet
python tools/train.py --cfg_file cfgs/pointpillar_models/pointpillar.yaml --batch_size 4
```

### 测试
```bash
python tools/test.py --cfg_file cfgs/pointpillar_models/pointpillar.yaml --ckpt output/pointpillar/ckpt/checkpoint_epoch_80.pth
```

### 可视化
```bash
python tools/demo.py --cfg_file cfgs/pointpillar_models/pointpillar.yaml --data_path /path/to/lidar.bin
```

---

## 关键类和函数索引

| 文件 | 类/函数 | 说明 |
|------|---------|------|
| tools/train.py | main() | 训练入口 |
| tools/train_utils/train_utils.py | train_model(), train_one_epoch() | 训练循环 |
| pcdet/models/__init__.py | model_fn_decorator() | 包装前向和loss提取 |
| pcdet/models/detectors/pointpillar.py | PointPillar | PointPillars模型 |
| pcdet/models/detectors/detector3d_template.py | Detector3DTemplate | 模型基类 |
| pcdet/models/backbones_2d/map_to_bev/pointpillar_scatter.py | PointPillarScatter | Pillar→BEV |
| pcdet/models/dense_heads/anchor_head_template.py | AnchorHeadTemplate, get_loss() | 检测头和loss |
| pcdet/utils/loss_utils.py | SigmoidFocalClassificationLoss, WeightedSmoothL1Loss | 损失函数 |
| tools/test.py | main(), repeat_eval_ckpt() | 测试入口 |
| pcdet/datasets/kitti/kitti_dataset.py | KittiDataset | KITTI数据集 |
