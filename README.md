# aki模式

动漫头像 4 倍放大流程（Real-CUGAN SE，8 向 TTA，Lab 校色，σ0.65 平滑，眼睛高光保护，三种格式输出）。

## 文件
- `up4x-latest-no-denoise.pth` / `up4x-latest-conservative.pth`：Real-CUGAN SE 4x 权重
- `cugan.py`：网络结构
- `aki_mode.py`：完整流程脚本

## 运行
```bash
pip install torch opencv-python numpy pillow
python3 aki_mode.py 原图 x y w h n0 输出前缀 眼框1 眼框2
```
- `x y w h`：原图上的正方形裁框（脚本自动在四周多取最多 16 像素余量，处理后裁回）
- `n0`：no-denoise（默认）；`n-1`：conservative（仅在可见 JPEG 方块或压缩杂点时使用）
- 眼框：放大 4 倍、裁回后的输出图坐标 `x0,y0,x1,y1`，可省略
- 输出：`前缀.png`、`前缀.jpg`（质量 100、4:4:4）、`前缀_499kb.jpg`（≤499,000 字节），并打印检查报告 JSON
