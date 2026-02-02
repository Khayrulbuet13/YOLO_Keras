# HLS4ml FPGA Synthesis Guide for YOLO

This guide explains how to train, optimize, and synthesize your YOLO model for FPGA deployment using HLS4ml.

## Overview

HLS4ml converts neural networks into FPGA firmware using high-level synthesis. For successful FPGA deployment, your model needs:

1. **Quantization** - Reduced precision (8-16 bit) to fit on FPGA
2. **Static operations** - No dynamic shapes or control flow
3. **Efficient activations** - Minimize expensive operations like softmax/sigmoid

## Architecture Changes

### Three Model Variants

1. **Original Model** (`tinysimov35_keras.py`)
   - Float32 precision
   - Dynamic inference path
   - Used for initial training and validation

2. **Quantized Model** (`tinysimov35_keras_quantized.py`)
   - QKeras quantization-aware training
   - 8-16 bit precision
   - Still has dynamic operations
   - Used for training with quantization

3. **Static HLS4ml Model** (`tinysimov35_keras_hls4ml.py`)
   - Fully static computational graph
   - Pre-computed anchor grids
   - Unrolled operations (no loops)
   - Fixed batch size (1)
   - Used for HLS4ml conversion

### Key Optimizations

#### Quantization Strategy

```
Layer Type          | Weight Bits | Activation Bits | Rationale
--------------------|-------------|-----------------|---------------------------
First Conv          | 16          | 16              | Most sensitive to quantization
Middle Conv Layers  | 8           | 8               | Most efficient, bulk of model
Last Conv + Head    | 12          | 12              | Balance accuracy/resources
```

#### Removed Dynamic Operations

- ❌ `tf.range` loops → ✅ Unrolled operations
- ❌ `tf.meshgrid` → ✅ Pre-computed anchor grids
- ❌ Dynamic reshapes → ✅ Static tensor shapes
- ❌ Variable batch size → ✅ Fixed batch size (1)

#### Expensive Activations

- **Softmax** (in DFL): Resource-intensive, kept for accuracy but flagged for optimization
- **Sigmoid** (in Head): Can use lookup table in HLS4ml config

## Workflow

### Step 1: Train with Quantization

Train your model using quantization-aware training:

```bash
# Train quantized model
python main_keras.py \
    --train \
    --quantized \
    --input-size 128x256 \
    --batch-size 4 \
    --epochs 500 \
    --save-path ./results/quantized_model \
    --dataset-dir ./Dataset/bionano_cellv2
```

This trains with QKeras quantized layers, allowing the model to adapt to reduced precision.

### Step 2: Profile Model Precision

Use HLS4ml profiling to determine optimal bit-widths:

```bash
# Profile model to analyze weight/activation distributions
python hls4ml_convert.py \
    --profile \
    --weights ./results/quantized_model/best.weights.h5 \
    --num-classes 1 \
    --img-size 256x128 \
    --test-samples 100
```

This generates distribution plots showing:
- Weight value ranges per layer
- Activation value ranges per layer
- Recommended precision settings

**Review the plots** in `hls4ml_prj_profile/` and adjust bit-widths in `hls4ml_convert.py` if needed.

### Step 3: Convert to HLS4ml

Convert the trained model to HLS4ml C++ code:

```bash
# Convert to HLS4ml
python hls4ml_convert.py \
    --convert \
    --weights ./results/quantized_model/best.weights.h5 \
    --num-classes 1 \
    --img-size 256x128 \
    --output-dir hls4ml_yolo \
    --fpga xczu9eg-ffvb1156-2-e \
    --clock-period 5 \
    --reuse-factor 1 \
    --io-type io_parallel
```

**Parameters:**
- `--fpga`: Your target FPGA part number
- `--clock-period`: Target clock period in nanoseconds (5ns = 200MHz)
- `--reuse-factor`: Higher = fewer resources, more latency (1 = maximum parallelism)
- `--io-type`: 
  - `io_parallel`: Fast, more resources
  - `io_stream`: Slower, fewer resources

### Step 4: Run C Simulation

Validate the HLS model matches the Keras model:

```bash
# Run C simulation
python hls4ml_convert.py \
    --convert \
    --csim \
    --weights ./results/quantized_model/best.weights.h5 \
    --num-classes 1 \
    --img-size 256x128 \
    --output-dir hls4ml_yolo
```

This compares HLS4ml predictions with Keras predictions and reports:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Maximum Absolute Error

**Acceptable thresholds:**
- MAE < 0.01: Excellent match
- MAE < 0.1: Acceptable (may need precision tuning)
- MAE > 0.1: Poor match (increase precision)

### Step 5: Synthesize to FPGA

Use Vivado HLS to synthesize the C++ code:

```bash
cd hls4ml_yolo
vivado_hls -f build_prj.tcl
```

Or use the HLS4ml Python API:

```python
import hls4ml
hls_model = hls4ml.converters.load_model('hls4ml_yolo')
hls_model.build(csim=False, synth=True, vsynth=True)
```

**Check synthesis reports:**
- Resource utilization (LUTs, FFs, BRAMs, DSPs)
- Timing (clock period achieved)
- Latency and throughput

## Configuration Tuning

### If Model Doesn't Fit on FPGA

1. **Increase reuse factor**
   ```bash
   --reuse-factor 4  # or 8, 16
   ```

2. **Reduce precision**
   - Edit `create_hls4ml_config()` in `hls4ml_convert.py`
   - Reduce bit-widths (e.g., 16→12, 12→8)

3. **Use streaming IO**
   ```bash
   --io-type io_stream
   ```

4. **Prune the model**
   - Reduce number of channels in `widths` parameter
   - Reduce number of blocks in `depths` parameter

### If Latency is Too High

1. **Decrease reuse factor**
   ```bash
   --reuse-factor 1  # Maximum parallelism
   ```

2. **Use parallel IO**
   ```bash
   --io-type io_parallel
   ```

3. **Increase clock period** (if timing fails)
   ```bash
   --clock-period 10  # 100 MHz instead of 200 MHz
   ```

### If Accuracy is Poor

1. **Increase precision**
   - Edit bit-widths in `create_hls4ml_config()`
   - Especially for first/last layers

2. **Use stable implementations**
   - Softmax: `'Implementation': 'stable'`
   - Sigmoid: `'Implementation': 'lut'`

3. **Re-train with lower quantization**
   - Increase bits in `tinysimov35_keras_quantized.py`

## Resource Estimates

For the small YOLO model (widths=[3,4,8,16,64,128], depths=[1,1,1,1]):

### With 8-bit quantization, ReuseFactor=1:
- **LUTs**: ~50k-100k
- **FFs**: ~30k-60k
- **BRAMs**: ~50-100
- **DSPs**: ~100-200
- **Latency**: ~1-10 μs (depends on clock)

### With 8-bit quantization, ReuseFactor=8:
- **LUTs**: ~10k-20k
- **FFs**: ~5k-10k
- **BRAMs**: ~50-100
- **DSPs**: ~20-40
- **Latency**: ~10-100 μs

*Note: Actual numbers depend on FPGA architecture and HLS tool version*

## Known Limitations

### Operations to Avoid

1. **Dynamic shapes** - All tensor shapes must be known at compile time
2. **Control flow** - No `if` statements or `for` loops with variable bounds
3. **tf.range, tf.meshgrid** - Not supported in HLS4ml
4. **Variable batch size** - Must be fixed (typically 1)

### Expensive Operations

1. **Softmax** - Very resource-intensive
   - Consider approximations or lookup tables
   - Or move to CPU post-processing

2. **Sigmoid** - Resource-intensive
   - Use lookup table: `'Implementation': 'lut'`
   - Or piecewise linear approximation

3. **Division** - Expensive
   - Replace with multiplication by reciprocal where possible

## Troubleshooting

### "Unsupported layer" error
- Check HLS4ml documentation for supported layers
- Some Keras layers may need custom HLS implementation

### "Shape mismatch" error
- Ensure all shapes are static (no `None` dimensions)
- Check that batch size is fixed to 1

### Poor synthesis timing
- Increase clock period
- Increase reuse factor
- Simplify complex operations

### High resource usage
- Increase reuse factor
- Reduce precision
- Reduce model size (fewer channels/layers)

### Accuracy degradation
- Increase precision (more bits)
- Use stable implementations for activations
- Re-train with quantization-aware training

## Testing Workflow

### 1. Validate Quantized Model

First, ensure the quantized model maintains accuracy:

```bash
# Test quantized model
python main_keras.py \
    --test \
    --quantized \
    --input-size 128x256 \
    --save-path ./results/quantized_model \
    --dataset-dir ./Dataset/bionano_cellv2
```

Compare metrics (mAP, precision, recall) with the float32 baseline.

### 2. Validate Static Model

Test that the static model produces similar outputs:

```python
import numpy as np
from nets.tinysimov35_keras_quantized import yolo_v8_s_quantized
from nets.tinysimov35_keras_hls4ml import yolo_v8_s_hls4ml, transfer_weights_to_static

# Load models
quantized = yolo_v8_s_quantized(num_classes=1, img_size=(256, 128))
quantized.load_weights('results/quantized_model/best.weights.h5')

static = yolo_v8_s_hls4ml(num_classes=1, img_h=256, img_w=128)
static = transfer_weights_to_static(quantized, static)

# Test on sample
test_img = np.random.randn(1, 256, 128, 3).astype(np.float32) / 255.0
out_quantized = quantized(test_img, training=False)
out_static = static(test_img)

# Compare
diff = np.abs(out_quantized - out_static)
print(f"Max difference: {np.max(diff)}")
print(f"Mean difference: {np.mean(diff)}")
```

### 3. Validate HLS4ml Model

Run C simulation (as shown in Step 4 above) to ensure HLS implementation matches.

## Next Steps

After successful FPGA synthesis:

1. **Integration**: Integrate the HLS IP core into your FPGA design
2. **Pre/Post-processing**: Implement image preprocessing and NMS on FPGA or CPU
3. **Optimization**: Fine-tune precision and parallelism based on actual resource usage
4. **Validation**: Test on real hardware with your dataset

## References

- [HLS4ml Documentation](https://fastmachinelearning.org/hls4ml/)
- [QKeras Documentation](https://github.com/google/qkeras)
- [Xilinx Vivado HLS User Guide](https://www.xilinx.com/support/documentation/sw_manuals/xilinx2020_2/ug902-vivado-high-level-synthesis.pdf)

## Support

For issues specific to:
- **HLS4ml**: https://github.com/fastmachinelearning/hls4ml/issues
- **QKeras**: https://github.com/google/qkeras/issues
- **This implementation**: Check the code comments and docstrings
