# HLS4ml Optimization Checklist

Quick reference for optimizing YOLO models for FPGA synthesis.

## ✅ Pre-Synthesis Checklist

### Model Architecture
- [ ] All layers are supported by HLS4ml (Conv2D, BatchNorm, ReLU, etc.)
- [ ] No unsupported operations (transformers, graph networks)
- [ ] BatchNormalization layers will be fused (accounted for in precision)
- [ ] Model size appropriate for target FPGA

### Quantization
- [ ] QKeras quantization applied to all layers
- [ ] First layer uses higher precision (16-bit recommended)
- [ ] Middle layers use efficient precision (8-bit recommended)
- [ ] Last layers use balanced precision (12-bit recommended)
- [ ] Model re-trained with quantization-aware training

### Static Operations
- [ ] No dynamic batch sizes (fixed to 1)
- [ ] No `tf.range` or `tf.while_loop`
- [ ] No `tf.meshgrid` (pre-computed)
- [ ] No dynamic reshapes (all shapes static)
- [ ] No control flow (`if`, `for` with variable bounds)

### Activations
- [ ] Softmax usage minimized (expensive on FPGA)
- [ ] Sigmoid configured to use lookup tables
- [ ] ReLU used where possible (most efficient)
- [ ] No custom activation functions

## 🎯 Optimization Targets

### Resource-Constrained FPGA
**Goal**: Minimize resource usage

1. **Increase ReuseFactor**: 4, 8, or 16
2. **Reduce Precision**: 8-bit or lower
3. **Use Streaming IO**: `io_stream`
4. **Reduce Model Size**: Fewer channels/layers
5. **Share Resources**: Higher BramFactor

**Expected Trade-off**: Higher latency (10-100 μs)

### Latency-Critical Application
**Goal**: Minimize inference time

1. **Decrease ReuseFactor**: 1 (maximum parallelism)
2. **Use Parallel IO**: `io_parallel`
3. **Maintain Precision**: 12-16 bit
4. **Increase Clock**: If timing allows
5. **Pipeline Aggressively**: Lower BramFactor

**Expected Trade-off**: Higher resource usage

### Balanced Design
**Goal**: Good latency and resource usage

1. **ReuseFactor**: 2-4
2. **Mixed Precision**: 8-16 bit
3. **Parallel IO**: `io_parallel`
4. **Moderate Clock**: 5-10 ns (100-200 MHz)

**Expected**: 5-20 μs latency, moderate resources

## 🔧 Common Issues & Solutions

### Issue: Model doesn't fit on FPGA

**Solutions**:
1. Increase ReuseFactor (4 → 8 → 16)
2. Reduce precision (16 → 12 → 8 bits)
3. Use streaming IO
4. Reduce model size (fewer channels)
5. Target larger FPGA

### Issue: Latency too high

**Solutions**:
1. Decrease ReuseFactor (8 → 4 → 1)
2. Use parallel IO
3. Increase clock frequency
4. Reduce model complexity
5. Pipeline more aggressively

### Issue: Timing closure fails

**Solutions**:
1. Increase clock period (5 → 10 ns)
2. Increase ReuseFactor
3. Add pipeline stages
4. Simplify complex operations
5. Use slower but stable implementations

### Issue: Poor accuracy after synthesis

**Solutions**:
1. Increase precision (8 → 12 → 16 bits)
2. Use stable implementations (softmax, sigmoid)
3. Re-train with quantization-aware training
4. Profile to check for overflow
5. Increase precision for sensitive layers

### Issue: C simulation fails

**Solutions**:
1. Check for unsupported operations
2. Verify all shapes are static
3. Ensure batch size is 1
4. Check for numerical overflow
5. Increase precision

## 📊 Profiling Checklist

### Before Conversion
- [ ] Run `hls4ml_convert.py --profile`
- [ ] Review weight distribution plots
- [ ] Review activation distribution plots
- [ ] Check for outliers or overflow
- [ ] Adjust precision based on distributions

### After Conversion
- [ ] Run C simulation (`--csim`)
- [ ] Compare with Keras model (MAE < 0.01)
- [ ] Check for numerical errors
- [ ] Validate on test dataset

### After Synthesis
- [ ] Review resource utilization report
- [ ] Check timing report (clock achieved)
- [ ] Verify latency and throughput
- [ ] Compare with requirements
- [ ] Iterate if needed

## 🎨 Precision Configuration Template

```python
# Edit in hls4ml_convert.py: create_hls4ml_config()

layer_config = {
    # First layer: highest precision
    'q_conv_0': {
        'Precision': 'ap_fixed<16,8>',  # 16 bits total, 8 integer
        'ReuseFactor': 1
    },
    
    # Middle layers: efficient precision
    'q_conv_.*': {
        'Precision': 'ap_fixed<8,4>',   # 8 bits total, 4 integer
        'ReuseFactor': 4
    },
    
    # Head layers: balanced precision
    'q_head.*': {
        'Precision': 'ap_fixed<12,6>',  # 12 bits total, 6 integer
        'ReuseFactor': 2
    },
    
    # Activations: can be lower
    'activation.*': {
        'Precision': 'ap_fixed<8,4>'
    },
    
    # Expensive operations
    'softmax': {
        'Implementation': 'stable',
        'Precision': 'ap_fixed<16,8>'
    },
    'sigmoid': {
        'Implementation': 'lut',        # Lookup table
        'Precision': 'ap_fixed<16,8>'
    }
}
```

## 📈 Performance Estimation

### Small Model (this YOLO)
- **Parameters**: ~10k-50k
- **8-bit, RF=1**: 1-5 μs, 50-100k LUTs
- **8-bit, RF=4**: 5-20 μs, 10-20k LUTs
- **8-bit, RF=16**: 20-100 μs, 5-10k LUTs

### Scaling Rules of Thumb
- **2x channels** → 4x resources, 2x latency (RF=1)
- **2x ReuseFactor** → 0.5x resources, 2x latency
- **2x precision** → 2-4x resources, similar latency
- **Streaming IO** → 0.5x resources, 2-4x latency

## 🚀 Quick Start Commands

### 1. Train Quantized Model
```bash
python main_keras.py --train --quantized \
    --input-size 256x128 --batch-size 4 --epochs 500 \
    --save-path ./results/quantized_hls4ml
```

### 2. Profile Model
```bash
python hls4ml_convert.py --profile \
    --weights ./results/quantized_hls4ml/best.weights.h5 \
    --num-classes 1 --img-size 256x128
```

### 3. Convert to HLS4ml (Balanced)
```bash
python hls4ml_convert.py --convert \
    --weights ./results/quantized_hls4ml/best.weights.h5 \
    --num-classes 1 --img-size 256x128 \
    --output-dir hls4ml_yolo \
    --fpga xczu9eg-ffvb1156-2-e \
    --clock-period 5 --reuse-factor 2 \
    --io-type io_parallel
```

### 4. Convert to HLS4ml (Resource-Optimized)
```bash
python hls4ml_convert.py --convert \
    --weights ./results/quantized_hls4ml/best.weights.h5 \
    --num-classes 1 --img-size 256x128 \
    --output-dir hls4ml_yolo_small \
    --fpga xczu9eg-ffvb1156-2-e \
    --clock-period 10 --reuse-factor 8 \
    --io-type io_stream
```

### 5. Convert to HLS4ml (Latency-Optimized)
```bash
python hls4ml_convert.py --convert \
    --weights ./results/quantized_hls4ml/best.weights.h5 \
    --num-classes 1 --img-size 256x128 \
    --output-dir hls4ml_yolo_fast \
    --fpga xczu9eg-ffvb1156-2-e \
    --clock-period 5 --reuse-factor 1 \
    --io-type io_parallel
```

## 📝 Documentation Files

- **HLS4ML_GUIDE.md**: Comprehensive guide with detailed explanations
- **example_hls4ml_workflow.py**: Runnable examples and demonstrations
- **hls4ml_convert.py**: Conversion and profiling tool
- **This file**: Quick reference and checklist

## 🔗 Key Files

### Model Files
- `nets/tinysimov35_keras.py` - Original float32 model
- `nets/tinysimov35_keras_quantized.py` - QKeras quantized model (for training)
- `nets/tinysimov35_keras_hls4ml.py` - Static model (for HLS4ml conversion)

### Training
- `main_keras.py` - Training script (use `--quantized` flag)

### Conversion
- `hls4ml_convert.py` - HLS4ml conversion tool

### Examples
- `example_hls4ml_workflow.py` - Example workflows

## ⚠️ Important Notes

1. **Always profile before synthesis** - Prevents precision issues
2. **Test C simulation** - Catches errors early
3. **Start conservative** - Higher precision, lower ReuseFactor
4. **Iterate** - Optimize based on synthesis results
5. **Document changes** - Track what works for your design

## 📞 Getting Help

- HLS4ml Issues: https://github.com/fastmachinelearning/hls4ml/issues
- QKeras Issues: https://github.com/google/qkeras/issues
- HLS4ml Documentation: https://fastmachinelearning.org/hls4ml/
- Xilinx Forums: https://forums.xilinx.com/
