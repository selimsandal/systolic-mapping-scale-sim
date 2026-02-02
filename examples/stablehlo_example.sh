#!/bin/bash
# Example: Running SCALE-Sim with StableHLO MLIR files

echo "================================================"
echo "StableHLO + SCALE-Sim Example"
echo "================================================"
echo ""

# Navigate to SCALE-Sim directory
cd "$(dirname "$0")/.." || exit 1

# Example 1: Direct MLIR input with TPUv4 config
echo "Example 1: ResNet-50 on TPUv4"
echo "----------------------------------------------"
python3 -m scalesim.scale \
  -t ../stablehlo/stablehlo_parse_min/resnet_50_inference_step_4_devices_1024_batch.mlir \
  -c ./configs/tpuv4.cfg \
  -p ./results/example_resnet50_tpuv4 \
  -i auto

echo ""
echo "Results saved to: ./results/example_resnet50_tpuv4/"
echo ""

# Example 2: Pre-convert MLIR to CSV
echo "Example 2: Pre-convert MLIR to CSV"
echo "----------------------------------------------"
python3 -m scalesim.stablehlo_converter \
  ../stablehlo/stablehlo_parse_min/resnet_50_inference_step_4_devices_1024_batch.mlir \
  -o ./results/resnet50_converted.csv

echo ""
echo "Converted topology: ./results/resnet50_converted.csv"
echo ""

# Example 3: Run with different config
echo "Example 3: Same model on Eyeriss architecture"
echo "----------------------------------------------"
python3 -m scalesim.scale \
  -t ./results/resnet50_converted.csv \
  -c ./configs/eyeriss.cfg \
  -p ./results/example_resnet50_eyeriss \
  -i gemm

echo ""
echo "Results saved to: ./results/example_resnet50_eyeriss/"
echo ""

echo "================================================"
echo "Examples completed!"
echo "================================================"
echo ""
echo "Compare results:"
echo "  - TPUv4:   ./results/example_resnet50_tpuv4/COMPUTE_REPORT.csv"
echo "  - Eyeriss: ./results/example_resnet50_eyeriss/COMPUTE_REPORT.csv"



