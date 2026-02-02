"""
StableHLO to SCALE-Sim Converter

This module converts StableHLO MLIR operations into SCALE-Sim topology format.
It parses StableHLO operations and translates them into either convolution or GEMM formats
that SCALE-Sim can simulate.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import math

# Try to import the StableHLO parser (now local to scalesim)
try:
    from scalesim.stablehlo_parser import StableHLOParser, OpInfo
    STABLEHLO_AVAILABLE = True
except ImportError as e:
    print(f"Warning: StableHLO parser not available: {e}")
    print("Note: JAX/JAXlib are required for MLIR parsing. Install with: pip install jax jaxlib")
    STABLEHLO_AVAILABLE = False
    StableHLOParser = None
    OpInfo = None


class StableHLOConverter:
    """
    Converter class that translates StableHLO operations to SCALE-Sim topology format.
    
    Supports:
    - stablehlo.convolution -> Conv topology format
    - stablehlo.dot_general -> GEMM topology format
    - stablehlo.dot -> GEMM topology format
    """
    
    def __init__(self, mlir_file: str, verbose: bool = True):
        """
        Initialize the converter with a StableHLO MLIR file.
        
        Args:
            mlir_file: Path to the .mlir file containing StableHLO operations
            verbose: Whether to print conversion progress
        """
        if not STABLEHLO_AVAILABLE:
            raise RuntimeError(
                "StableHLO parser is not available. Please install jax/jaxlib "
                "and ensure stablehlo_parse_min is accessible."
            )
        
        self.mlir_file = mlir_file
        self.verbose = verbose
        self.parser = StableHLOParser(mlir_path=str(mlir_file))
        self.ops = self.parser.get_ops_list()
        
        if self.verbose:
            print(f"Loaded {len(self.ops)} operations from {mlir_file}")
    
    def _convert_convolution_to_topology(self, op: OpInfo, op_idx: int, as_gemm: bool = False) -> Optional[List]:
        """
        Convert a StableHLO convolution operation to SCALE-Sim topology format.
        
        SCALE-Sim Conv format:
        [layer_name, ifmap_h, ifmap_w, filter_h, filter_w, num_ch, num_filt, stride_h, stride_w, N, M]
        
        SCALE-Sim GEMM format (for mixed workloads):
        [layer_name, M, K, 1, K, 1, N, 1, 1, N_sparse, M_sparse]
        where M = ofmap_h * ofmap_w, K = filter_h * filter_w * channels, N = num_filters
        
        Args:
            op: OpInfo object containing the convolution operation
            op_idx: Index of the operation for naming
            as_gemm: If True, convert to GEMM format for mixed workloads
            
        Returns:
            List in SCALE-Sim topology format, or None if conversion fails
        """
        if len(op.input_types) < 2 or len(op.output_types) < 1:
            if self.verbose:
                print(f"Warning: Convolution op {op_idx} has insufficient inputs/outputs")
            return None
        
        # Get input and filter shapes
        input_shape, _ = op.input_types[0]  # [batch, in_channels, height, width] or similar
        filter_shape, _ = op.input_types[1]  # [out_channels, in_channels, kh, kw] or similar
        output_shape, _ = op.output_types[0]
        
        # Parse dimension numbers if available
        dim_numbers = op.extra.get('dimension_numbers', {})
        
        if not dim_numbers:
            # Default assumption: [N, C, H, W] format for input, [OC, IC, KH, KW] for filter
            if len(input_shape) == 4 and len(filter_shape) == 4:
                batch = input_shape[0]
                in_channels = input_shape[1]
                ifmap_h = input_shape[2]
                ifmap_w = input_shape[3]
                
                out_channels = filter_shape[0]
                filter_h = filter_shape[2]
                filter_w = filter_shape[3]
            else:
                if self.verbose:
                    print(f"Warning: Unexpected convolution shapes at op {op_idx}")
                return None
        else:
            # Use dimension numbers to properly index
            lhs_dims = dim_numbers.get('lhs', {})
            rhs_dims = dim_numbers.get('rhs', {})
            
            # Extract from lhs (input)
            batch_dim = lhs_dims.get('batch_dimension', 0)
            feature_dim = lhs_dims.get('feature_dimension', 1)
            spatial_dims = [i for i in range(len(input_shape)) 
                           if i not in [batch_dim, feature_dim]]
            
            batch = input_shape[batch_dim]
            in_channels = input_shape[feature_dim]
            
            if len(spatial_dims) >= 2:
                ifmap_h = input_shape[spatial_dims[0]]
                ifmap_w = input_shape[spatial_dims[1]]
            elif len(spatial_dims) == 1:
                ifmap_h = input_shape[spatial_dims[0]]
                ifmap_w = 1
            else:
                ifmap_h = ifmap_w = 1
            
            # Extract from rhs (filter)
            out_feature_dim = rhs_dims.get('output_feature_dimension', 0)
            in_feature_dim = rhs_dims.get('input_feature_dimension', 1)
            filter_spatial_dims = [i for i in range(len(filter_shape))
                                   if i not in [out_feature_dim, in_feature_dim]]
            
            out_channels = filter_shape[out_feature_dim]
            
            if len(filter_spatial_dims) >= 2:
                filter_h = filter_shape[filter_spatial_dims[0]]
                filter_w = filter_shape[filter_spatial_dims[1]]
            elif len(filter_spatial_dims) == 1:
                filter_h = filter_shape[filter_spatial_dims[0]]
                filter_w = 1
            else:
                filter_h = filter_w = 1
        
        # Calculate stride (approximate from output shape)
        # output_h = (input_h - filter_h) / stride_h + 1
        if len(output_shape) >= 3:
            output_h = output_shape[-2] if len(output_shape) == 4 else output_shape[-1]
            stride_h = max(1, (ifmap_h - filter_h) // (output_h - 1)) if output_h > 1 else 1
            
            if len(output_shape) == 4:
                output_w = output_shape[-1]
                stride_w = max(1, (ifmap_w - filter_w) // (output_w - 1)) if output_w > 1 else 1
            else:
                stride_w = 1
        else:
            stride_h = stride_w = 1
        
        layer_name = f"conv_{op_idx}"
        
        # Sparsity ratio (default 1:1, meaning no sparsity)
        N_sparse, M_sparse = 1, 1
        
        if as_gemm:
            # Convert to GEMM format for mixed workloads
            # M = ofmap_h * ofmap_w (number of output pixels per channel)
            # K = filter_h * filter_w * in_channels (filter volume)
            # N = out_channels (number of filters)
            M = int(output_h * output_w) if len(output_shape) >= 3 else 1
            K = int(filter_h * filter_w * in_channels)
            N = int(out_channels)
            
            topology_entry = [
                layer_name,
                M,
                K,
                1,
                K,
                1,
                N,
                1,
                1,
                N_sparse,
                M_sparse
            ]
            
            if self.verbose:
                print(f"  Converted {op.op_name} -> {layer_name} (as GEMM): "
                      f"M={M}, N={N}, K={K} "
                      f"(from conv: ifmap={ifmap_h}x{ifmap_w}, filter={filter_h}x{filter_w}, ch={in_channels}→{out_channels})")
        else:
            # Standard convolution format
            topology_entry = [
                layer_name,
                int(ifmap_h),
                int(ifmap_w),
                int(filter_h),
                int(filter_w),
                int(in_channels),
                int(out_channels),
                int(stride_h),
                int(stride_w),
                N_sparse,
                M_sparse
            ]
            
            if self.verbose:
                print(f"  Converted {op.op_name} -> {layer_name}: "
                      f"ifmap={ifmap_h}x{ifmap_w}, filter={filter_h}x{filter_w}, "
                      f"ch={in_channels}, filters={out_channels}, stride={stride_h}x{stride_w}")
        
        return topology_entry
    
    def _convert_dot_general_to_gemm(self, op: OpInfo, op_idx: int) -> Optional[List]:
        """
        Convert a StableHLO dot_general operation to SCALE-Sim GEMM format.
        
        SCALE-Sim GEMM format:
        [layer_name, M, K, 1, K, 1, N, 1, 1, N_sparsity, M_sparsity]
        
        Simplified format for topology file (MNK):
        [layer_name, M, N, K, N_sparsity:M_sparsity]
        
        Args:
            op: OpInfo object containing the dot_general operation
            op_idx: Index of the operation for naming
            
        Returns:
            List in SCALE-Sim GEMM topology format, or None if conversion fails
        """
        if len(op.input_types) < 2 or len(op.output_types) < 1:
            if self.verbose:
                print(f"Warning: Dot_general op {op_idx} has insufficient inputs/outputs")
            return None
        
        lhs_shape, _ = op.input_types[0]
        rhs_shape, _ = op.input_types[1]
        output_shape, _ = op.output_types[0]
        
        # Get dimension information from extra
        dims = op.extra.get('dims', {})
        lhs_contracting = dims.get('lhs', [])
        rhs_contracting = dims.get('rhs', [])
        batch_dims = dims.get('batch', [])
        
        # Calculate M, N, K for GEMM
        # M: non-contracting dimensions from lhs
        # N: non-contracting dimensions from rhs
        # K: contracting dimensions
        
        # Calculate K (contracting dimension size)
        K = 1
        for dim_idx in lhs_contracting:
            if dim_idx < len(lhs_shape):
                K *= lhs_shape[dim_idx]
        
        # Calculate M (batch * non-contracting from lhs)
        lhs_batch_size = 1
        for batch_pair in batch_dims:
            lhs_batch_idx = batch_pair[0]
            if lhs_batch_idx < len(lhs_shape):
                lhs_batch_size *= lhs_shape[lhs_batch_idx]
        
        M = lhs_batch_size
        for dim_idx in range(len(lhs_shape)):
            if dim_idx not in lhs_contracting and dim_idx not in [b[0] for b in batch_dims]:
                M *= lhs_shape[dim_idx]
        
        # Calculate N (batch * non-contracting from rhs)
        rhs_batch_size = 1
        for batch_pair in batch_dims:
            rhs_batch_idx = batch_pair[1]
            if rhs_batch_idx < len(rhs_shape):
                rhs_batch_size *= rhs_shape[rhs_batch_idx]
        
        N = rhs_batch_size
        for dim_idx in range(len(rhs_shape)):
            if dim_idx not in rhs_contracting and dim_idx not in [b[1] for b in batch_dims]:
                N *= rhs_shape[dim_idx]
        
        # Ensure we have valid dimensions
        M = max(1, int(M))
        N = max(1, int(N))
        K = max(1, int(K))
        
        layer_name = f"gemm_{op_idx}"
        
        # Sparsity ratio (default 1:1)
        N_sparse, M_sparse = 1, 1
        
        # Return in the internal format used by SCALE-Sim's GEMM loader
        # Format: [name, M, K, 1, K, 1, N, 1, 1, N_sparse, M_sparse]
        topology_entry = [
            layer_name,
            M,
            K,
            1,
            K,
            1,
            N,
            1,
            1,
            N_sparse,
            M_sparse
        ]
        
        if self.verbose:
            print(f"  Converted {op.op_name} -> {layer_name}: M={M}, N={N}, K={K}")
        
        return topology_entry
    
    def _convert_dot_to_gemm(self, op: OpInfo, op_idx: int) -> Optional[List]:
        """
        Convert a StableHLO dot operation to SCALE-Sim GEMM format.
        
        For basic dot (matrix multiplication), assume:
        - lhs: [M, K]
        - rhs: [K, N]
        - output: [M, N]
        
        Args:
            op: OpInfo object containing the dot operation
            op_idx: Index of the operation for naming
            
        Returns:
            List in SCALE-Sim GEMM topology format, or None if conversion fails
        """
        if len(op.input_types) < 2 or len(op.output_types) < 1:
            if self.verbose:
                print(f"Warning: Dot op {op_idx} has insufficient inputs/outputs")
            return None
        
        lhs_shape, _ = op.input_types[0]
        rhs_shape, _ = op.input_types[1]
        
        # For simple matrix multiplication: [M, K] x [K, N] -> [M, N]
        if len(lhs_shape) >= 2 and len(rhs_shape) >= 2:
            M = int(lhs_shape[-2]) if len(lhs_shape) >= 2 else 1
            K = int(lhs_shape[-1])
            N = int(rhs_shape[-1])
            
            # Handle batch dimensions
            batch_size = 1
            for dim in lhs_shape[:-2]:
                batch_size *= dim
            M *= batch_size
        else:
            if self.verbose:
                print(f"Warning: Unexpected dot shapes at op {op_idx}")
            return None
        
        layer_name = f"gemm_{op_idx}"
        
        # Sparsity ratio (default 1:1)
        N_sparse, M_sparse = 1, 1
        
        topology_entry = [
            layer_name,
            M,
            K,
            1,
            K,
            1,
            N,
            1,
            1,
            N_sparse,
            M_sparse
        ]
        
        if self.verbose:
            print(f"  Converted {op.op_name} -> {layer_name}: M={M}, N={N}, K={K}")
        
        return topology_entry
    
    def convert_to_topology(self) -> Tuple[List[List], str]:
        """
        Convert all StableHLO operations to SCALE-Sim topology format.
        
        For mixed workloads (conv + matmul), all operations are converted to GEMM format
        since SCALE-Sim can handle convolutions as GEMM operations (im2col).
        
        Returns:
            Tuple of (topology_entries, input_type) where:
            - topology_entries: List of topology entries
            - input_type: Either "conv" or "gemm" indicating the format used
        """
        # First pass: count operation types to determine format
        conv_count = 0
        gemm_count = 0
        
        for op in self.ops:
            if "convolution" in op.op_name.lower():
                conv_count += 1
            elif "dot_general" in op.op_name.lower() or op.op_name == "stablehlo.dot":
                gemm_count += 1
        
        # Determine format: if we have both conv and gemm, use gemm format for all
        use_gemm_format = (gemm_count > 0)
        input_type = "gemm" if use_gemm_format else "conv"
        
        if self.verbose:
            print(f"\nConverting {len(self.ops)} operations to SCALE-Sim topology...")
            if conv_count > 0 and gemm_count > 0:
                print(f"  → Detected mixed workload ({conv_count} conv + {gemm_count} matmul)")
                print(f"  → Using GEMM format for all layers (convs converted to GEMM via im2col)")
        
        # Second pass: convert operations with the determined format
        topology_entries = []
        
        for idx, op in enumerate(self.ops):
            entry = None
            
            if "convolution" in op.op_name.lower():
                entry = self._convert_convolution_to_topology(op, idx, as_gemm=use_gemm_format)
                if entry:
                    pass  # Already counted
            
            elif "dot_general" in op.op_name.lower():
                entry = self._convert_dot_general_to_gemm(op, idx)
                if entry:
                    pass  # Already counted
            
            elif op.op_name == "stablehlo.dot":
                entry = self._convert_dot_to_gemm(op, idx)
                if entry:
                    pass  # Already counted
            
            else:
                if self.verbose:
                    print(f"  Skipping unsupported op: {op.op_name}")
            
            if entry:
                topology_entries.append(entry)
        
        if self.verbose:
            print(f"\nConversion complete:")
            print(f"  Total operations: {len(self.ops)}")
            print(f"  Converted convolutions: {conv_count}")
            print(f"  Converted GEMMs: {gemm_count}")
            print(f"  Output format: {input_type}")
        
        return topology_entries, input_type
    
    def save_to_csv(self, output_path: str, input_type: str = None) -> str:
        """
        Convert StableHLO operations and save to a SCALE-Sim topology CSV file.
        
        Args:
            output_path: Path where the CSV file should be saved
            input_type: Override the auto-detected input type ("conv" or "gemm")
            
        Returns:
            The input type that was used
        """
        topology_entries, detected_type = self.convert_to_topology()
        
        # Use provided input_type or the detected one
        final_input_type = input_type if input_type else detected_type
        
        # Create output directory if needed
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write CSV file
        with open(output_path, 'w') as f:
            if final_input_type == "conv":
                # Conv format header
                f.write("Layer name, IFMAP Height, IFMAP Width, Filter Height, Filter Width, "
                       "Channels, Num Filter, Strides,\n")
                
                for entry in topology_entries:
                    # entry: [name, ifmap_h, ifmap_w, filter_h, filter_w, ch, num_filt, stride_h, stride_w, N, M]
                    # CSV format needs stride only once (stride_h is used)
                    line = f"{entry[0]}, {entry[1]}, {entry[2]}, {entry[3]}, {entry[4]}, " \
                           f"{entry[5]}, {entry[6]}, {entry[7]},\n"
                    f.write(line)
            else:
                # GEMM format header
                f.write("Layer,M,N,K,\n")
                
                for entry in topology_entries:
                    # entry: [name, M, K, 1, K, 1, N, 1, 1, N_sparse, M_sparse]
                    # Extract M, N, K from the entry
                    name = entry[0]
                    M = entry[1]
                    N = entry[6]
                    K = entry[2]
                    line = f"{name},{M},{N},{K},\n"
                    f.write(line)
        
        if self.verbose:
            print(f"\nTopology saved to: {output_path}")
            print(f"Format: {final_input_type}")
        
        return final_input_type


def convert_stablehlo_to_topology(mlir_file: str, output_csv: str = None, 
                                   verbose: bool = False) -> Tuple[str, str]:
    """
    Convenience function to convert a StableHLO MLIR file to SCALE-Sim topology CSV.
    
    Args:
        mlir_file: Path to the .mlir file
        output_csv: Path for the output CSV (if None, generates from input name)
        verbose: Whether to print conversion progress
        
    Returns:
        Tuple of (output_csv_path, input_type)
    """
    if not STABLEHLO_AVAILABLE:
        raise RuntimeError(
            "StableHLO parser is not available. Please install jax/jaxlib:\n"
            "  pip install jax jaxlib"
        )
    
    # Generate output path if not provided
    if output_csv is None:
        mlir_path = Path(mlir_file)
        output_csv = str(mlir_path.parent / f"{mlir_path.stem}_topology.csv")
    
    # Convert and save
    converter = StableHLOConverter(mlir_file, verbose=verbose)
    input_type = converter.save_to_csv(output_csv)
    
    return output_csv, input_type


def convert_mlir_if_needed(topology_file: str, inp_type: str, logpath: str) -> Tuple[str, str, bool]:
    """
    Check if the topology file is a .mlir file and convert it if needed.
    
    This function is used by SCALE-Sim's main entry point to automatically
    detect and convert MLIR files to topology CSV format.
    
    Args:
        topology_file: Path to the topology or MLIR file
        inp_type: Input type (conv/gemm/auto)
        logpath: Directory for logs and converted files
        
    Returns:
        Tuple of (topology_csv_path, input_type, is_converted)
        - topology_csv_path: Path to the topology CSV (converted or original)
        - input_type: Final input type to use (conv/gemm)
        - is_converted: True if conversion was performed
    """
    topology_path = Path(topology_file)
    
    # Check if it's a .mlir file
    if topology_path.suffix.lower() == '.mlir':
        if not STABLEHLO_AVAILABLE:
            print("ERROR: Cannot process .mlir files - StableHLO parser not available")
            print("Please install required dependencies:")
            print("  pip install jax jaxlib")
            import sys
            sys.exit(1)
        
        print(f"\nDetected StableHLO MLIR file: {topology_file}")
        print("Converting to SCALE-Sim topology format...\n")
        
        # Generate output CSV path in the log directory
        output_csv = Path(logpath) / f"{topology_path.stem}_converted_topology.csv"
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert the MLIR file
        converted_csv, detected_type = convert_stablehlo_to_topology(
            str(topology_file),
            str(output_csv),
            verbose=False
        )
        
        # Use detected type if input type is auto or matches
        if inp_type == "auto":
            final_type = detected_type
        else:
            final_type = inp_type
            print(f"\nNote: Using user-specified input type '{inp_type}' "
                  f"(auto-detected: '{detected_type}')")
        
        print(f"\nConverted topology saved to: {converted_csv}")
        return converted_csv, final_type, True
    
    # Not a MLIR file, return as-is
    return topology_file, inp_type, False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert StableHLO MLIR to SCALE-Sim topology CSV")
    parser.add_argument("mlir_file", type=str, help="Path to the .mlir file")
    parser.add_argument("-o", "--output", type=str, default=None,
                       help="Output CSV file path (default: <input>_topology.csv)")
    parser.add_argument("-q", "--quiet", action="store_true",
                       help="Suppress verbose output")
    
    args = parser.parse_args()
    
    try:
        output_csv, input_type = convert_stablehlo_to_topology(
            args.mlir_file,
            args.output,
            verbose=not args.quiet
        )
        print(f"\nSuccess! Use with SCALE-Sim:")
        print(f"  python3 -m scalesim.scale -t {output_csv} -i {input_type} -c <config_file>")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

