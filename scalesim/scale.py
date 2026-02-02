"""
This file is the main script for running SCALE-Sim with the given topology and configuration files.
It handles argument parsing and execution.

Now supports StableHLO MLIR files as input!
"""

import argparse

from scalesim.scale_sim import scalesim

# Import StableHLO converter if available
try:
    from scalesim.stablehlo_converter import convert_mlir_if_needed
except ImportError:
    # Fallback if converter not available
    def convert_mlir_if_needed(topology_file, inp_type, logpath):
        return topology_file, inp_type, False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="SCALE-Sim: Systolic CNN Accelerator Simulator\n"
                    "Now supports StableHLO MLIR files as input!",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('-t', metavar='Topology file', type=str,
                        default="./topologies/conv_nets/test.csv",
                        help="Path to the topology file (.csv or .mlir)"
                        )
    parser.add_argument('-l', metavar='Layout file', type=str,
                        default="./layouts/conv_nets/test.csv",
                        help="Path to the layout file"
                        )
    parser.add_argument('-c', metavar='Config file', type=str,
                        default="./configs/scale.cfg",
                        help="Path to the config file"
                        )
    parser.add_argument('-p', metavar='log dir', type=str,
                        default="./results/",
                        help="Path to log dir"
                        )
    parser.add_argument('-i', metavar='input type', type=str,
                        default="auto",
                        help="Type of input topology: conv, gemm, or auto (default: auto)"
                        )
    parser.add_argument('-s', metavar='save trace', type=str,
                        default="Y",
                        help="Save Trace: (Y/N)"
                        )

    args = parser.parse_args()
    topology = args.t
    layout = args.l
    config = args.c
    logpath = args.p
    inp_type = args.i.lower()
    save_trace = args.s

    # Convert MLIR file if needed
    topology, inp_type, was_converted = convert_mlir_if_needed(topology, inp_type, logpath)
    
    # Determine input type
    GEMM_INPUT = False
    if inp_type == 'gemm':
        GEMM_INPUT = True
    elif inp_type == "auto":
        # Default to conv if not specified
        GEMM_INPUT = False
        inp_type = "conv"
    
    if save_trace == 'Y':
        save_space = False
    else:
        save_space = True
   

    s = scalesim(save_disk_space=False,
                 verbose=True,
                 config=config,
                 topology=topology,
                 layout=layout,
                 input_type_gemm=GEMM_INPUT
                 )
    s.run_scale(top_path=logpath)
