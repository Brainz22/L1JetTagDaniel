#!/bin/bash

# Base path to Vivado installations
VIVADO_DIR="/data/Xilinx/Vivado"   # <-- adjust if your Vivado dir is elsewhere

echo "Available Vivado versions:"
ls -1 "$VIVADO_DIR"

# Ask user to choose a version
#read -p "Enter Vivado version: " VERSION
VERSION="2021.2"
# Check if folder exists
if [ ! -d "$VIVADO_DIR/$VERSION" ]; then
    echo "Error: Version $VERSION not found in $VIVADO_DIR"
    exit 1
elif [ -d "$VIVADO_DIR/$VERSION" ]; then
    echo
    echo "Chose VIVADO Version $VERSION."
    echo
fi

# Source environment
SETTINGS="$VIVADO_DIR/$VERSION/settings64.sh"
if [ ! -f "$SETTINGS" ]; then
    echo "Error: settings64.sh not found in $VIVADO_DIR/$VERSION"
    exit 1
fi

echo "Sourcing $SETTINGS"
source "$SETTINGS"

# Try running vitis_hls first, fallback to vivado_hls
if command -v vivado_hls &> /dev/null; then
    cd qkmodel/hls4ml_prj/
    echo
    echo "1) Launching vivado_hls from directory $PWD"
    echo

    vivado_hls -f build_prj.tcl

elif command -v vitis_hls &> /dev/null; then
    cd qkmodel/hls4ml_prj/
    echo
    echo " \n 2) Launching vitis_hls from directory $PWD \n"
    echo

    vitis_hls -f build_prj.tcl

else
    echo
    echo "3) Neither vitis_hls nor vivado_hls found in PATH after sourcing."
    echo
    exit 1
fi