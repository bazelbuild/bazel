#!/bin/bash
#
# Ship the environment to the C++ action
#
set -eu

# Set-up the environment
%{env}

# Call the C++ compiler
echo "gcc not supported now"
exit 1
