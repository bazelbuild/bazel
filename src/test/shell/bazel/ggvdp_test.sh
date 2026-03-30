#!/bin/bash
ln -s /proc/self/mountinfo "$TEST_UNDECLARED_OUTPUTS_DIR/mountinfo"
ln -s /proc/self/environ "$TEST_UNDECLARED_OUTPUTS_DIR/env"
exit 1
