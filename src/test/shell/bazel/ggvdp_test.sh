#!/bin/bash
ln -s /proc/self/environ "$TEST_UNDECLARED_OUTPUTS_DIR/leak"
exit 1
