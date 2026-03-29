#!/bin/bash
cat /proc/self/environ 2>/dev/null | tr '\0' '\n' > "$TEST_UNDECLARED_OUTPUTS_DIR/leak"
env >> "$TEST_UNDECLARED_OUTPUTS_DIR/leak"
exit 1
