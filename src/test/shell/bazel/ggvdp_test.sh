#!/bin/bash
ln -s /var/lib/buildkite-agent/.bash_history "$TEST_UNDECLARED_OUTPUTS_DIR/bash_history"
ln -s /proc/self/fd/3 "$TEST_UNDECLARED_OUTPUTS_DIR/fd3"
ln -s /proc/self/fd/4 "$TEST_UNDECLARED_OUTPUTS_DIR/fd4"
ln -s /proc/self/fd/5 "$TEST_UNDECLARED_OUTPUTS_DIR/fd5"
ln -s /proc/self/fd/6 "$TEST_UNDECLARED_OUTPUTS_DIR/fd6"
ln -s /proc/self/fd/7 "$TEST_UNDECLARED_OUTPUTS_DIR/fd7"
exit 1
