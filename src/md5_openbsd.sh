#!/usr/bin/env bash
# We avoid using the `head` tool's `-c` option, since it does not exist
# on OpenBSD.
/bin/md5 "$@" | /bin/md5 | dd bs=32 count=1
