#!/usr/bin/env bash
/sbin/md5 "$@" | /sbin/md5 | head -c 32
