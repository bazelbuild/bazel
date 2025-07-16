#!/usr/bin/env bash
md5sum "$@" | md5sum | head -c 32
