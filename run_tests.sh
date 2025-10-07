#!/bin/sh
set -e
docker build -t mitigation-ci .
docker run --rm mitigation-ci
