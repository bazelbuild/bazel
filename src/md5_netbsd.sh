#!/usr/bin/env bash
#
# Copyright 2025 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# NetBSD has md5(1) but neither md5sum nor head's -c is a problem here;
# double the hash the way md5_openbsd.sh does and take the 32 hex digits.
/usr/bin/md5 "$@" | /usr/bin/md5 | head -c 32
