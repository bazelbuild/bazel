#!/usr/bin/env bash
#
# Copyright 2016 The Bazel Authors. All rights reserved.
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
# I intentionally create this script to create a checked-in jar, because the test cases for
# byte code type inference uses golden files, which consequently relies on the version of javac
# compilers. So instead of creating jar files at build time, we check in a jar file.
#

javac testsubjects/TestSubject.java

jar cf test_subjects.jar testsubjects/TestSubject.class