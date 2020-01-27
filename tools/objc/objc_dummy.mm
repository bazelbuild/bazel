// Copyright 2015 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*
 * Include this file in a target if it requires some source but you don't have
 * any.
 *
 * ios_extension_binary rules only generate a static library Xcode target, and
 * the ios_extension will generate an actual bundling Xcode target. Application
 * and app extension targets need at least one source file for Xcode to be
 * happy, so we can add this file for them.
 */

static int dummy __attribute__((unused,used)) = 0;
