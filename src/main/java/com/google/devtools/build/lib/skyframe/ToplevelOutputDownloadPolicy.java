// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import com.google.common.collect.ImmutableList;

/**
 * An invocation's policy for downloading top-level outputs that are only available as remote
 * metadata.
 *
 * <p>Since the policy determines which outputs the completion functions materialize in the local
 * filesystem, a change to it (e.g. a different download mode or a switch between {@code bazel
 * build} and {@code bazel run}) must invalidate completion functions, which happens by injecting
 * it as {@link PrecomputedValue#TOPLEVEL_OUTPUT_DOWNLOAD_POLICY}.
 */
public record ToplevelOutputDownloadPolicy(
    String outputsMode, String commandName, ImmutableList<String> downloadRegexes) {}
