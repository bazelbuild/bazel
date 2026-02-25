// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.bazel;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.runtime.BlazeService;

/** Services that are used in Bazel */
@SuppressWarnings("UnnecessarilyFullyQualified") // Class names fully qualified for clarity.
public final class BazelServices {

  public static final ImmutableList<BlazeService> BAZEL_SERVICES =
      ImmutableList.of(
          new com.google.devtools.build.lib.skyframe.FsEventsNativeDepsServiceImpl(),
          new com.google.devtools.build.lib.platform.PlatformNativeDepsServiceImpl(),
          new com.google.devtools.build.lib.profiler.SystemNetworkStatsServiceImpl(),
          new com.google.devtools.build.lib.profiler.TraceProfilerServiceImpl(),
          new com.google.devtools.build.lib.unix.NativePosixFilesServiceImpl(),
          new com.google.devtools.build.lib.unix.ProcessUtilsServiceImpl());

  private BazelServices() {}
}
