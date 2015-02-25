// Copyright 2014 Google Inc. All rights reserved.
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
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;

import java.util.List;

/**
 * The main class.
 */
public final class BazelMain {
  private static final List<Class<? extends BlazeModule>> BAZEL_MODULES = ImmutableList.of(
      com.google.devtools.build.lib.bazel.BazelShutdownLoggerModule.class,
      com.google.devtools.build.lib.bazel.BazelWorkspaceStatusModule.class,
      com.google.devtools.build.lib.bazel.BazelDiffAwarenessModule.class,
      com.google.devtools.build.lib.bazel.BazelRepositoryModule.class,
      com.google.devtools.build.lib.bazel.rules.BazelRulesModule.class,
      com.google.devtools.build.lib.standalone.StandaloneModule.class,
      com.google.devtools.build.lib.runtime.BuildSummaryStatsModule.class,
      com.google.devtools.build.lib.webstatusserver.WebStatusServerModule.class
  );

  public static void main(String[] args) {
    BlazeRuntime.main(BAZEL_MODULES, args);
  }
}
