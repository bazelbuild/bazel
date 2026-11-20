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
package com.google.devtools.build.lib.bazel.commands;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.bazel.bzlmod.BazelDepGraphValue;
import com.google.devtools.build.lib.bazel.bzlmod.Module;
import com.google.devtools.build.lib.bazel.bzlmod.modcommand.ModuleRegistrations.RegisteredTarget;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.pkgcache.FilteringPolicy;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.skyframe.toolchains.RegisteredExecutionPlatformsFunction;

/**
 * Collects the execution platforms that modules register, for {@code mod show_execution_platforms}.
 */
final class ExecutionPlatformCollector extends RegistrationCollector {

  ExecutionPlatformCollector(
      SkyframeExecutor skyframeExecutor,
      int threads,
      ExtendedEventHandler eventHandler,
      BazelDepGraphValue depGraph) {
    super(skyframeExecutor, threads, eventHandler, depGraph);
  }

  @Override
  protected ImmutableList<String> patterns(Module module) {
    return module.getExecutionPlatformsToRegister();
  }

  @Override
  protected FilteringPolicy filteringPolicy() {
    return RegisteredExecutionPlatformsFunction.HAS_PLATFORM_INFO;
  }

  @Override
  protected RegisteredTarget registeredTarget(Label label) {
    return new RegisteredTarget.Platform(label);
  }
}
