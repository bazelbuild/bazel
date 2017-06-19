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

package com.google.devtools.build.lib.sandbox;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.actions.ActionContext;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.exec.ActionContextConsumer;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.util.OS;

/**
 * {@link ActionContextConsumer} that requests the action contexts necessary for sandboxed
 * execution.
 */
final class SandboxActionContextConsumer implements ActionContextConsumer {

  private final ImmutableMultimap<Class<? extends ActionContext>, String> contexts;
  private final ImmutableMap<String, String> spawnContexts;

  public SandboxActionContextConsumer(CommandEnvironment cmdEnv) {
    ImmutableMultimap.Builder<Class<? extends ActionContext>, String> contexts =
        ImmutableMultimap.builder();
    ImmutableMap.Builder<String, String> spawnContexts = ImmutableMap.builder();

    if ((OS.getCurrent() == OS.LINUX && LinuxSandboxedStrategy.isSupported(cmdEnv))
        || (OS.getCurrent() == OS.DARWIN && DarwinSandboxRunner.isSupported(cmdEnv))
        || (OS.isPosixCompatible() && ProcessWrapperSandboxedStrategy.isSupported(cmdEnv))) {
      // This makes the "sandboxed" strategy available via --spawn_strategy=sandboxed,
      // but it is not necessarily the default.
      contexts.put(SpawnActionContext.class, "sandboxed");

      // This makes the "sandboxed" strategy the default Spawn strategy, unless it is
      // overridden by a later BlazeModule.
      spawnContexts.put("", "sandboxed");
    }

    this.contexts = contexts.build();
    this.spawnContexts = spawnContexts.build();
  }

  @Override
  public ImmutableMap<String, String> getSpawnActionContexts() {
    return spawnContexts;
  }

  @Override
  public Multimap<Class<? extends ActionContext>, String> getActionContexts() {
    return contexts;
  }
}
