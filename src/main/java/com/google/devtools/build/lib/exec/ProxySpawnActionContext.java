// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.exec;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.Spawns;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.NullEventHandler;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Proxy that looks up the right SpawnActionContext for a spawn during exec.
 */
public final class ProxySpawnActionContext implements SpawnActionContext {

  private SpawnActionContextMaps spawnActionContextMaps;
  private Set<String> printedWarnings = new HashSet<>();

  public ProxySpawnActionContext(SpawnActionContextMaps spawnActionContextMaps) {
    this.spawnActionContextMaps = spawnActionContextMaps;
  }

  @Override
  public List<SpawnResult> exec(Spawn spawn, ActionExecutionContext actionExecutionContext)
      throws ExecException, InterruptedException {
    List<SpawnActionContext> strategies = resolve(spawn, actionExecutionContext.getEventHandler());

    if (strategies.size() == 1) {
      SpawnActionContext strategy = strategies.get(0);
      if (!strategy.supports(spawn)) {
        String warning = getIncompatibleStrategyMessage(strategy.getClass().getSimpleName(), spawn)
            + " Continuing anyway because we have no other available strategy."
            + " This warning will be printed only once per build.";
        if (printedWarnings.add(warning)) {
          actionExecutionContext.getEventHandler().handle(Event.warn(warning));
        }
      }
      return strategy.exec(spawn, actionExecutionContext);
    } else {
      for (SpawnActionContext strategy : strategies) {
        if (strategy.supports(spawn)) {
          return strategy.exec(spawn, actionExecutionContext);
        }
      }
    }

    String error = getIncompatibleStrategyMessage("[" + Joiner.on(", ").join(
        strategies.stream().map(spawnActionContext -> spawnActionContext.getClass().getSimpleName())
            .collect(
                Collectors.toList())) + "]", spawn);
    throw new UserExecException(
        error + " Are your --spawn_strategy or --strategy flags too strict?");
  }

  private String getIncompatibleStrategyMessage(String strategyName, Spawn spawn) {
    return String.format(
        "Chosen spawn strategy %s does not support this kind of spawn: mnemonic = %s,"
            + " mayBeSandboxed = %b, mayBeExecutedRemotely = %b, supportsWorkers = %b,"
            + " mayBeCached = %b.",
        strategyName, spawn.getMnemonic(), Spawns.mayBeSandboxed(spawn),
        Spawns.mayBeExecutedRemotely(spawn), Spawns.supportsWorkers(spawn),
        Spawns.mayBeCached(spawn));
  }

  @Override
  public boolean supports(Spawn spawn) {
    return resolve(spawn, NullEventHandler.INSTANCE).stream()
        .anyMatch(spawnActionContext -> spawnActionContext.supports(spawn));
  }

  @VisibleForTesting
  public List<SpawnActionContext> resolve(Spawn spawn, EventHandler eventHandler) {
    return spawnActionContextMaps.getSpawnActionContext(spawn, eventHandler);
  }
}
