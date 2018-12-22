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
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.NullEventHandler;
import java.util.List;

/** Proxy that looks up the right SpawnActionContext for a spawn during exec. */
public final class ProxySpawnActionContext implements SpawnActionContext {

  private SpawnActionContextMaps spawnActionContextMaps;

  public ProxySpawnActionContext(SpawnActionContextMaps spawnActionContextMaps) {
    this.spawnActionContextMaps = spawnActionContextMaps;
  }

  @Override
  public List<SpawnResult> exec(Spawn spawn, ActionExecutionContext actionExecutionContext)
      throws ExecException, InterruptedException {
    List<SpawnActionContext> strategies = resolve(spawn, actionExecutionContext.getEventHandler());

    for (SpawnActionContext strategy : strategies) {
      if (strategy.supports(spawn)) {
        return strategy.exec(spawn, actionExecutionContext);
      }
    }

    throw new UserExecException(String.format(
        "No usable spawn strategy found for spawn with mnemonic %s. Are your --spawn_strategy or --strategy flags too strict?",
        spawn.getMnemonic()));

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
