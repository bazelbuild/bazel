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
package com.google.devtools.build.lib.standalone;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionContextConsumer;
import com.google.devtools.build.lib.actions.Executor.ActionContext;
import com.google.devtools.build.lib.actions.SpawnActionContext;

import java.util.Map;

/**
 * {@link ActionContextConsumer} that requests the action contexts necessary for standalone
 * execution.
 */
public class StandaloneContextConsumer implements ActionContextConsumer {

  @Override
  public Map<String, String> getSpawnActionContexts() {
    return ImmutableMap.of();
  }

  @Override
  public Map<Class<? extends ActionContext>, String> getActionContexts() {
    return ImmutableMap.<Class<? extends ActionContext>, String>builder()
        .put(SpawnActionContext.class, "standalone")
        .build();
  }

}
