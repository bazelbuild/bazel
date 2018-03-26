// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.exec.ActionContextConsumer;
import com.google.devtools.build.lib.exec.SpawnActionContextMaps;

/**
 * {@link ActionContextConsumer} that requests the action contexts necessary for standalone
 * execution.
 */
public class StandaloneActionContextConsumer implements ActionContextConsumer {

  @Override
  public void populate(SpawnActionContextMaps.Builder builder) {
    // This makes the "sandboxed" strategy the default Spawn strategy, unless it is overridden by a
    // later BlazeModule.
    builder.strategyByMnemonicMap().put("", "standalone");

    // This makes the "standalone" strategy available via --spawn_strategy=standalone, but it is not
    // necessarily the default.
    builder.strategyByContextMap().put(SpawnActionContext.class, "standalone");
  }
}
