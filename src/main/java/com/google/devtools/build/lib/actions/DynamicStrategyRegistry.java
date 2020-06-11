// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.actions;

import java.util.List;

/** Registry providing access to dynamic spawn strategies for both remote and local modes. */
public interface DynamicStrategyRegistry extends ActionContext {

  /** Indicator for whether a strategy is meant for remote or local branch of dynamic execution. */
  enum DynamicMode {
    REMOTE,
    LOCAL
  }

  /**
   * Returns the spawn strategy implementations that {@linkplain SpawnStrategy#canExec can execute}
   * the given spawn in the order that they were registered for the provided dynamic mode.
   */
  List<SandboxedSpawnStrategy> getDynamicSpawnActionContexts(Spawn spawn, DynamicMode dynamicMode);

  /**
   * Notifies all strategies applying to at least one mnemonic (including the empty all-catch one)
   * in this registry that they are {@link ActionContext#usedContext used}.
   *
   * @param actionContextRegistry a complete registry containing all available action contexts
   */
  void notifyUsedDynamic(ActionContext.ActionContextRegistry actionContextRegistry);
}
