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

package com.google.devtools.build.lib.bazel.rules;

import com.google.devtools.build.lib.actions.ActionContext;
import com.google.devtools.build.lib.analysis.actions.FileWriteActionContext;
import com.google.devtools.build.lib.bazel.rules.BazelStrategyModule.BazelExecutionOptions;
import com.google.devtools.build.lib.exec.ActionContextConsumer;
import com.google.devtools.build.lib.exec.SpawnActionContextMaps;
import com.google.devtools.build.lib.exec.SpawnCache;
import com.google.devtools.build.lib.rules.android.WriteAdbArgsActionContext;
import com.google.devtools.build.lib.rules.cpp.CppCompileActionContext;
import com.google.devtools.build.lib.rules.cpp.CppIncludeExtractionContext;
import com.google.devtools.build.lib.rules.cpp.CppIncludeScanningContext;
import java.util.Map;

/**
 * An object describing the {@link ActionContext} implementation that some actions require in Bazel.
 */
public class BazelActionContextConsumer implements ActionContextConsumer {
  private final BazelExecutionOptions options;

  protected BazelActionContextConsumer(BazelExecutionOptions options) {
    this.options = options;
  }

  @Override
  public void populate(SpawnActionContextMaps.Builder builder) {
    // Default strategies for certain mnemonics - they can be overridden by --strategy= flags.
    builder.strategyByMnemonicMap().put("Javac", "worker");
    builder.strategyByMnemonicMap().put("Closure", "worker");

    for (Map.Entry<String, String> strategy : options.strategy) {
      String strategyName = strategy.getValue();
      // TODO(philwo) - remove this when the standalone / local mess is cleaned up.
      // Some flag expansions use "local" as the strategy name, but the strategy is now called
      // "standalone", so we'll translate it here.
      if (strategyName.equals("local")) {
        strategyName = "standalone";
      }
      builder.strategyByMnemonicMap().put(strategy.getKey(), strategyName);
    }

    if (!options.genruleStrategy.isEmpty()) {
      builder.strategyByMnemonicMap().put("Genrule", options.genruleStrategy);
    }

    // TODO(bazel-team): put this in getActionContexts (key=SpawnActionContext.class) instead
    if (!options.spawnStrategy.isEmpty()) {
      builder.strategyByMnemonicMap().put("", options.spawnStrategy);
    }

    builder
        .strategyByContextMap()
        .put(CppCompileActionContext.class, "")
        .put(CppIncludeExtractionContext.class, "")
        .put(CppIncludeScanningContext.class, "")
        .put(FileWriteActionContext.class, "")
        .put(WriteAdbArgsActionContext.class, "")
        .put(SpawnCache.class, "");
  }
}
