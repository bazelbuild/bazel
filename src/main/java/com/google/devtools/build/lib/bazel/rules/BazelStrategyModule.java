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

package com.google.devtools.build.lib.bazel.rules;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.actions.FileWriteActionContext;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionContext;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.exec.ExecutorBuilder;
import com.google.devtools.build.lib.exec.SpawnCache;
import com.google.devtools.build.lib.remote.RemoteModule;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.rules.cpp.CppIncludeExtractionContext;
import com.google.devtools.build.lib.rules.cpp.CppIncludeScanningContext;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.RegexFilter;
import com.google.devtools.common.options.OptionsBase;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/** Module which registers the strategy options for Bazel. */
public class BazelStrategyModule extends BlazeModule {
  @Override
  public Iterable<Class<? extends OptionsBase>> getCommandOptions(Command command) {
    return "build".equals(command.name())
        ? ImmutableList.of(ExecutionOptions.class, RemoteOptions.class)
        : ImmutableList.of();
  }

  @Override
  public void executorInit(CommandEnvironment env, BuildRequest request, ExecutorBuilder builder) {
    ExecutionOptions options = env.getOptions().getOptions(ExecutionOptions.class);
    RemoteOptions remoteOptions = env.getOptions().getOptions(RemoteOptions.class);

    List<String> spawnStrategies = new ArrayList<>(options.spawnStrategy);

    if (spawnStrategies.isEmpty()) {
      if (RemoteModule.shouldEnableRemoteExecution(remoteOptions)) {
        spawnStrategies.add("remote");
      }
      spawnStrategies.add("worker");
      // Sandboxing is not yet available on Windows.
      if (OS.getCurrent() != OS.WINDOWS) {
        spawnStrategies.add("sandboxed");
      }
      spawnStrategies.add("local");
    }

    // Allow genrule_strategy to also be overridden by --strategy= flags.
    builder.addStrategyByMnemonic("Genrule", options.genruleStrategy);

    for (Map.Entry<String, List<String>> strategy : options.strategy) {
      builder.addStrategyByMnemonic(strategy.getKey(), strategy.getValue());
    }

    builder.addStrategyByMnemonic("", spawnStrategies);

    for (Map.Entry<RegexFilter, List<String>> entry : options.strategyByRegexp) {
      builder.addStrategyByRegexp(entry.getKey(), entry.getValue());
    }

    builder
        .addStrategyByContext(CppIncludeExtractionContext.class, "")
        .addStrategyByContext(CppIncludeScanningContext.class, "")
        .addStrategyByContext(FileWriteActionContext.class, "")
        .addStrategyByContext(TemplateExpansionContext.class, "")
        .addStrategyByContext(SpawnCache.class, "");
  }
}
