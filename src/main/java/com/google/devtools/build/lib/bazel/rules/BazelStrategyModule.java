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
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.exec.ModuleActionContextRegistry;
import com.google.devtools.build.lib.exec.SpawnCache;
import com.google.devtools.build.lib.exec.SpawnStrategyRegistry;
import com.google.devtools.build.lib.remote.RemoteModule;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.rules.cpp.CppIncludeExtractionContext;
import com.google.devtools.build.lib.rules.cpp.CppIncludeScanningContext;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.RegexFilter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsClass;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/** Module which registers the strategy options for Bazel. */
public class BazelStrategyModule extends BlazeModule {
  /** Strategy options that only exist in Bazel. */
  @OptionsClass
  public abstract static class Options extends OptionsBase {
    @Option(
        name = "file_write_strategy",
        defaultValue = "local",
        documentationCategory = OptionDocumentationCategory.EXECUTION_STRATEGY,
        effectTags = {OptionEffectTag.EXECUTION},
        help =
            "Specifies which strategy to use for file write actions such as ctx.actions.write."
                + " 'local' writes the file to disk. 'remote' requires a disk or remote cache and"
                + " stores the contents there, recording them as a remote output when building"
                + " without the bytes, so that the file is only written to disk if it is needed by"
                + " a local action or requested via --remote_download_outputs or"
                + " --remote_download_regex. If there is no disk cache and Bazel isn't allowed to"
                + " upload to the remote cache, the file is only kept off disk if the remote cache"
                + " already contains its contents.")
    public abstract String getFileWriteStrategy();

    @Option(
        name = "template_expansion_strategy",
        defaultValue = "local",
        documentationCategory = OptionDocumentationCategory.EXECUTION_STRATEGY,
        effectTags = {OptionEffectTag.EXECUTION},
        help =
            "Specifies which strategy to use for template expansion actions such as"
                + " ctx.actions.expand_template. 'local' writes the expanded template to disk."
                + " 'remote' keeps it off disk in the same way as --file_write_strategy=remote.")
    public abstract String getTemplateExpansionStrategy();
  }

  @Override
  public Iterable<Class<? extends OptionsBase>> getCommandOptions(String commandName) {
    return commandName.equals("build")
        ? ImmutableList.of(ExecutionOptions.class, RemoteOptions.class, Options.class)
        : ImmutableList.of();
  }

  @Override
  public void registerActionContexts(
      ModuleActionContextRegistry.Builder registryBuilder,
      CommandEnvironment env,
      BuildRequest buildRequest) {
    Options options = env.getOptions().getOptions(Options.class);
    registryBuilder
        .restrictTo(CppIncludeExtractionContext.class, "")
        .restrictTo(CppIncludeScanningContext.class, "")
        .restrictTo(FileWriteActionContext.class, options.getFileWriteStrategy())
        .restrictTo(TemplateExpansionContext.class, options.getTemplateExpansionStrategy())
        .restrictTo(SpawnCache.class, "");
  }

  @Override
  public void registerSpawnStrategies(
      SpawnStrategyRegistry.Builder registryBuilder, CommandEnvironment env) {
    ExecutionOptions options = env.getOptions().getOptions(ExecutionOptions.class);
    RemoteOptions remoteOptions = env.getOptions().getOptions(RemoteOptions.class);

    List<String> spawnStrategies = new ArrayList<>(options.getSpawnStrategy());

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
    registryBuilder.setDefaultStrategies(spawnStrategies);

    // By adding this filter before the ones derived from --strategy the latter can override the
    // former.
    registryBuilder.addMnemonicFilter("Genrule", options.getGenruleStrategy());

    for (Map.Entry<String, List<String>> strategy : options.getStrategy()) {
      registryBuilder.addMnemonicFilter(strategy.getKey(), strategy.getValue());
    }

    for (Map.Entry<RegexFilter, List<String>> entry : options.getStrategyByRegexp()) {
      registryBuilder.addDescriptionFilter(entry.getKey(), entry.getValue());
    }

    for (Map.Entry<Label, List<String>> strategy : options.getAllowedStrategiesByExecPlatform()) {
      registryBuilder.addExecPlatformFilter(strategy.getKey(), strategy.getValue());
    }
  }
}
