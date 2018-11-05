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
import com.google.devtools.build.lib.exec.ExecutorBuilder;
import com.google.devtools.build.lib.exec.SpawnCache;
import com.google.devtools.build.lib.rules.android.WriteAdbArgsActionContext;
import com.google.devtools.build.lib.rules.cpp.CppCompileActionContext;
import com.google.devtools.build.lib.rules.cpp.CppIncludeExtractionContext;
import com.google.devtools.build.lib.rules.cpp.CppIncludeScanningContext;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.util.RegexFilter;
import com.google.devtools.common.options.Converters.AssignmentConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import java.util.List;
import java.util.Map;

/** Module which registers the strategy options for Bazel. */
public class BazelStrategyModule extends BlazeModule {
  /**
   * Execution options affecting how we execute the build actions (but not their semantics).
   */
  public static class BazelExecutionOptions extends OptionsBase {
    @Option(
      name = "spawn_strategy",
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "Specify how spawn actions are executed by default. "
              + "'standalone' means run all of them locally without any kind of sandboxing. "
              + "'sandboxed' means to run them in a sandboxed environment with limited privileges "
              + "(details depend on platform support)."
    )
    public String spawnStrategy;

    @Option(
      name = "genrule_strategy",
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "Specify how to execute genrules. "
              + "'standalone' means run all of them locally. "
              + "'sandboxed' means run them in namespaces based sandbox (available only on Linux)"
    )
    public String genruleStrategy;

    @Option(
      name = "strategy",
      allowMultiple = true,
      converter = AssignmentConverter.class,
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "Specify how to distribute compilation of other spawn actions. "
              + "Example: 'Javac=local' means to spawn Java compilation locally. "
              + "'JavaIjar=sandboxed' means to spawn Java Ijar actions in a sandbox. "
    )
    public List<Map.Entry<String, String>> strategy;

    @Option(
        name = "strategy_regexp",
        allowMultiple = true,
        converter = RegexFilterAssignmentConverter.class,
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        defaultValue = "",
        help =
            "Override which spawn strategy should be used to execute spawn actions that have "
                + "descriptions matching a certain regex_filter. See --per_file_copt for details on"
                + "regex_filter matching. "
                + "The first regex_filter that matches the description is used. "
                + "This option overrides other flags for specifying strategy. "
                + "Example: --strategy_regexp=//foo.*\\.cc,-//foo/bar=local means to run actions "
                + "using local strategy if their descriptions match //foo.*.cc but not //foo/bar. "
                + "Example: --strategy_regexp='Compiling.*/bar=local "
                + " --strategy_regexp=Compiling=sandboxed will run 'Compiling //foo/bar/baz' with "
                + "the 'local' strategy, but reversing the order would run it with 'sandboxed'. "
    )
    public List<Map.Entry<RegexFilter, String>> strategyByRegexp;
  }

  @Override
  public Iterable<Class<? extends OptionsBase>> getCommandOptions(Command command) {
    return "build".equals(command.name())
        ? ImmutableList.<Class<? extends OptionsBase>>of(BazelExecutionOptions.class)
        : ImmutableList.<Class<? extends OptionsBase>>of();
  }

  @Override
  public void executorInit(CommandEnvironment env, BuildRequest request, ExecutorBuilder builder) {
    builder.addActionContext(new WriteAdbArgsActionContext(env.getClientEnv().get("HOME")));
    BazelExecutionOptions options = env.getOptions().getOptions(BazelExecutionOptions.class);

    // Default strategies for certain mnemonics - they can be overridden by --strategy= flags.
    builder.addStrategyByMnemonic("Javac", "worker");
    builder.addStrategyByMnemonic("Closure", "worker");
    builder.addStrategyByMnemonic("DexBuilder", "worker");

    // Allow genrule_strategy to also be overridden by --strategy= flags.
    builder.addStrategyByMnemonic("Genrule", options.genruleStrategy);

    for (Map.Entry<String, String> strategy : options.strategy) {
      String strategyName = strategy.getValue();
      // TODO(philwo) - remove this when the standalone / local mess is cleaned up.
      // Some flag expansions use "local" as the strategy name, but the strategy is now called
      // "standalone", so we'll translate it here.
      if (strategyName.equals("local")) {
        strategyName = "standalone";
      }
      builder.addStrategyByMnemonic(strategy.getKey(), strategyName);
    }

    builder.addStrategyByMnemonic("", options.spawnStrategy);

    for (Map.Entry<RegexFilter, String> entry : options.strategyByRegexp) {
      builder.addStrategyByRegexp(entry.getKey(), entry.getValue());
    }

    builder
        .addStrategyByContext(CppCompileActionContext.class, "")
        .addStrategyByContext(CppIncludeExtractionContext.class, "")
        .addStrategyByContext(CppIncludeScanningContext.class, "")
        .addStrategyByContext(FileWriteActionContext.class, "")
        .addStrategyByContext(TemplateExpansionContext.class, "")
        .addStrategyByContext(WriteAdbArgsActionContext.class, "")
        .addStrategyByContext(SpawnCache.class, "");
  }
}
