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
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.exec.ExecutorBuilder;
import com.google.devtools.build.lib.rules.android.WriteAdbArgsActionContext;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
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
      category = "strategy",
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
      category = "strategy",
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
      category = "strategy",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "Specify how to distribute compilation of other spawn actions. "
              + "Example: 'Javac=local' means to spawn Java compilation locally. "
              + "'JavaIjar=sandboxed' means to spawn Java Ijar actions in a sandbox. "
    )
    public List<Map.Entry<String, String>> strategy;
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
    builder.addActionContextConsumer(new BazelActionContextConsumer(options));
  }
}
