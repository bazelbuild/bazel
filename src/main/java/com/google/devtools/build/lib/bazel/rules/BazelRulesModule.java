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

import com.google.common.base.Supplier;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.bazel.rules.cpp.BazelCppRuleClasses;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.exec.ExecutorBuilder;
import com.google.devtools.build.lib.query2.output.OutputFormatter;
import com.google.devtools.build.lib.rules.android.WriteAdbArgsActionContext;
import com.google.devtools.build.lib.rules.cpp.FdoSupportFunction;
import com.google.devtools.build.lib.rules.cpp.FdoSupportValue;
import com.google.devtools.build.lib.rules.genquery.GenQuery;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.WorkspaceBuilder;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.util.ResourceFileLoader;
import com.google.devtools.common.options.Converters.AssignmentConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionsBase;
import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * Module implementing the rule set of Bazel.
 */
public class BazelRulesModule extends BlazeModule {
  /**
   * Execution options affecting how we execute the build actions (but not their semantics).
   */
  public static class BazelExecutionOptions extends OptionsBase {
    @Option(
      name = "spawn_strategy",
      defaultValue = "",
      category = "strategy",
      help =
          "Specify how spawn actions are executed by default."
              + "'standalone' means run all of them locally."
              + "'sandboxed' means run them in namespaces based sandbox (available only on Linux)"
    )
    public String spawnStrategy;

    @Option(
      name = "genrule_strategy",
      defaultValue = "",
      category = "strategy",
      help =
          "Specify how to execute genrules."
              + "'standalone' means run all of them locally."
              + "'sandboxed' means run them in namespaces based sandbox (available only on Linux)"
    )
    public String genruleStrategy;

    @Option(name = "strategy",
        allowMultiple = true,
        converter = AssignmentConverter.class,
        defaultValue = "",
        category = "strategy",
        help = "Specify how to distribute compilation of other spawn actions. "
            + "Example: 'Javac=local' means to spawn Java compilation locally. "
            + "'JavaIjar=sandboxed' means to spawn Java Ijar actions in a sandbox. ")
    public List<Map.Entry<String, String>> strategy;
  }

  private CommandEnvironment env;

  @Override
  public void beforeCommand(Command command, CommandEnvironment env) {
    this.env = env;
    env.getEventBus().register(this);
  }

  @Override
  public void afterCommand() {
    this.env = null;
  }

  @Override
  public Iterable<Class<? extends OptionsBase>> getCommandOptions(Command command) {
    return "build".equals(command.name())
        ? ImmutableList.<Class<? extends OptionsBase>>of(BazelExecutionOptions.class)
        : ImmutableList.<Class<? extends OptionsBase>>of();
  }

  @Override
  public void initializeRuleClasses(ConfiguredRuleClassProvider.Builder builder) {
    builder.setToolsRepository(BazelRuleClassProvider.TOOLS_REPOSITORY);
    BazelRuleClassProvider.setup(builder);
    try {
      // Load auto-configuration files, it is made outside of the rule class provider so that it
      // will not be loaded for our Java tests.
      builder.addWorkspaceFileSuffix(
          ResourceFileLoader.loadResource(BazelCppRuleClasses.class, "cc_configure.WORKSPACE"));
      builder.addWorkspaceFileSuffix(
          ResourceFileLoader.loadResource(BazelRulesModule.class, "xcode_configure.WORKSPACE"));
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }
  }

  @Override
  public void workspaceInit(BlazeDirectories directories, WorkspaceBuilder builder) {
    builder.addSkyFunction(FdoSupportValue.SKYFUNCTION, new FdoSupportFunction());
    builder.addPrecomputedValue(PrecomputedValue.injected(
        GenQuery.QUERY_OUTPUT_FORMATTERS,
        new Supplier<ImmutableList<OutputFormatter>>() {
          @Override
          public ImmutableList<OutputFormatter> get() {
            return env.getRuntime().getQueryOutputFormatters();
          }
        }));
  }

  @Override
  public void executorInit(CommandEnvironment env, BuildRequest request, ExecutorBuilder builder) {
    builder.addActionContext(new WriteAdbArgsActionContext(env.getClientEnv().get("HOME")));
    BazelExecutionOptions options = env.getOptions().getOptions(BazelExecutionOptions.class);
    builder.addActionContextConsumer(new BazelActionContextConsumer(options));
  }
}
