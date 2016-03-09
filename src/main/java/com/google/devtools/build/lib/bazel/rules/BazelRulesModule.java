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
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.Multimap;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.actions.ActionContextConsumer;
import com.google.devtools.build.lib.actions.ActionContextProvider;
import com.google.devtools.build.lib.actions.Executor.ActionContext;
import com.google.devtools.build.lib.actions.SimpleActionContextProvider;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.actions.FileWriteActionContext;
import com.google.devtools.build.lib.query2.output.OutputFormatter;
import com.google.devtools.build.lib.rules.android.WriteAdbArgsActionContext;
import com.google.devtools.build.lib.rules.cpp.CppCompileActionContext;
import com.google.devtools.build.lib.rules.cpp.CppLinkActionContext;
import com.google.devtools.build.lib.rules.cpp.IncludeScanningContext;
import com.google.devtools.build.lib.rules.genquery.GenQuery;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.GotOptionsEvent;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.common.options.Converters.AssignmentConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionsBase;

import java.util.List;
import java.util.Map;
import java.util.TreeMap;

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

  private static class BazelActionContextConsumer implements ActionContextConsumer {
    BazelExecutionOptions options;

    private BazelActionContextConsumer(BazelExecutionOptions options) {
      this.options = options;

    }
    @Override
    public Map<String, String> getSpawnActionContexts() {
      Map<String, String> contexts = new TreeMap<>(String.CASE_INSENSITIVE_ORDER);

      contexts.put("Genrule", options.genruleStrategy);

      for (Map.Entry<String, String> strategy : options.strategy) {
        String strategyName = strategy.getValue();
        // TODO(philwo) - remove this when the standalone / local mess is cleaned up.
        // Some flag expansions use "local" as the strategy name, but the strategy is now called
        // "standalone", so we'll translate it here.
        if (strategyName.equals("local")) {
          strategyName = "standalone";
        }
        contexts.put(strategy.getKey(), strategyName);
      }

      // TODO(bazel-team): put this in getActionContexts (key=SpawnActionContext.class) instead
      contexts.put("", options.spawnStrategy);

      return ImmutableMap.copyOf(contexts);
    }

    @Override
    public Multimap<Class<? extends ActionContext>, String> getActionContexts() {
      return ImmutableMultimap.<Class<? extends ActionContext>, String>builder()
          .put(CppCompileActionContext.class, "")
          .put(CppLinkActionContext.class, "")
          .put(IncludeScanningContext.class, "")
          .put(FileWriteActionContext.class, "")
          .put(WriteAdbArgsActionContext.class, "")
          .build();
    }
  }

  private CommandEnvironment env;
  private BazelExecutionOptions options;

  @Override
  public void beforeCommand(Command command, CommandEnvironment env) {
    this.env = env;
    env.getEventBus().register(this);
  }

  @Override
  public void afterCommand() {
    this.env = null;
    this.options = null;
  }

  @Override
  public Iterable<Class<? extends OptionsBase>> getCommandOptions(Command command) {
    return command.builds()
        ? ImmutableList.<Class<? extends OptionsBase>>of(BazelExecutionOptions.class)
        : ImmutableList.<Class<? extends OptionsBase>>of();
  }

  @Override
  public Iterable<ActionContextProvider> getActionContextProviders() {
    return ImmutableList.<ActionContextProvider>of(new SimpleActionContextProvider(
        new WriteAdbArgsActionContext(env.getClientEnv().get("HOME"))));
  }

  @Override
  public Iterable<ActionContextConsumer> getActionContextConsumers() {
    return ImmutableList.<ActionContextConsumer>of(
        new BazelActionContextConsumer(options));
  }

  @Subscribe
  public void gotOptions(GotOptionsEvent event) {
    options = event.getOptions().getOptions(BazelExecutionOptions.class);
  }

  @Override
  public void initializeRuleClasses(ConfiguredRuleClassProvider.Builder builder) {
    BazelRuleClassProvider.setup(builder);
  }

  @Override
  public Iterable<PrecomputedValue.Injected> getPrecomputedSkyframeValues() {
    return ImmutableList.of(PrecomputedValue.injected(
        GenQuery.QUERY_OUTPUT_FORMATTERS,
        new Supplier<ImmutableList<OutputFormatter>>() {
          @Override
          public ImmutableList<OutputFormatter> get() {
            return env.getRuntime().getQueryOutputFormatters();
          }
        }));
  }
}
