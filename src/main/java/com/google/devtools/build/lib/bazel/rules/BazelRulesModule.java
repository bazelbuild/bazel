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

package com.google.devtools.build.lib.bazel.rules;

import com.google.common.base.Supplier;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.actions.ActionContextConsumer;
import com.google.devtools.build.lib.actions.ActionContextProvider;
import com.google.devtools.build.lib.actions.ActionGraph;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Executor.ActionContext;
import com.google.devtools.build.lib.actions.ExecutorInitException;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.query2.output.OutputFormatter;
import com.google.devtools.build.lib.rules.cpp.CppCompileActionContext;
import com.google.devtools.build.lib.rules.cpp.CppLinkActionContext;
import com.google.devtools.build.lib.rules.cpp.LocalGccStrategy;
import com.google.devtools.build.lib.rules.cpp.LocalLinkStrategy;
import com.google.devtools.build.lib.rules.genquery.GenQuery;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.GotOptionsEvent;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsProvider;

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
        defaultValue = "standalone",
        category = "strategy",
        help = "Specify how spawn actions are executed by default."
            + "'standalone' means run all of them locally."
            + "'sandboxed' means run them in namespaces based sandbox (available only on Linux)")
    public String spawnStrategy;

    @Option(
        name = "genrule_strategy",
        defaultValue = "standalone", 
        category = "strategy",
        help = "Specify how to execute genrules."
            + "'standalone' means run all of them locally."
            + "'sandboxed' means run them in namespaces based sandbox (available only on Linux)")

    public String genruleStrategy;
  }

  private static class BazelActionContextConsumer implements ActionContextConsumer {
    BazelExecutionOptions options;

    private BazelActionContextConsumer(BazelExecutionOptions options) {
      this.options = options;

    }
    @Override
    public Map<String, String> getSpawnActionContexts() {
      ImmutableMap.Builder<String, String> builder = ImmutableMap.builder();

      builder.put("Genrule", options.genruleStrategy);

      // TODO(bazel-team): put this in getActionContexts (key=SpawnActionContext.class) instead
      builder.put("", options.spawnStrategy);

      return builder.build();
    }

    @Override
    public Map<Class<? extends ActionContext>, String> getActionContexts() {
      ImmutableMap.Builder<Class<? extends ActionContext>, String> builder =
          ImmutableMap.builder();
      builder.put(CppCompileActionContext.class, "");
      builder.put(CppLinkActionContext.class, "");
      return builder.build();
    }
  }

  private class BazelActionContextProvider implements ActionContextProvider {
    @Override
    public Iterable<ActionContext> getActionContexts() {
      return ImmutableList.of(
          new LocalGccStrategy(optionsProvider),
          new LocalLinkStrategy());
    }

    @Override
    public void executorCreated(Iterable<ActionContext> usedContexts)
        throws ExecutorInitException {
    }

    @Override
    public void executionPhaseStarting(ActionInputFileCache actionInputFileCache,
        ActionGraph actionGraph, Iterable<Artifact> topLevelArtifacts)
        throws ExecutorInitException, InterruptedException {
    }

    @Override
    public void executionPhaseEnding() {
    }
  }

  private BlazeRuntime runtime;
  private OptionsProvider optionsProvider;

  @Override
  public void beforeCommand(BlazeRuntime blazeRuntime, Command command) {
    this.runtime = blazeRuntime;
    runtime.getEventBus().register(this);
  }

  @Override
  public Iterable<Class<? extends OptionsBase>> getCommandOptions(Command command) {
    return command.builds()
        ? ImmutableList.<Class<? extends OptionsBase>>of(BazelExecutionOptions.class)
        : ImmutableList.<Class<? extends OptionsBase>>of();
  }

  @Override
  public ActionContextConsumer getActionContextConsumer() {
    return new BazelActionContextConsumer(
        optionsProvider.getOptions(BazelExecutionOptions.class));
  }

  @Override
  public ActionContextProvider getActionContextProvider() {
    return new BazelActionContextProvider();
  }

  @Subscribe
  public void gotOptions(GotOptionsEvent event) {
    optionsProvider = event.getOptions();
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
            return runtime.getQueryOutputFormatters();
          }
        }));
  }
}
