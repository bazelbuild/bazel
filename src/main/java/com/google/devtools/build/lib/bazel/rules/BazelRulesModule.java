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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.actions.ActionContextConsumer;
import com.google.devtools.build.lib.actions.Executor.ActionContext;
import com.google.devtools.build.lib.blaze.BlazeModule;
import com.google.devtools.build.lib.blaze.BlazeRuntime;
import com.google.devtools.build.lib.blaze.Command;
import com.google.devtools.build.lib.blaze.GotOptionsEvent;
import com.google.devtools.build.lib.view.ConfiguredRuleClassProvider;
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
  public static class Google3ExecutionOptions extends OptionsBase {
    @Option(name = "spawn_strategy", defaultValue = "", category = "strategy", help =
        "Specify where spawn actions are executed by default. This is "
        + "overridden by the more specific strategy options.")
    public String spawnStrategy;

    @Option(name = "genrule_strategy",
        defaultValue = "",
        category = "strategy",
        help = "Specify how to execute genrules. "
            + "'standalone' means run all of them locally.")
    public String genruleStrategy;
  }

  private static class Google3ActionContextConsumer implements ActionContextConsumer {
    Google3ExecutionOptions options;

    private Google3ActionContextConsumer(Google3ExecutionOptions options) {
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
      return builder.build();
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
        ? ImmutableList.<Class<? extends OptionsBase>>of(Google3ExecutionOptions.class)
        : ImmutableList.<Class<? extends OptionsBase>>of();
  }

  @Override
  public ActionContextConsumer getActionContextConsumer() {
    return new Google3ActionContextConsumer(
        optionsProvider.getOptions(Google3ExecutionOptions.class));
  }

  @Subscribe
  public void gotOptions(GotOptionsEvent event) {
    optionsProvider = event.getOptions();
  }

  @Override
  public void initializeRuleClasses(ConfiguredRuleClassProvider.Builder builder) {
    BazelRuleClassProvider.setup(builder);
  }
}
