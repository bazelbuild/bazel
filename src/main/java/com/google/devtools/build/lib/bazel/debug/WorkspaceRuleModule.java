// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.bazel.debug;

import com.google.common.collect.ImmutableList;
import com.google.common.eventbus.EventBus;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.common.options.OptionsBase;

/** A module for logging workspace rule events */
public final class WorkspaceRuleModule extends BlazeModule {
  private Reporter reporter;
  private EventBus eventBus;

  @Override
  public void beforeCommand(CommandEnvironment env) {

    reporter = env.getReporter();
    eventBus = env.getEventBus();

    if (env.getOptions() == null || env.getOptions().getOptions(DebuggingOptions.class) == null) {
      reporter.handle(Event.error("Installation is corrupt: could not retrieve debugging options"));
      return;
    }

    if (env.getOptions().getOptions(DebuggingOptions.class).workspaceRulesLogging != null) {
      eventBus.register(this);
    }
  }

  @Override
  public Iterable<Class<? extends OptionsBase>> getCommonCommandOptions() {
    return ImmutableList.<Class<? extends OptionsBase>>of(DebuggingOptions.class);
  }

  @Subscribe
  public void workspaceRuleEventReceived(WorkspaceRuleEvent event) {
    reporter.handle(Event.info(event.logMessage()));
  }
}
