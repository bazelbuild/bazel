// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.bazel.repository.skylark;

import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.repository.RequestRepositoryInformationEvent;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * Module reporting back the place an external repository was defined, if requested by some error
 * involving that repository. This also covers cases where the definition of the repository is not
 * directly available, e.g., during detection of a dependency cycle.
 */
public final class SkylarkRepositoryDebugModule extends BlazeModule {
  Map<String, String> repositoryDefinitions;
  Reporter reporter;
  Set<String> reported;

  @Override
  public void beforeCommand(CommandEnvironment env) {
    repositoryDefinitions = new HashMap<>();
    reported = new HashSet<>();
    reporter = env.getReporter();
    env.getEventBus().register(this);
  }

  @Override
  public void afterCommand() {
    repositoryDefinitions = null;
    reporter = null;
    reported = null;
  }

  @Subscribe
  public synchronized void definitionLocation(SkylarkRepositoryDefinitionLocationEvent event) {
    repositoryDefinitions.put(event.getName(), event.getDefinitionInformation());
  }

  @Subscribe
  public void requestDefinition(RequestRepositoryInformationEvent event) {
    String toReport = null;
    synchronized (this) {
      if (!reported.contains(event.getName())
          && repositoryDefinitions.containsKey(event.getName())) {
        toReport = repositoryDefinitions.get(event.getName());
      }
    }
    if (toReport != null) {
      reporter.handle(Event.info(toReport));
    }
  }
}
