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
package com.google.devtools.build.lib.metrics;

import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.analysis.AnalysisPhaseCompleteEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics.ActionSummary;
import com.google.devtools.build.lib.buildtool.buildevent.BuildCompleteEvent;
import com.google.devtools.build.lib.runtime.CommandEnvironment;

class MetricsCollector {

  private final CommandEnvironment env;
  private int actionsConstructed;

  MetricsCollector(CommandEnvironment env) {
    this.env = env;
    env.getEventBus().register(this);
  }

  static void installInEnv(CommandEnvironment env) {
    new MetricsCollector(env);
  }

  @Subscribe
  public void onAnalysisPhaseComplete(AnalysisPhaseCompleteEvent event) {
    actionsConstructed = event.getActionsConstructed();
  }

  @Subscribe
  public void onBuildComplete(BuildCompleteEvent event) {
    env.getEventBus().post(new BuildMetricsEvent(createBuildMetrics()));
  }

  private BuildMetrics createBuildMetrics() {
    return BuildMetrics.newBuilder()
        .setActionSummary(ActionSummary.newBuilder().setActionsCreated(actionsConstructed).build())
        .build();
  }
}
