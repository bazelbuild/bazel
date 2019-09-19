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
package com.google.devtools.build.lib.runtime;

import com.google.common.eventbus.EventBus;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics;
import com.google.devtools.build.lib.buildtool.buildevent.BuildCompleteEvent;
import com.google.devtools.build.lib.metrics.BuildMetricsEvent;

/**
 * Non-functional variant of {@code MetricsModule} to be supplied by the BlazeRuntime when
 * MetricsModule is not installed (tests).
 */
public class DummyMetricsModule extends BlazeModule {

  @Override
  public void beforeCommand(CommandEnvironment env) {
    EventBus eventBus = env.getEventBus();
    eventBus.register(new DummyMetricsCollector(eventBus));
  }

  @Override
  public boolean postsBuildMetricsEvent() {
    return true;
  }

  /**
   * Non-functional variant of {@code MetricsCollector} that posts an empty BuildMetricsEvent after
   * the build completes.
   */
  static final class DummyMetricsCollector {

    private final EventBus eventBus;

    DummyMetricsCollector(EventBus eventBus) {
      this.eventBus = eventBus;
    }

    @Subscribe
    public void onBuildComplete(BuildCompleteEvent event) {
      eventBus.post(BuildMetricsEvent.create(BuildMetrics.getDefaultInstance()));
    }
  }
}
