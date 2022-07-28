// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis;

import com.google.common.eventbus.EventBus;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.skyframe.TopLevelStatusEvents.TopLevelEntityAnalysisConcludedEvent;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.Set;

/**
 * A watcher for analysis-related work that sends out a signal when all such work in the build is
 * done. There's one instance of this class per build.
 */
public class AnalysisOperationWatcher implements AutoCloseable {
  // Since the events are fired from within a SkyFunction, it's possible that the same event is
  // fired multiple times. A simple counter would therefore not suffice.
  private final Set<SkyKey> threadSafeExpectedKeys;
  private final EventBus eventBus;
  private final AnalysisOperationWatcherFinisher finisher;

  private AnalysisOperationWatcher(
      Set<SkyKey> threadSafeExpectedKeys,
      EventBus eventBus,
      AnalysisOperationWatcherFinisher finisher) {
    this.threadSafeExpectedKeys = threadSafeExpectedKeys;
    this.eventBus = eventBus;
    this.finisher = finisher;
  }

  /** Creates an AnalysisOperationWatcher and registers it with the provided eventBus. */
  public static AnalysisOperationWatcher createAndRegisterWithEventBus(
      Set<SkyKey> threadSafeExpectedKeys,
      EventBus eventBus,
      AnalysisOperationWatcherFinisher finisher) {
    AnalysisOperationWatcher watcher =
        new AnalysisOperationWatcher(threadSafeExpectedKeys, eventBus, finisher);
    eventBus.register(watcher);
    return watcher;
  }

  @Subscribe
  public void handleTopLevelEntityAnalysisConcluded(TopLevelEntityAnalysisConcludedEvent e) {
    if (threadSafeExpectedKeys.isEmpty()) {
      return;
    }
    threadSafeExpectedKeys.remove(e.getAnalyzedTopLevelKey());
    if (threadSafeExpectedKeys.isEmpty()) {
      try {
        finisher.analysisFinishedCallback();
      } catch (InterruptedException exception) {
        // Subscribers in general shouldn't throw exceptions. We therefore try to preserve the
        // interrupted status here.
        Thread.currentThread().interrupt();
      }
    }
  }

  @Override
  public void close() {
    eventBus.unregister(this);
  }

  /** A callback to be called when all the expected keys have been analyzed. */
  @FunctionalInterface
  public interface AnalysisOperationWatcherFinisher {
    void analysisFinishedCallback() throws InterruptedException;
  }
}
