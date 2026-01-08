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
package com.google.devtools.build.lib.skyframe;

import java.util.concurrent.atomic.AtomicInteger;

/**
 * A class that, when being told the end of a target or aspect being configured, keeps track of the
 * configuration progress and provides it as a human-readable string intended for the progress bar.
 *
 * <p>Non-final to be mockable.
 */
public class AnalysisProgressReceiver {

  private final AtomicInteger configuredTargetsCompleted = new AtomicInteger();
  private final AtomicInteger configuredTargetsDownloaded = new AtomicInteger();
  private final AtomicInteger configuredAspectsCompleted = new AtomicInteger();
  private final AtomicInteger configuredAspectsDownloaded = new AtomicInteger();

  /** Register that a target has been configured. */
  void doneConfigureTarget() {
    configuredTargetsCompleted.incrementAndGet();
  }

  /** Register that a configured target has been downloaded from a remote cache. */
  void doneDownloadedConfiguredTarget() {
    configuredTargetsCompleted.incrementAndGet();
    configuredTargetsDownloaded.incrementAndGet();
  }

  /** Register that a aspect has been configured. */
  void doneConfigureAspect() {
    configuredAspectsCompleted.incrementAndGet();
  }

  /** Register that a configured target has been downloaded from a remote cache. */
  void doneDownloadedConfiguredAspect() {
    configuredAspectsCompleted.incrementAndGet();
    configuredAspectsDownloaded.incrementAndGet();
  }

  /**
   * Reset all instance variables of this object to a state equal to that of a newly
   * constructed object.
   */
  public void reset() {
    configuredTargetsCompleted.set(0);
    configuredTargetsDownloaded.set(0);
    configuredAspectsCompleted.set(0);
    configuredAspectsDownloaded.set(0);
  }

  /**
   * Return a snapshot of the configuration progress as human-readable description of the number of
   * targets and aspects configured so far.
   */
  public String getProgressString() {
    StringBuilder sb = new StringBuilder();

    long targets = configuredTargetsCompleted.get();
    sb.append(targets).append(targets == 1 ? " target" : " targets").append(" configured");

    long downloadedTargets = configuredTargetsDownloaded.get();
    if (downloadedTargets > 0) {
      sb.append(" (").append(downloadedTargets).append(" remote cache hits)");
    }

    long aspects = configuredAspectsCompleted.get();
    if (aspects > 0) {
      sb.append(", ")
          .append(aspects)
          .append(aspects == 1 ? " aspect application" : " aspect applications");
      long downloadedAspects = configuredAspectsDownloaded.get();
      if (downloadedAspects > 0) {
        sb.append(" (").append(downloadedAspects).append(" remote cache hits)");
      }
    }

    return sb.toString();
  }
}
