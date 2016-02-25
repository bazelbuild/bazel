// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.actions.ActionCompletionEvent;
import com.google.devtools.build.lib.actions.ActionStartedEvent;
import com.google.devtools.build.lib.analysis.AnalysisPhaseCompleteEvent;
import com.google.devtools.build.lib.buildtool.buildevent.BuildCompleteEvent;
import com.google.devtools.build.lib.buildtool.buildevent.BuildStartingEvent;
import com.google.devtools.build.lib.pkgcache.LoadingPhaseCompleteEvent;
import com.google.devtools.build.lib.util.io.AnsiTerminalWriter;

import java.io.IOException;
import java.util.ArrayDeque;
import java.util.Deque;

/**
 * An experimental state tracker for the new experimental UI.
 */
class ExperimentalStateTracker {

  static final int SAMPLE_SIZE = 3;

  private String status;
  private String additionalMessage;
  private int actionsStarted;
  private int actionsCompleted;
  private final Deque<String> runningActions;
  private boolean ok;

  ExperimentalStateTracker() {
    this.runningActions = new ArrayDeque<>();
    this.ok = true;
  }

  void buildStarted(BuildStartingEvent event) {
    status = "Loading";
    additionalMessage = "";
  }

  void loadingComplete(LoadingPhaseCompleteEvent event) {
    int count = event.getTargets().size();
    status = "Analysing";
    additionalMessage = "" + count + " targets";
  }

  void analysisComplete(AnalysisPhaseCompleteEvent event) {
    status = null;
  }

  void buildComplete(BuildCompleteEvent event) {
    if (event.getResult().getSuccess()) {
      status = "INFO";
      additionalMessage = "Build completed successfully, " + actionsCompleted + " total actions";
    } else {
      ok = false;
      status = "FAILED";
      additionalMessage = "Build did NOT complete successfully";
    }
  }

  synchronized void actionStarted(ActionStartedEvent event) {
    actionsStarted++;
    String name = event.getAction().getPrimaryOutput().getPath().getPathString();
    runningActions.addLast(name);
  }

  synchronized void actionCompletion(ActionCompletionEvent event) {
    actionsCompleted++;
    String name = event.getAction().getPrimaryOutput().getPath().getPathString();
    runningActions.remove(name);
  }

  private void sampleOldestActions(AnsiTerminalWriter terminalWriter) throws IOException {
    int count = 0;
    for (String action : runningActions) {
      count++;
      terminalWriter.newline().append("    " + action);
      if (count >= SAMPLE_SIZE) {
        break;
      }
    }
    if (count < runningActions.size()) {
      terminalWriter.newline().append("    ...");
    }
  }

  synchronized void writeProgressBar(AnsiTerminalWriter terminalWriter) throws IOException {
    if (status != null) {
      if (ok) {
        terminalWriter.okStatus();
      } else {
        terminalWriter.failStatus();
      }
      terminalWriter.append(status + ":").normal().append(" " + additionalMessage);
      return;
    }
    String statusMessage = " " + actionsCompleted + " actions completed, "
        + (actionsStarted - actionsCompleted) + " actions running";
    terminalWriter.okStatus().append("Building:").normal().append(" " + statusMessage);
    sampleOldestActions(terminalWriter);
  }
}
