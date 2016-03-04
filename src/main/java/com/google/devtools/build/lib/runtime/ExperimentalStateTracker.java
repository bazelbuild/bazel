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

import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionCompletionEvent;
import com.google.devtools.build.lib.actions.ActionStartedEvent;
import com.google.devtools.build.lib.analysis.AnalysisPhaseCompleteEvent;
import com.google.devtools.build.lib.buildtool.ExecutionProgressReceiver;
import com.google.devtools.build.lib.buildtool.buildevent.BuildCompleteEvent;
import com.google.devtools.build.lib.buildtool.buildevent.BuildStartingEvent;
import com.google.devtools.build.lib.buildtool.buildevent.ExecutionProgressReceiverAvailableEvent;
import com.google.devtools.build.lib.pkgcache.LoadingPhaseCompleteEvent;
import com.google.devtools.build.lib.util.io.AnsiTerminalWriter;

import java.io.IOException;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Map;
import java.util.TreeMap;

/**
 * An experimental state tracker for the new experimental UI.
 */
class ExperimentalStateTracker {

  static final int SAMPLE_SIZE = 3;

  private String status;
  private String additionalMessage;

  // currently running actions, using the path of the primary
  // output as unique identifier.
  private final Deque<String> runningActions;
  private final Map<String, Action> actions;

  private int actionsCompleted;
  private boolean ok;

  private ExecutionProgressReceiver executionProgressReceiver;

  ExperimentalStateTracker() {
    this.runningActions = new ArrayDeque<>();
    this.actions = new TreeMap<>();
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

  void progressReceiverAvailable(ExecutionProgressReceiverAvailableEvent event) {
    executionProgressReceiver = event.getExecutionProgressReceiver();
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
    Action action = event.getAction();
    String name = action.getPrimaryOutput().getPath().getPathString();
    runningActions.addLast(name);
    actions.put(name, action);
  }

  synchronized void actionCompletion(ActionCompletionEvent event) {
    actionsCompleted++;
    Action action = event.getAction();
    String name = action.getPrimaryOutput().getPath().getPathString();
    runningActions.remove(name);
    actions.remove(name);

    // As callers to the experimental state tracker assume we will fully report the new state once
    // informed of an action completion, we need to make sure the progress receiver is aware of the
    // completion, even though it might be called later on the event bus.
    executionProgressReceiver.actionCompleted(action);
  }

  private String describeAction(String name) {
    Action action = actions.get(name);
    String message = action.getProgressMessage();
    if (message != null) {
      return message;
    }
    return action.prettyPrint();
  }

  private void sampleOldestActions(AnsiTerminalWriter terminalWriter) throws IOException {
    int count = 0;
    for (String action : runningActions) {
      count++;
      terminalWriter.newline().append("    " + describeAction(action));
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
    if (executionProgressReceiver != null) {
      terminalWriter.okStatus().append(executionProgressReceiver.getProgressString());
    } else {
      terminalWriter.okStatus().append("Building:");
    }
    if (runningActions.size() == 1) {
      String statusMessage = describeAction(runningActions.peekFirst());
      terminalWriter.normal().append(" " + statusMessage);
    } else {
      String statusMessage = " " + runningActions.size() + " actions running";
      terminalWriter.normal().append(" " + statusMessage);
      sampleOldestActions(terminalWriter);
    }
  }
}
