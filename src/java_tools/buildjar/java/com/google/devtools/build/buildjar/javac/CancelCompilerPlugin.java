// Copyright 2021 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.buildjar.javac;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.buildjar.InvalidCommandLineException;
import com.google.devtools.build.buildjar.javac.plugins.BlazeJavaCompilerPlugin;
import com.sun.source.util.TaskEvent;
import com.sun.source.util.TaskListener;
import com.sun.tools.javac.api.MultiTaskListener;
import com.sun.tools.javac.comp.AttrContext;
import com.sun.tools.javac.comp.Env;
import com.sun.tools.javac.util.Context;

/**
 * A helper plugin to stop the java compilation at different stages when the worker cancellation is
 * enabled.
 */
public class CancelCompilerPlugin extends BlazeJavaCompilerPlugin implements TaskListener {

  private final int requestId;
  private final WorkerCancellationRegistry cancellationRegistry;

  /**
   * @param requestId the id of the javac request that needs to be cancelled.
   * @param cancellationRegistry this registry handles which requests to be cancelled.
   */
  public CancelCompilerPlugin(int requestId, WorkerCancellationRegistry cancellationRegistry) {
    this.requestId = requestId;
    this.cancellationRegistry = cancellationRegistry;
  }

  @Override
  public void initializeContext(Context context) {
    super.initializeContext(context);
    MultiTaskListener.instance(context).add(this);
  }

  @Override
  public void processArgs(
      ImmutableList<String> standardJavacopts, ImmutableList<String> blazeJavacopts)
      throws InvalidCommandLineException {
    cancelRequest();
  }

  @Override
  public void postAttribute(Env<AttrContext> env) {
    cancelRequest();
  }

  @Override
  public void postFlow(Env<AttrContext> env) {
    cancelRequest();
  }

  @Override
  public void started(TaskEvent e) {
    cancelRequest();
  }

  @Override
  public void finished(TaskEvent e) {
    cancelRequest();
  }

  /** A subclass of RuntimeException specific to when compilation fails because of cancellation. */
  public static class CancelRequestException extends RuntimeException {}

  private void cancelRequest() {
    if (cancellationRegistry.checkIfRequestIsCancelled(requestId)) {
      throw new CancelRequestException();
    }
  }
}
