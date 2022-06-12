// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote.common;

import build.bazel.remote.execution.v2.ExecuteResponse;
import build.bazel.remote.execution.v2.RequestMetadata;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.Spawn;
import javax.annotation.Nullable;

/** A context that provide remote execution related information for executing an action remotely. */
public class RemoteActionExecutionContext {
  /** The current step of the context. */
  public enum Step {
    INIT,
    CHECK_ACTION_CACHE,
    UPLOAD_INPUTS,
    EXECUTE_REMOTELY,
    UPLOAD_OUTPUTS,
    DOWNLOAD_OUTPUTS,
    UPLOAD_BES_FILES,
  }

  @Nullable private final Spawn spawn;
  private final RequestMetadata requestMetadata;
  private final NetworkTime networkTime;

  @Nullable private ExecuteResponse executeResponse;
  private Step step;

  private RemoteActionExecutionContext(
      @Nullable Spawn spawn, RequestMetadata requestMetadata, NetworkTime networkTime) {
    this.spawn = spawn;
    this.requestMetadata = requestMetadata;
    this.networkTime = networkTime;
    this.step = Step.INIT;
  }

  /** Returns current {@link Step} of the context. */
  public Step getStep() {
    return step;
  }

  /** Sets current {@link Step} of the context. */
  public void setStep(Step step) {
    this.step = step;
  }

  /** Returns the {@link Spawn} of the action being executed or {@code null}. */
  @Nullable
  public Spawn getSpawn() {
    return spawn;
  }

  /** Returns the {@link RequestMetadata} for the action being executed. */
  public RequestMetadata getRequestMetadata() {
    return requestMetadata;
  }

  /**
   * Returns the {@link NetworkTime} instance used to measure the network time during the action
   * execution.
   */
  public NetworkTime getNetworkTime() {
    return networkTime;
  }

  @Nullable
  public ActionExecutionMetadata getSpawnOwner() {
    Spawn spawn = getSpawn();
    if (spawn == null) {
      return null;
    }

    return spawn.getResourceOwner();
  }

  public void setExecuteResponse(@Nullable ExecuteResponse executeResponse) {
    this.executeResponse = executeResponse;
  }

  @Nullable
  public ExecuteResponse getExecuteResponse() {
    return executeResponse;
  }

  /** Creates a {@link RemoteActionExecutionContext} with given {@link RequestMetadata}. */
  public static RemoteActionExecutionContext create(RequestMetadata metadata) {
    return new RemoteActionExecutionContext(/*spawn=*/ null, metadata, new NetworkTime());
  }

  /**
   * Creates a {@link RemoteActionExecutionContext} with given {@link Spawn} and {@link
   * RequestMetadata}.
   */
  public static RemoteActionExecutionContext create(
      @Nullable Spawn spawn, RequestMetadata metadata) {
    return new RemoteActionExecutionContext(spawn, metadata, new NetworkTime());
  }
}
