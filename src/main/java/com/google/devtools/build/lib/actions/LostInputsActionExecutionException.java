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

package com.google.devtools.build.lib.actions;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.Path;
import javax.annotation.Nullable;

/**
 * An {@link ActionExecutionException} thrown when an action fails to execute because one or more of
 * its inputs was lost. In some cases, Bazel may know how to fix this on its own.
 */
public class LostInputsActionExecutionException extends ActionExecutionException {

  /** Maps lost input digests to their ActionInputs. */
  private final ImmutableMap<String, ActionInput> lostInputs;

  private final ActionInputDepOwners owners;

  /**
   * If an ActionStartedEvent was emitted, then:
   *
   * <ul>
   *   <li>if rewinding is attempted, then an ActionRewindEvent should be emitted.
   *   <li>if rewinding fails, then an ActionCompletionEvent should be emitted.
   * </ul>
   */
  private boolean actionStartedEventAlreadyEmitted;

  /** Used to report the action execution failure if rewinding also fails. */
  @Nullable private Path primaryOutputPath;

  /**
   * Used to report the action execution failure if rewinding also fails. Note that this will be
   * closed, so it may only be used for reporting.
   */
  @Nullable private FileOutErr fileOutErr;

  /** Used to inform rewinding that lost inputs were found during input discovery. */
  private boolean fromInputDiscovery;

  public LostInputsActionExecutionException(
      String message,
      ImmutableMap<String, ActionInput> lostInputs,
      ActionInputDepOwners owners,
      Action action,
      Exception cause) {
    super(message, cause, action, /*catastrophe=*/ false);
    this.lostInputs = lostInputs;
    this.owners = owners;
  }

  public ImmutableMap<String, ActionInput> getLostInputs() {
    return lostInputs;
  }

  public ActionInputDepOwners getOwners() {
    return owners;
  }

  public Path getPrimaryOutputPath() {
    return primaryOutputPath;
  }

  public void setPrimaryOutputPath(Path primaryOutputPath) {
    this.primaryOutputPath = primaryOutputPath;
  }

  public FileOutErr getFileOutErr() {
    return fileOutErr;
  }

  public void setFileOutErr(FileOutErr fileOutErr) {
    this.fileOutErr = fileOutErr;
  }

  public boolean isActionStartedEventAlreadyEmitted() {
    return actionStartedEventAlreadyEmitted;
  }

  public void setActionStartedEventAlreadyEmitted() {
    this.actionStartedEventAlreadyEmitted = true;
  }

  public boolean isFromInputDiscovery() {
    return fromInputDiscovery;
  }

  public void setFromInputDiscovery() {
    this.fromInputDiscovery = true;
  }

  /**
   * Converts to the "lost inputs" subtype of the other exception type ({@link ExecException}) used
   * during action execution.
   *
   * <p>May not be used if this exception has been decorated with additional information from its
   * context (e.g. from {@link #setPrimaryOutputPath} or other setters) because that information
   * would be lost if so.
   */
  public LostInputsExecException toExecException() {
    Preconditions.checkState(!actionStartedEventAlreadyEmitted);
    Preconditions.checkState(primaryOutputPath == null);
    Preconditions.checkState(fileOutErr == null);
    Preconditions.checkState(!fromInputDiscovery);
    return new LostInputsExecException(lostInputs, owners, this);
  }
}
