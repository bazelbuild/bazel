// Copyright 2026 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis.actions;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.ActionResult;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.server.FailureDetails.Execution;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.Fingerprint;
import java.io.IOException;
import javax.annotation.Nullable;

/**
 * Action that <em>declares</em> its output to be a copy of its input: the same content,
 * re-addressed at the output's exec path, sharing the input's digest.
 *
 * <p>This action performs no spawn and writes nothing to the filesystem. It only injects the
 * output's metadata as a {@linkplain FileArtifactValue#createForContentCopy content-copy
 * FileArtifactValue}, which points at the input's content and requests {@linkplain
 * FileArtifactValue#isContentCopy content materialization}. Deciding <em>how</em> to place
 * that content (hard link, copy, reflink, remote digest reuse) is the execution strategy's job when
 * it stages the artifact -- this action never makes that decision, exactly as path mapping keeps
 * laydown out of the action.
 *
 * <p>Contrast {@link SymlinkAction}, which is realized as a followable symlink; a copy is realized
 * as content so its {@code realpath} is stable within the consuming tree. Strategies that cannot
 * materialize content do not honor the request; that is an accepted limitation.
 */
public final class CopyAction extends AbstractAction {
  private static final String GUID = "6b4f2c11-0e7a-4d3f-9a2c-8b1d5e6f7a90";

  @Nullable private final String progressMessage;

  public static CopyAction create(
      ActionOwner owner, Artifact input, Artifact output, String progressMessage) {
    return new CopyAction(owner, input, output, progressMessage);
  }

  private CopyAction(
      ActionOwner owner, Artifact primaryInput, Artifact primaryOutput, String progressMessage) {
    super(
        owner,
        NestedSetBuilder.create(Order.STABLE_ORDER, checkNotNull(primaryInput)),
        ImmutableSet.of(primaryOutput));
    this.progressMessage = progressMessage;
  }

  @Override
  public ActionResult execute(ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException {
    Artifact input = getPrimaryInput();
    FileArtifactValue inputMetadata;
    try {
      inputMetadata =
          checkNotNull(
              actionExecutionContext.getInputMetadataProvider().getInputMetadata(input),
              "missing metadata for %s",
              input);
    } catch (IOException e) {
      String message =
          String.format(
              "failed to read metadata of '%s' for copy '%s': %s",
              input.getExecPathString(), getPrimaryOutput().getExecPathString(), e.getMessage());
      throw new ActionExecutionException(
          message, e, this, false, createDetailedExitCode(message));
    }

    // Declaration only: no spawn, no filesystem write. The output shares the input's content
    // (digest) and is flagged for content materialization; the executor stages it accordingly.
    actionExecutionContext
        .getOutputMetadataStore()
        .injectFile(
            getPrimaryOutput(),
            FileArtifactValue.createForContentCopy(inputMetadata, input.getPath().asFragment()));
    return ActionResult.EMPTY;
  }

  @Override
  protected void computeKey(
      ActionKeyContext actionKeyContext,
      @Nullable InputMetadataProvider inputMetadataProvider,
      Fingerprint fp) {
    fp.addString(GUID);
  }

  @Override
  protected String getRawProgressMessage() {
    return progressMessage;
  }

  @Override
  public String getMnemonic() {
    return "Copy";
  }

  @Override
  public boolean mayInsensitivelyPropagateInputs() {
    return true;
  }

  @Override
  public PlatformInfo getExecutionPlatform() {
    return PlatformInfo.EMPTY_PLATFORM_INFO;
  }

  @Override
  public ImmutableMap<String, String> getExecProperties() {
    return ImmutableMap.of();
  }

  private static DetailedExitCode createDetailedExitCode(String message) {
    return DetailedExitCode.of(
        FailureDetail.newBuilder()
            .setMessage(message)
            .setExecution(Execution.newBuilder().setCode(Execution.Code.SOURCE_INPUT_IO_EXCEPTION))
            .build());
  }
}
