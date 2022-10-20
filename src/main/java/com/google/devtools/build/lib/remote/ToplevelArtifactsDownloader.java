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
package com.google.devtools.build.lib.remote;

import static com.google.common.util.concurrent.Futures.addCallback;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;

import com.google.common.collect.ImmutableMap;
import com.google.common.eventbus.AllowConcurrentEvents;
import com.google.common.eventbus.Subscribe;
import com.google.common.flogger.GoogleLogger;
import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CompletionContext;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.analysis.AspectCompleteEvent;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.TargetCompleteEvent;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper.ArtifactsInOutputGroup;
import com.google.devtools.build.lib.remote.AbstractActionInputPrefetcher.Priority;
import com.google.devtools.build.lib.remote.util.StaticMetadataProvider;
import com.google.devtools.build.lib.skyframe.ActionExecutionValue;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import javax.annotation.Nullable;

/**
 * Class that uses {@link AbstractActionInputPrefetcher} to download artifacts of toplevel targets
 * in the background.
 */
public class ToplevelArtifactsDownloader {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private final MemoizingEvaluator memoizingEvaluator;
  private final AbstractActionInputPrefetcher actionInputPrefetcher;

  public ToplevelArtifactsDownloader(
      MemoizingEvaluator memoizingEvaluator, AbstractActionInputPrefetcher actionInputPrefetcher) {
    this.memoizingEvaluator = memoizingEvaluator;
    this.actionInputPrefetcher = actionInputPrefetcher;
  }

  @Subscribe
  @AllowConcurrentEvents
  public void onAspectComplete(AspectCompleteEvent event) {
    if (event.failed()) {
      return;
    }

    downloadTargetOutputs(
        event.getCompletionContext(), event.getOutputGroups(), /* runfiles = */ null);
  }

  @Subscribe
  @AllowConcurrentEvents
  public void onTargetComplete(TargetCompleteEvent event) {
    if (event.failed()) {
      return;
    }

    downloadTargetOutputs(
        event.getCompletionContext(),
        event.getOutputs(),
        event.getExecutableTargetData().getRunfiles());
  }

  private void downloadTargetOutputs(
      CompletionContext completionContext,
      ImmutableMap<String, ArtifactsInOutputGroup> outputGroups,
      @Nullable Runfiles runfiles) {

    var builder = ImmutableMap.<ActionInput, FileArtifactValue>builder();
    for (ArtifactsInOutputGroup outputs : outputGroups.values()) {
      if (!outputs.areImportant()) {
        continue;
      }
      for (Artifact output : outputs.getArtifacts().toList()) {
        var metadata = completionContext.getFileArtifactValue(output);
        if (metadata != null) {
          builder.put(output, metadata);
        }
      }
    }

    try {
      appendRunfiles(runfiles, builder);
    } catch (InterruptedException ignored) {
      Thread.currentThread().interrupt();
      return;
    }

    var outputsAndMetadata = builder.buildKeepingLast();
    ListenableFuture<Void> future =
        actionInputPrefetcher.prefetchFiles(
            outputsAndMetadata.keySet(),
            new StaticMetadataProvider(outputsAndMetadata),
            Priority.LOW);

    addCallback(
        future,
        new FutureCallback<Void>() {
          @Override
          public void onSuccess(Void unused) {}

          @Override
          public void onFailure(Throwable throwable) {
            logger.atWarning().withCause(throwable).log("Failed to download toplevel artifacts.");
          }
        },
        directExecutor());
  }

  private void appendRunfiles(
      @Nullable Runfiles runfiles, ImmutableMap.Builder<ActionInput, FileArtifactValue> builder)
      throws InterruptedException {
    if (runfiles == null) {
      return;
    }

    for (Artifact runfile : runfiles.getArtifacts().toList()) {
      var actionExecutionValue =
          (ActionExecutionValue) memoizingEvaluator.getExistingValue(Artifact.key(runfile));
      if (actionExecutionValue != null) {
        if (runfile.isTreeArtifact()) {
          TreeArtifactValue metadata = actionExecutionValue.getAllTreeArtifactValues().get(runfile);
          if (metadata != null) {
            builder.putAll(metadata.getChildValues());
          }
        } else {
          FileArtifactValue metadata = actionExecutionValue.getAllFileValues().get(runfile);
          if (metadata != null) {
            builder.put(runfile, metadata);
          }
        }
      }
    }
  }
}
