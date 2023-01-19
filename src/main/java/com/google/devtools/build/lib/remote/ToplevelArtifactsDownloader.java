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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static com.google.common.util.concurrent.Futures.addCallback;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static com.google.devtools.build.lib.packages.TargetUtils.isTestRuleName;

import com.google.common.collect.ImmutableMap;
import com.google.common.eventbus.AllowConcurrentEvents;
import com.google.common.eventbus.Subscribe;
import com.google.common.flogger.GoogleLogger;
import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.analysis.AspectCompleteEvent;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.ConfiguredTargetValue;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.TargetCompleteEvent;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper.ArtifactsInOutputGroup;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.analysis.test.TestAttempt;
import com.google.devtools.build.lib.remote.AbstractActionInputPrefetcher.Priority;
import com.google.devtools.build.lib.remote.util.StaticMetadataProvider;
import com.google.devtools.build.lib.skyframe.ActionExecutionValue;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import com.google.devtools.build.skyframe.SkyValue;
import javax.annotation.Nullable;

/**
 * Class that uses {@link AbstractActionInputPrefetcher} to download artifacts of toplevel targets
 * in the background.
 */
public class ToplevelArtifactsDownloader {
  private enum CommandMode {
    UNKNOWN,
    BUILD,
    TEST,
    RUN;
  }

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private final CommandMode commandMode;
  private final boolean downloadToplevel;
  private final MemoizingEvaluator memoizingEvaluator;
  private final AbstractActionInputPrefetcher actionInputPrefetcher;
  private final PathToMetadataConverter pathToMetadataConverter;

  public ToplevelArtifactsDownloader(
      String commandName,
      boolean downloadToplevel,
      MemoizingEvaluator memoizingEvaluator,
      AbstractActionInputPrefetcher actionInputPrefetcher,
      PathToMetadataConverter pathToMetadataConverter) {
    switch (commandName) {
      case "build":
        this.commandMode = CommandMode.BUILD;
        break;
      case "test":
        this.commandMode = CommandMode.TEST;
        break;
      case "run":
        this.commandMode = CommandMode.RUN;
        break;
      default:
        this.commandMode = CommandMode.UNKNOWN;
    }
    this.downloadToplevel = downloadToplevel;
    this.memoizingEvaluator = memoizingEvaluator;
    this.actionInputPrefetcher = actionInputPrefetcher;
    this.pathToMetadataConverter = pathToMetadataConverter;
  }

  /**
   * Interface that converts {@link Path} to metadata {@link FileArtifactValue}.
   *
   * <p>{@link ToplevelArtifactsDownloader} is only used with {@code ActionFileSystem} together. If
   * we see a {@link Path}, its underlying file system must be {@code ActionFileSystem}. We use this
   * interface to avoid passing in the actionFs implementation.
   */
  public interface PathToMetadataConverter {
    @Nullable
    FileArtifactValue getMetadata(Path path);
  }

  @Subscribe
  @AllowConcurrentEvents
  public void onTestAttempt(TestAttempt event) {
    for (Pair<String, Path> pair : event.getFiles()) {
      Path path = checkNotNull(pair.getSecond());
      // Since the event is fired within action execution, the skyframe doesn't know the outputs of
      // test actions yet, so we can't get their metadata through skyframe. However, the fileSystem
      // of the path is an ActionFileSystem, we use it to get the metadata for this file.
      FileArtifactValue metadata = pathToMetadataConverter.getMetadata(path);
      if (metadata != null) {
        ListenableFuture<Void> future =
            actionInputPrefetcher.downloadFileAsync(path.asFragment(), metadata, Priority.LOW);
        addCallback(
            future,
            new FutureCallback<Void>() {
              @Override
              public void onSuccess(Void unused) {}

              @Override
              public void onFailure(Throwable throwable) {
                logger.atWarning().withCause(throwable).log(
                    "Failed to download test output %s.", path);
              }
            },
            directExecutor());
      }
    }
  }

  @Subscribe
  @AllowConcurrentEvents
  public void onAspectComplete(AspectCompleteEvent event) {
    if (!shouldDownloadToplevelOutputs(event.getAspectKey().getBaseConfiguredTargetKey())) {
      return;
    }

    if (event.failed()) {
      return;
    }

    downloadTargetOutputs(event.getOutputGroups(), /* runfiles = */ null);
  }

  @Subscribe
  @AllowConcurrentEvents
  public void onTargetComplete(TargetCompleteEvent event) {
    if (!shouldDownloadToplevelOutputs(event.getConfiguredTargetKey())) {
      return;
    }

    if (event.failed()) {
      return;
    }

    downloadTargetOutputs(
        event.getOutputs(),
        event.getExecutableTargetData().getRunfiles());
  }

  private boolean shouldDownloadToplevelOutputs(ConfiguredTargetKey configuredTargetKey) {
    switch (commandMode) {
      case RUN:
        // Always download outputs of toplevel targets in RUN mode
        return true;
      case TEST:
        // Do not download test binary in test mode.
        try {
          var configuredTargetValue =
              (ConfiguredTargetValue) memoizingEvaluator.getExistingValue(configuredTargetKey);
          if (configuredTargetValue == null) {
            return false;
          }
          ConfiguredTarget configuredTarget = configuredTargetValue.getConfiguredTarget();
          if (configuredTarget instanceof RuleConfiguredTarget) {
            var ruleConfiguredTarget = (RuleConfiguredTarget) configuredTarget;
            var isTestRule = isTestRuleName(ruleConfiguredTarget.getRuleClassString());
            return !isTestRule && downloadToplevel;
          }
          return downloadToplevel;
        } catch (InterruptedException ignored) {
          Thread.currentThread().interrupt();
          return false;
        }
      default:
        return downloadToplevel;
    }
  }

  private void downloadTargetOutputs(
      ImmutableMap<String, ArtifactsInOutputGroup> outputGroups, @Nullable Runfiles runfiles) {

    var builder = ImmutableMap.<ActionInput, FileArtifactValue>builder();
    try {
      for (ArtifactsInOutputGroup outputs : outputGroups.values()) {
        if (!outputs.areImportant()) {
          continue;
        }
        for (Artifact output : outputs.getArtifacts().toList()) {
          appendArtifact(output, builder);
        }
      }

      appendRunfiles(runfiles, builder);
    } catch (InterruptedException ignored) {
      Thread.currentThread().interrupt();
      return;
    }

    var outputsAndMetadata = builder.buildKeepingLast();
    ListenableFuture<Void> future =
        actionInputPrefetcher.prefetchFiles(
            outputsAndMetadata.keySet().stream()
                .filter(ToplevelArtifactsDownloader::isNonTreeArtifact)
                .collect(toImmutableSet()),
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

  private static boolean isNonTreeArtifact(ActionInput actionInput) {
    return !(actionInput instanceof Artifact && ((Artifact) actionInput).isTreeArtifact());
  }

  private void appendRunfiles(
      @Nullable Runfiles runfiles, ImmutableMap.Builder<ActionInput, FileArtifactValue> builder)
      throws InterruptedException {
    if (runfiles == null) {
      return;
    }

    for (Artifact runfile : runfiles.getArtifacts().toList()) {
      appendArtifact(runfile, builder);
    }
  }

  private void appendArtifact(
      Artifact artifact, ImmutableMap.Builder<ActionInput, FileArtifactValue> builder)
      throws InterruptedException {
    SkyValue value = memoizingEvaluator.getExistingValue(Artifact.key(artifact));
    if (value instanceof ActionExecutionValue) {
      FileArtifactValue metadata = ((ActionExecutionValue) value).getAllFileValues().get(artifact);
      if (metadata != null) {
        builder.put(artifact, metadata);
      }
    } else if (value instanceof TreeArtifactValue) {
      builder.put(artifact, ((TreeArtifactValue) value).getMetadata());
      builder.putAll(((TreeArtifactValue) value).getChildValues());
    }
  }
}
