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
import static com.google.common.util.concurrent.Futures.addCallback;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;

import com.google.common.collect.ImmutableList;
import com.google.common.eventbus.AllowConcurrentEvents;
import com.google.common.eventbus.Subscribe;
import com.google.common.flogger.GoogleLogger;
import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher.MetadataSupplier;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher.Priority;
import com.google.devtools.build.lib.analysis.test.CoverageReport;
import com.google.devtools.build.lib.analysis.test.TestAttempt;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import javax.annotation.Nullable;

/**
 * Class that uses {@link AbstractActionInputPrefetcher} to download artifacts of toplevel targets
 * in the background.
 */
public class ToplevelArtifactsDownloader {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private final AbstractActionInputPrefetcher actionInputPrefetcher;
  private final PathFragment execRoot;
  private final PathToMetadataSupplier pathToMetadataSupplier;

  public ToplevelArtifactsDownloader(
      AbstractActionInputPrefetcher actionInputPrefetcher,
      PathFragment execRoot,
      PathToMetadataSupplier pathToMetadataSupplier) {
    this.actionInputPrefetcher = actionInputPrefetcher;
    this.execRoot = execRoot;
    this.pathToMetadataSupplier = pathToMetadataSupplier;
  }

  /**
   * Interface that converts a {@link Path} into a {@link MetadataSupplier} suitable for retrieving
   * metadata for that path.
   *
   * <p>{@link ToplevelArtifactsDownloader} may only used in conjunction with filesystems that
   * implement {@link MetadataSupplier}.
   */
  public interface PathToMetadataSupplier {
    @Nullable
    MetadataSupplier getMetadataSupplier(Path path);
  }

  private void downloadTestOutput(Path path) {
    // Since the event is fired within action execution, the skyframe doesn't know the outputs of
    // test actions yet, so we can't get their metadata through skyframe. However, since the path
    // belongs to a filesystem that implements MetadataProvider, we use it to get the metadata.
    //
    // If the test hit action cache, the filesystem is local filesystem because the actual test
    // action didn't get the chance to execute. In this case the MetadataProvider is null, which
    // is fine because test outputs are already downloaded (otherwise the action cache wouldn't
    // have been hit).
    MetadataSupplier metadataSupplier = pathToMetadataSupplier.getMetadataSupplier(path);
    if (metadataSupplier != null) {
      // RemoteActionFileSystem#getInput returns null for undeclared test outputs.
      ActionInput input = ActionInputHelper.fromPath(path.asFragment().relativeTo(execRoot));
      ListenableFuture<Void> future =
          actionInputPrefetcher.prefetchFiles(
              ImmutableList.of(input), metadataSupplier, Priority.LOW);
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

  @Subscribe
  @AllowConcurrentEvents
  public void onTestAttempt(TestAttempt event) {
    for (Pair<String, Path> pair : event.getFiles()) {
      Path path = checkNotNull(pair.getSecond());
      downloadTestOutput(path);
    }
  }

  @Subscribe
  @AllowConcurrentEvents
  public void onCoverageReport(CoverageReport event) {
    for (var file : event.getFiles()) {
      downloadTestOutput(file);
    }
  }
}
