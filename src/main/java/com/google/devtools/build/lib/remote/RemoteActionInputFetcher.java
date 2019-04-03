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
package com.google.devtools.build.lib.remote;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.MetadataProvider;
import com.google.devtools.build.lib.actions.cache.VirtualActionInput;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.Utils;
import com.google.devtools.build.lib.vfs.Path;
import io.grpc.Context;
import java.io.IOException;
import java.io.OutputStream;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import javax.annotation.concurrent.GuardedBy;

/**
 * Stages output files that are stored remotely to the local filesystem.
 *
 * <p>This is necessary for remote caching/execution when {@code
 * --experimental_remote_download_outputs=minimal} is specified.
 */
class RemoteActionInputFetcher implements ActionInputPrefetcher {

  private final Object lock = new Object();

  @GuardedBy("lock")
  private final Set<Path> downloadedPaths = new HashSet<>();

  @GuardedBy("lock")
  private final Map<Path, ListenableFuture<Void>> downloadsInProgress = new HashMap<>();

  private final AbstractRemoteActionCache remoteCache;
  private final Path execRoot;
  private final Context ctx;

  RemoteActionInputFetcher(AbstractRemoteActionCache remoteCache, Path execRoot, Context ctx) {
    this.remoteCache = Preconditions.checkNotNull(remoteCache);
    this.execRoot = Preconditions.checkNotNull(execRoot);
    this.ctx = Preconditions.checkNotNull(ctx);
  }

  /**
   * Fetches remotely stored action outputs, that are inputs to this spawn, and stores them under
   * their path in the output base.
   *
   * <p>This method blocks until all downloads have finished.
   *
   * <p>This method is safe to be called concurrently from spawn runners before running any local
   * spawn.
   */
  @Override
  public void prefetchFiles(
      Iterable<? extends ActionInput> inputs, MetadataProvider metadataProvider)
      throws IOException, InterruptedException {
    try (SilentCloseable c = Profiler.instance().profile("Remote.fetchInputs")) {
      Map<Path, ListenableFuture<Void>> downloadsToWaitFor = new HashMap<>();
      for (ActionInput input : inputs) {
        if (input instanceof VirtualActionInput) {
          VirtualActionInput paramFileActionInput = (VirtualActionInput) input;
          Path outputPath = execRoot.getRelative(paramFileActionInput.getExecPath());
          outputPath.getParentDirectory().createDirectoryAndParents();
          try (OutputStream out = outputPath.getOutputStream()) {
            paramFileActionInput.writeTo(out);
          }
        } else {
          FileArtifactValue metadata = metadataProvider.getMetadata(input);
          if (metadata == null || !metadata.isRemote()) {
            continue;
          }

          Path path = execRoot.getRelative(input.getExecPath());
          synchronized (lock) {
            if (downloadedPaths.contains(path)) {
              continue;
            }

            ListenableFuture<Void> download = downloadsInProgress.get(path);
            if (download == null) {
              Context prevCtx = ctx.attach();
              try {
                download =
                    remoteCache.downloadFile(
                        path, DigestUtil.buildDigest(metadata.getDigest(), metadata.getSize()));
                downloadsInProgress.put(path, download);
              } finally {
                ctx.detach(prevCtx);
              }
            }
            downloadsToWaitFor.putIfAbsent(path, download);
          }
        }
      }

      IOException ioException = null;
      InterruptedException interruptedException = null;
      try {
        for (Map.Entry<Path, ListenableFuture<Void>> entry : downloadsToWaitFor.entrySet()) {
          try {
            Utils.getFromFuture(entry.getValue());
            entry.getKey().setExecutable(true);
          } catch (IOException e) {
            if (e instanceof CacheNotFoundException) {
              e =
                  new IOException(
                      String.format(
                          "Failed to fetch file with hash '%s' because it does not exist remotely."
                              + " --experimental_remote_download_outputs=minimal does not work if"
                              + " your remote cache evicts files during builds.",
                          ((CacheNotFoundException) e).getMissingDigest().getHash()));
            }
            ioException = ioException == null ? e : ioException;
          } catch (InterruptedException e) {
            interruptedException = interruptedException == null ? e : interruptedException;
          }
        }
      } finally {
        synchronized (lock) {
          for (Path path : downloadsToWaitFor.keySet()) {
            downloadsInProgress.remove(path);
            downloadedPaths.add(path);
          }
        }
      }

      if (interruptedException != null) {
        throw interruptedException;
      }
      if (ioException != null) {
        throw ioException;
      }
    }
  }

  ImmutableSet<Path> downloadedFiles() {
    synchronized (lock) {
      return ImmutableSet.copyOf(downloadedPaths);
    }
  }
}
