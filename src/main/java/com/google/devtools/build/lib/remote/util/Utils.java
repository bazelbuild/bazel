// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote.util;

import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.Digest;
import com.google.common.base.Preconditions;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableSet;
import com.google.common.util.concurrent.AsyncCallable;
import com.google.common.util.concurrent.FluentFuture;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnMetrics;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.SpawnResult.Status;
import com.google.devtools.build.lib.authandtls.CallCredentialsProvider;
import com.google.devtools.build.lib.remote.common.CacheNotFoundException;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient.ActionKey;
import com.google.devtools.build.lib.remote.options.RemoteOutputsMode;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.Spawn.Code;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Collection;
import java.util.Collections;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.function.BiFunction;
import javax.annotation.Nullable;

/** Utility methods for the remote package. * */
public final class Utils {

  private Utils() {}

  /**
   * Returns the result of a {@link ListenableFuture} if successful, or throws any checked {@link
   * Exception} directly if it's an {@link IOException} or else wraps it in an {@link IOException}.
   */
  public static <T> T getFromFuture(ListenableFuture<T> f)
      throws IOException, InterruptedException {
    try {
      return f.get();
    } catch (ExecutionException e) {
      Throwable cause = e.getCause();
      if (cause instanceof InterruptedException) {
        throw (InterruptedException) cause;
      }
      if (cause instanceof IOException) {
        throw (IOException) cause;
      }
      if (cause instanceof RuntimeException) {
        throw (RuntimeException) cause;
      }
      throw new IOException(cause);
    }
  }

  /**
   * Returns the (exec root relative) path of a spawn output that should be made available via
   * {@link SpawnResult#getInMemoryOutput(ActionInput)}.
   */
  @Nullable
  public static PathFragment getInMemoryOutputPath(Spawn spawn) {
    String outputPath =
        spawn.getExecutionInfo().get(ExecutionRequirements.REMOTE_EXECUTION_INLINE_OUTPUTS);
    if (outputPath != null) {
      return PathFragment.create(outputPath);
    }
    return null;
  }

  /** Constructs a {@link SpawnResult}. */
  public static SpawnResult createSpawnResult(
      int exitCode,
      boolean cacheHit,
      String runnerName,
      @Nullable InMemoryOutput inMemoryOutput,
      SpawnMetrics spawnMetrics,
      String mnemonic) {
    SpawnResult.Builder builder =
        new SpawnResult.Builder()
            .setStatus(exitCode == 0 ? Status.SUCCESS : Status.NON_ZERO_EXIT)
            .setExitCode(exitCode)
            .setRunnerName(cacheHit ? runnerName + " cache hit" : runnerName)
            .setCacheHit(cacheHit)
            .setSpawnMetrics(spawnMetrics)
            .setRemote(true);
    if (exitCode != 0) {
      builder.setFailureDetail(
          FailureDetail.newBuilder()
              .setMessage(mnemonic + " returned a non-zero exit code when running remotely")
              .setSpawn(FailureDetails.Spawn.newBuilder().setCode(Code.NON_ZERO_EXIT))
              .build());
    }
    if (inMemoryOutput != null) {
      builder.setInMemoryOutput(inMemoryOutput.getOutput(), inMemoryOutput.getContents());
    }
    return builder.build();
  }

  /** Returns {@code true} if all spawn outputs should be downloaded to disk. */
  public static boolean shouldDownloadAllSpawnOutputs(
      RemoteOutputsMode remoteOutputsMode, int exitCode, boolean hasTopLevelOutputs) {
    return remoteOutputsMode.downloadAllOutputs()
        ||
        // In case the action failed, download all outputs. It might be helpful for debugging
        // and there is no point in injecting output metadata of a failed action.
        exitCode != 0
        ||
        // If one output of a spawn is a top level output then download all outputs. Spawns
        // are typically structured in a way that either all or no outputs are top level and
        // it's much simpler to implement under this assumption.
        (remoteOutputsMode.downloadToplevelOutputsOnly() && hasTopLevelOutputs);
  }

  /** Returns {@code true} if outputs contains one or more top level outputs. */
  public static boolean hasFilesToDownload(
      Collection<? extends ActionInput> outputs, ImmutableSet<ActionInput> filesToDownload) {
    if (filesToDownload.isEmpty()) {
      return false;
    }
    return !Collections.disjoint(outputs, filesToDownload);
  }

  public static String grpcAwareErrorMessage(IOException e) {
    io.grpc.Status errStatus = io.grpc.Status.fromThrowable(e);
    if (!errStatus.getCode().equals(io.grpc.Status.UNKNOWN.getCode())) {
      // If the error originated in the gRPC library then display it as "STATUS: error message"
      // to the user
      return String.format("%s: %s", errStatus.getCode().name(), errStatus.getDescription());
    }
    return e.getMessage();
  }

  @SuppressWarnings("ProtoParseWithRegistry")
  public static ListenableFuture<ActionResult> downloadAsActionResult(
      ActionKey actionDigest,
      BiFunction<Digest, OutputStream, ListenableFuture<Void>> downloadFunction) {
    ByteArrayOutputStream data = new ByteArrayOutputStream(/* size= */ 1024);
    ListenableFuture<Void> download = downloadFunction.apply(actionDigest.getDigest(), data);
    return FluentFuture.from(download)
        .transformAsync(
            (v) -> {
              try {
                return Futures.immediateFuture(ActionResult.parseFrom(data.toByteArray()));
              } catch (InvalidProtocolBufferException e) {
                return Futures.immediateFailedFuture(e);
              }
            },
            MoreExecutors.directExecutor())
        .catching(CacheNotFoundException.class, (e) -> null, MoreExecutors.directExecutor());
  }

  public static void verifyBlobContents(Digest expected, Digest actual) throws IOException {
    if (!expected.equals(actual)) {
      String msg =
          String.format(
              "Output download failed: Expected digest '%s/%d' does not match "
                  + "received digest '%s/%d'.",
              expected.getHash(), expected.getSizeBytes(), actual.getHash(), actual.getSizeBytes());
      throw new IOException(msg);
    }
  }

  /** An in-memory output file. */
  public static final class InMemoryOutput {
    private final ActionInput output;
    private final ByteString contents;

    public InMemoryOutput(ActionInput output, ByteString contents) {
      this.output = output;
      this.contents = contents;
    }

    public ActionInput getOutput() {
      return output;
    }

    public ByteString getContents() {
      return contents;
    }
  }

  /**
   * Call an asynchronous code block. If the block throws unauthenticated error, refresh the
   * credentials using {@link CallCredentialsProvider} and call it again.
   *
   * <p>If any other exception thrown by the code block, it will be caught and wrapped in the
   * returned {@link ListenableFuture}.
   */
  public static <V> ListenableFuture<V> refreshIfUnauthenticatedAsync(
      AsyncCallable<V> call, CallCredentialsProvider callCredentialsProvider) {
    Preconditions.checkNotNull(call);
    Preconditions.checkNotNull(callCredentialsProvider);

    try {
      return Futures.catchingAsync(
          call.call(),
          Throwable.class,
          (e) -> refreshIfUnauthenticatedAsyncOnException(e, call, callCredentialsProvider),
          MoreExecutors.directExecutor());
    } catch (Throwable t) {
      return refreshIfUnauthenticatedAsyncOnException(t, call, callCredentialsProvider);
    }
  }

  private static <V> ListenableFuture<V> refreshIfUnauthenticatedAsyncOnException(
      Throwable t, AsyncCallable<V> call, CallCredentialsProvider callCredentialsProvider) {
    io.grpc.Status status = io.grpc.Status.fromThrowable(t);
    if (status != null
        && (status.getCode() == io.grpc.Status.Code.UNAUTHENTICATED
            || status.getCode() == io.grpc.Status.Code.PERMISSION_DENIED)) {
      try {
        callCredentialsProvider.refresh();
        return call.call();
      } catch (Throwable tt) {
        t.addSuppressed(tt);
      }
    }

    return Futures.immediateFailedFuture(t);
  }

  /** Same as {@link #refreshIfUnauthenticatedAsync} but calling a synchronous code block. */
  public static <V> V refreshIfUnauthenticated(
      Callable<V> call, CallCredentialsProvider callCredentialsProvider)
      throws IOException, InterruptedException {
    Preconditions.checkNotNull(call);
    Preconditions.checkNotNull(callCredentialsProvider);

    try {
      return call.call();
    } catch (Exception e) {
      io.grpc.Status status = io.grpc.Status.fromThrowable(e);
      if (status != null
          && (status.getCode() == io.grpc.Status.Code.UNAUTHENTICATED
              || status.getCode() == io.grpc.Status.Code.PERMISSION_DENIED)) {
        try {
          callCredentialsProvider.refresh();
          return call.call();
        } catch (Exception ex) {
          e.addSuppressed(ex);
        }
      }

      Throwables.throwIfInstanceOf(e, IOException.class);
      Throwables.throwIfInstanceOf(e, InterruptedException.class);
      Throwables.throwIfUnchecked(e);
      throw new AssertionError(e);
    }
  }
}
