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

import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.SpawnResult.Status;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.protobuf.ByteString;
import java.io.IOException;
import java.util.concurrent.ExecutionException;
import javax.annotation.Nullable;

/** Utility methods for the remote package. * */
public class Utils {

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
      if (e.getCause() instanceof IOException) {
        throw (IOException) e.getCause();
      }
      if (e.getCause() instanceof RuntimeException) {
        throw (RuntimeException) e.getCause();
      }
      throw new IOException(e.getCause());
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
  public static SpawnResult createSpawnResult(int exitCode, boolean cacheHit, String runnerName,
      @Nullable InMemoryOutput inMemoryOutput) {
    SpawnResult.Builder builder = new SpawnResult.Builder()
        .setStatus(exitCode == 0 ? Status.SUCCESS : Status.NON_ZERO_EXIT)
        .setExitCode(exitCode)
        .setRunnerName(cacheHit ? runnerName + " cache hit" : runnerName)
        .setCacheHit(cacheHit);
    if (inMemoryOutput != null) {
      builder.setInMemoryOutput(inMemoryOutput.getOutput(), inMemoryOutput.getContents());
    }
    return builder.build();
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
}
