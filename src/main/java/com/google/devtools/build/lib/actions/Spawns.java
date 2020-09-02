// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.Spawn.Code;
import com.google.devtools.build.lib.util.CommandDescriptionForm;
import com.google.devtools.build.lib.util.CommandFailureUtils;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.time.Duration;
import java.util.Collection;
import java.util.Map;

/** Helper methods relating to implementations of {@link Spawn}. */
public final class Spawns {
  private Spawns() {}

  /**
   * Returns {@code true} if the result of {@code spawn} may be cached.
   */
  public static boolean mayBeCached(Spawn spawn) {
    return !spawn.getExecutionInfo().containsKey(ExecutionRequirements.NO_CACHE)
        && !spawn.getExecutionInfo().containsKey(ExecutionRequirements.LOCAL);
  }

  /** Returns {@code true} if the result of {@code spawn} may be cached remotely. */
  public static boolean mayBeCachedRemotely(Spawn spawn) {
    return mayBeCached(spawn)
        && !spawn.getExecutionInfo().containsKey(ExecutionRequirements.NO_REMOTE)
        && !spawn.getExecutionInfo().containsKey(ExecutionRequirements.NO_REMOTE_CACHE);
  }

  /** Returns {@code true} if {@code spawn} may be executed remotely. */
  public static boolean mayBeExecutedRemotely(Spawn spawn) {
    return !spawn.getExecutionInfo().containsKey(ExecutionRequirements.LOCAL)
        && !spawn.getExecutionInfo().containsKey(ExecutionRequirements.NO_REMOTE)
        && !spawn.getExecutionInfo().containsKey(ExecutionRequirements.NO_REMOTE_EXEC);
  }

  /** Returns {@code true} if {@code spawn} may be executed locally. */
  public static boolean mayBeExecutedLocally(Spawn spawn) {
    return !spawn.getExecutionInfo().containsKey(ExecutionRequirements.NO_LOCAL);
  }

  /** Returns whether a Spawn can be executed in a sandbox environment. */
  public static boolean mayBeSandboxed(Spawn spawn) {
    return !spawn.getExecutionInfo().containsKey(ExecutionRequirements.LEGACY_NOSANDBOX)
        && !spawn.getExecutionInfo().containsKey(ExecutionRequirements.NO_SANDBOX)
        && !spawn.getExecutionInfo().containsKey(ExecutionRequirements.LOCAL);
  }

  /** Returns whether a Spawn needs network access in order to run successfully. */
  public static boolean requiresNetwork(Spawn spawn, boolean defaultSandboxDisallowNetwork) {
    if (spawn.getExecutionInfo().containsKey(ExecutionRequirements.BLOCK_NETWORK)) {
      return false;
    }
    if (spawn.getExecutionInfo().containsKey(ExecutionRequirements.REQUIRES_NETWORK)) {
      return true;
    }

    return defaultSandboxDisallowNetwork;
  }

  /**
   * Returns whether a Spawn claims to support being executed with the persistent worker strategy
   * according to its execution info tags.
   */
  public static boolean supportsWorkers(Spawn spawn) {
    return "1".equals(spawn.getExecutionInfo().get(ExecutionRequirements.SUPPORTS_WORKERS));
  }

  /**
   * Returns whether a Spawn claims to support being executed with the persistent multiplex worker
   * strategy according to its execution info tags.
   */
  public static boolean supportsMultiplexWorkers(Spawn spawn) {
    return "1"
        .equals(spawn.getExecutionInfo().get(ExecutionRequirements.SUPPORTS_MULTIPLEX_WORKERS));
  }

  /**
   * Returns which worker protocol format a Spawn claims a persistent worker uses. Defaults to proto
   * if the protocol format is not specified.
   */
  public static ExecutionRequirements.WorkerProtocolFormat getWorkerProtocolFormat(Spawn spawn)
      throws IOException {
    String protocolFormat =
        spawn.getExecutionInfo().get(ExecutionRequirements.REQUIRES_WORKER_PROTOCOL);

    if (protocolFormat != null) {
      switch (protocolFormat) {
        case "json":
          return ExecutionRequirements.WorkerProtocolFormat.JSON;
        case "proto":
          return ExecutionRequirements.WorkerProtocolFormat.PROTO;
        default:
          throw new IOException(
              "requires-worker-protocol must be set to a valid worker protocol format: json or"
                  + " proto");
      }
    } else {
      return ExecutionRequirements.WorkerProtocolFormat.PROTO;
    }
  }

  /** Returns the mnemonic that should be used in the worker's key. */
  public static String getWorkerKeyMnemonic(Spawn spawn) {
    String customValue = spawn.getExecutionInfo().get(ExecutionRequirements.WORKER_KEY_MNEMONIC);
    return customValue != null ? customValue : spawn.getMnemonic();
  }

  /**
   * Parse the timeout key in the spawn execution info, if it exists. Otherwise, return {@link
   * Duration#ZERO}.
   */
  public static Duration getTimeout(Spawn spawn) throws ExecException {
    return getTimeout(spawn, Duration.ZERO);
  }

  /**
   * Parse the timeout key in the spawn execution info, if it exists. Otherwise, return
   * defaultTimeout, or {@code Duration.ZERO} if that is null.
   */
  public static Duration getTimeout(Spawn spawn, Duration defaultTimeout) throws ExecException {
    String timeoutStr = spawn.getExecutionInfo().get(ExecutionRequirements.TIMEOUT);
    if (timeoutStr == null) {
      return defaultTimeout == null ? Duration.ZERO : defaultTimeout;
    }
    try {
      return Duration.ofSeconds(Integer.parseInt(timeoutStr));
    } catch (NumberFormatException e) {
      throw new UserExecException(
          e,
          FailureDetail.newBuilder()
              .setMessage("could not parse timeout")
              .setSpawn(FailureDetails.Spawn.newBuilder().setCode(Code.INVALID_TIMEOUT))
              .build());
    }
  }

  /**
   * Returns whether a local {@link Spawn} runner implementation should prefetch the inputs before
   * execution, based on the spawns execution info.
   */
  public static boolean shouldPrefetchInputsForLocalExecution(Spawn spawn) {
    String disablePrefetchRequest =
        spawn.getExecutionInfo().get(ExecutionRequirements.DISABLE_LOCAL_PREFETCH);
    return (disablePrefetchRequest == null) || disablePrefetchRequest.equals("0");
  }

  /** Convert a spawn into a Bourne shell command. */
  public static String asShellCommand(Spawn spawn, Path workingDirectory, boolean prettyPrintArgs) {
    return asShellCommand(
        spawn.getArguments(),
        workingDirectory,
        spawn.getEnvironment(),
        prettyPrintArgs);
  }

  /** Convert a working dir + environment map + arg list into a Bourne shell command. */
  public static String asShellCommand(
      Collection<String> arguments,
      Path workingDirectory,
      Map<String, String> environment,
      boolean prettyPrintArgs) {

    // We print this command out in such a way that it can safely be
    // copied+pasted as a Bourne shell command.  This is extremely valuable for
    // debugging.
    return CommandFailureUtils.describeCommand(
        CommandDescriptionForm.COMPLETE,
        prettyPrintArgs,
        arguments,
        environment,
        workingDirectory.getPathString());
  }
}
