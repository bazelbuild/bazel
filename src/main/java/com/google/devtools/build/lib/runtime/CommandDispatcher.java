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
package com.google.devtools.build.lib.runtime;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.InvocationPolicy;
import com.google.devtools.build.lib.server.IdleTask;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.io.CommandExtensionReporter;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.protobuf.Any;
import java.util.List;
import java.util.Optional;
import java.util.function.Supplier;

/**
 * Dispatches to the commands; that is, given a command line, this abstraction looks up the
 * appropriate command object, parses the options required by the object, and calls its exec method.
 */
public interface CommandDispatcher {

  /** What to do if the command lock is not available. */
  enum LockingMode {
    WAIT, // Wait until it is available
    ERROR_OUT, // Return with an error
  }

  /** How much output to emit on the console. */
  enum UiVerbosity {
    QUIET, // Only errors
    NORMAL, // Everything
  }

  /**
   * Executes a single command. Returns a {@link BlazeCommandResult} to indicate either an exit
   * code, the desire to shut down the server, or that a given binary should be executed by the
   * client.
   */
  BlazeCommandResult exec(
      InvocationPolicy invocationPolicy,
      List<String> args,
      OutErr outErr,
      LockingMode lockingMode,
      UiVerbosity uiVerbosity,
      String clientDescription,
      long firstContactTimeMillis,
      Optional<List<Pair<String, String>>> startupOptionsTaggedWithBazelRc,
      Supplier<ImmutableList<IdleTask.Result>> idleTaskResultsSupplier,
      List<Any> commandExtensions,
      CommandExtensionReporter commandExtensionReporter)
      throws InterruptedException;
}
