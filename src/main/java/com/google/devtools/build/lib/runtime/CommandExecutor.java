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
package com.google.devtools.build.lib.runtime;

import com.google.devtools.build.lib.runtime.BlazeCommandDispatcher.LockingMode;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.InvocationPolicy;
import com.google.devtools.build.lib.server.ServerCommand;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.io.OutErr;
import java.util.List;
import java.util.Optional;

/**
 * Executes a Blaze command.
 *
 * <p>This is the common execution path between the gRPC server and the legacy AF_UNIX server.
 */
public class CommandExecutor implements ServerCommand {
  private final BlazeCommandDispatcher dispatcher;

  CommandExecutor(BlazeCommandDispatcher dispatcher) {
    this.dispatcher = dispatcher;
  }

  @Override
  public BlazeCommandResult exec(
      InvocationPolicy invocationPolicy,
      List<String> args,
      OutErr outErr,
      LockingMode lockingMode,
      String clientDescription,
      long firstContactTime,
      Optional<List<Pair<String, String>>> startupOptionsTaggedWithBazelRc)
      throws InterruptedException {

    return dispatcher.exec(
        invocationPolicy,
        args,
        outErr,
        lockingMode,
        clientDescription,
        firstContactTime,
        startupOptionsTaggedWithBazelRc);
  }
}
