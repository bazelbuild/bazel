// Copyright 2025 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe.serialization.analysis;

import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.protobuf.ByteString;
import java.util.Collection;

/** Interface to the remote analysis cache. */
public interface RemoteAnalysisCacheClient {

  /** Timeout when accessing the future in order to shutdown the client. */
  int SHUTDOWN_TIMEOUT_IN_SECONDS = 5;

  /** Usage statistics. */
  record Stats(long bytesSent, long bytesReceived, long requestsSent, long batches) {}

  /** Looks up an entry in the remote analysis cache based on a serialized key. */
  ListenableFuture<ByteString> lookup(ByteString key);

  /** Returns the usage statistics. */
  Stats getStats();

  /** Adds the cached targets into the metadata table */
  ListenableFuture<Boolean> addTopLevelTargets(
      String invocationId,
      long evaluatingVersion,
      String configurationHash,
      String bazelVersion,
      Collection<String> targets,
      Collection<String> configFlags);

  /** Looks up the targets in the metadata table */
  void lookupTopLevelTargets(
      long evaluatingVersion,
      String configurationHash,
      String bazelVersion,
      EventHandler eventHandler,
      Runnable bailOutCallback)
      throws InterruptedException;

  void shutdown();
}
