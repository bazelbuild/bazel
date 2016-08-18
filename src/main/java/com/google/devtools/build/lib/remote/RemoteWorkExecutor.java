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

package com.google.devtools.build.lib.remote;

import com.google.common.collect.ImmutableMap;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.remote.RemoteProtocol.RemoteWorkResponse;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.util.Collection;

/**
 * Interface for exeucting work remotely.
 */
@ThreadCompatible
public interface RemoteWorkExecutor {
  /**
   * Submit the work to this work executor. The output of running this action should be written to
   * {@link RemoteActionCache} indexed by {@code actionOutputKey}.
   *
   * <p>Returns a future for the response of this work request.
   */
  ListenableFuture<RemoteWorkResponse> executeRemotely(
      Path execRoot,
      ActionInputFileCache cache,
      String actionOutputKey,
      Collection<String> arguments,
      Collection<ActionInput> inputs,
      ImmutableMap<String, String> environment,
      Collection<? extends ActionInput> outputs,
      int timeout)
      throws IOException, WorkTooLargeException, InterruptedException;
}
