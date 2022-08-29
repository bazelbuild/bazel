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
package com.google.devtools.build.lib.runtime;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;

import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.devtools.build.lib.concurrent.ExecutorUtil;
import com.google.devtools.build.lib.util.AbruptExitException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import javax.annotation.Nullable;

/** A {@link BlazeModule} that waits for submitted tasks to terminate after every command. */
public class BlockWaitingModule extends BlazeModule {
  @Nullable private ExecutorService executorService;

  @Override
  public void beforeCommand(CommandEnvironment env) throws AbruptExitException {
    checkState(executorService == null, "executorService must be null");

    executorService =
        Executors.newCachedThreadPool(
            new ThreadFactoryBuilder().setNameFormat("block-waiting-%d").build());
  }

  @SuppressWarnings("FutureReturnValueIgnored")
  public void submit(Runnable task) {
    checkNotNull(executorService, "executorService must not be null");

    executorService.submit(task);
  }

  @Override
  public void afterCommand() throws AbruptExitException {
    checkNotNull(executorService, "executorService must not be null");

    if (ExecutorUtil.interruptibleShutdown(executorService)) {
      Thread.currentThread().interrupt();
    }

    executorService = null;
  }
}
