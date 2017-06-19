// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.exec;

import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.devtools.build.lib.actions.ActionContext;
import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.rules.fileset.FilesetActionContext;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 * Context for Fileset manifest actions. It currently only provides a ThreadPoolExecutor.
 *
 * <p>Fileset is a legacy, google-internal mechanism to make parts of the source tree appear as a
 * tree in the output directory.
 */
@ExecutionStrategy(contextType = FilesetActionContext.class)
public final class FilesetActionContextImpl implements FilesetActionContext {
  // TODO(bazel-team): it would be nice if this weren't shipped in Bazel at all.

  /**
   * Factory class.
   */
  public static class Provider extends ActionContextProvider {
    private FilesetActionContextImpl impl;
    private final Reporter reporter;
    private final ThreadPoolExecutor filesetPool;

    public Provider(Reporter reporter, String workspaceName) {
      this.reporter = reporter;
      this.filesetPool = newFilesetPool(100);
      this.impl = new FilesetActionContextImpl(filesetPool, workspaceName);
    }

    private static ThreadPoolExecutor newFilesetPool(int threads) {
      ThreadPoolExecutor pool = new ThreadPoolExecutor(threads, threads, 3L, TimeUnit.SECONDS,
                                                       new LinkedBlockingQueue<Runnable>());
      // Do not consume threads when not in use.
      pool.allowCoreThreadTimeOut(true);
      pool.setThreadFactory(new ThreadFactoryBuilder().setNameFormat("Fileset worker %d").build());
      return pool;
    }

    @Override
    public Iterable<? extends ActionContext> getActionContexts() {
      return ImmutableList.of(impl);
    }

    @Override
    public void executionPhaseEnding() {
      BlazeExecutor.shutdownHelperPool(reporter, filesetPool, "Fileset");
    }
  }

  private final ThreadPoolExecutor filesetPool;
  private final String workspaceName;

  private FilesetActionContextImpl(ThreadPoolExecutor filesetPool, String workspaceName) {
    this.filesetPool = filesetPool;
    this.workspaceName = workspaceName;
  }

  @Override
  public ThreadPoolExecutor getFilesetPool() {
    return filesetPool;
  }

  @Override
  public String getWorkspaceName() {
    return workspaceName;
  }
}
