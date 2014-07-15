// Copyright 2014 Google Inc. All rights reserved.
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
import com.google.devtools.build.lib.actions.ActionContextProvider;
import com.google.devtools.build.lib.actions.ActionGraph;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactMTimeCache;
import com.google.devtools.build.lib.actions.BlazeExecutor;
import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.actions.Executor.ActionContext;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.view.fileset.FilesetActionContext;
import com.google.devtools.build.lib.view.fileset.FilesetManifestAction;

import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 * Context for {@link FilesetManifestAction}. It currently only provides a ThreadPoolExecutor.
 */
@ExecutionStrategy(contextType = FilesetActionContext.class)
public final class FilesetActionContextImpl implements FilesetActionContext {
  /**
   * Factory class.
   */
  public static class Provider implements ActionContextProvider {
    private FilesetActionContextImpl impl;
    private final Reporter reporter;
    private final ThreadPoolExecutor filesetPool;

    public Provider(Reporter reporter) {
      this.reporter = reporter;
      this.filesetPool = newFilesetPool(100);
      this.impl = new FilesetActionContextImpl(filesetPool);
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
    public Iterable<ActionContext> getActionContexts() {
      return ImmutableList.<ActionContext>of(impl);
    }

    @Override
    public void executorCreated(Iterable<ActionContext> usedStrategies) {}

    @Override
    public void executionPhaseStarting(
        ActionInputFileCache actionInputFileCache,
        ArtifactMTimeCache artifactMTimeCache,
        ActionGraph actionGraph,
        Iterable<Artifact> topLevelArtifacts) {}

    @Override
    public void executionPhaseEnding() {
      BlazeExecutor.shutdownHelperPool(reporter, filesetPool, "Fileset");
    }
  }

  private final ThreadPoolExecutor filesetPool;

  private FilesetActionContextImpl(ThreadPoolExecutor filesetPool) {
    this.filesetPool = filesetPool;
  }

  @Override
  public ThreadPoolExecutor getFilesetPool() {
    return filesetPool;
  }
}
