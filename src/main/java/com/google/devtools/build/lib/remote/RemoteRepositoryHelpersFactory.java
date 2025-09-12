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
package com.google.devtools.build.lib.remote;

import com.google.devtools.build.lib.remote.common.RemoteExecutionClient;
import com.google.devtools.build.lib.runtime.RepoContentsCache;
import com.google.devtools.build.lib.runtime.RepositoryHelpersFactory;
import com.google.devtools.build.lib.runtime.RepositoryRemoteExecutor;
import javax.annotation.Nullable;

/** Factory for {@link RemoteRepositoryRemoteExecutor}. */
class RemoteRepositoryHelpersFactory implements RepositoryHelpersFactory {

  private final CombinedCache cache;
  @Nullable private final RemoteExecutionClient remoteExecutor;
  private final String buildRequestId;
  private final String commandId;

  private final String remoteInstanceName;
  private final boolean acceptCached;
  private final boolean uploadLocalResults;

  RemoteRepositoryHelpersFactory(
      CombinedCache cache,
      @Nullable RemoteExecutionClient remoteExecutor,
      String buildRequestId,
      String commandId,
      String remoteInstanceName,
      boolean acceptCached,
      boolean uploadLocalResults) {
    this.cache = cache;
    this.remoteExecutor = remoteExecutor;
    this.buildRequestId = buildRequestId;
    this.commandId = commandId;
    this.remoteInstanceName = remoteInstanceName;
    this.acceptCached = acceptCached;
    this.uploadLocalResults = uploadLocalResults;
  }

  @Override
  public RepositoryRemoteExecutor createExecutor() {
    if (remoteExecutor == null) {
      return null;
    }
    return new RemoteRepositoryRemoteExecutor(
        (RemoteExecutionCache) cache,
        remoteExecutor,
        cache.digestUtil,
        buildRequestId,
        commandId,
        remoteInstanceName,
        acceptCached);
  }

  @Nullable
  @Override
  public RepoContentsCache createRepoContentsCache() {
    return new RemoteRepoContentsCache(
        cache, buildRequestId, commandId, acceptCached, uploadLocalResults);
  }
}
