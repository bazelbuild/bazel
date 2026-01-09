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
import com.google.devtools.build.lib.runtime.RemoteRepoContentsCache;
import com.google.devtools.build.lib.runtime.RepositoryRemoteExecutor;
import com.google.devtools.build.lib.runtime.RepositoryRemoteHelpersFactory;
import javax.annotation.Nullable;

/** Factory for {@link RemoteRepositoryRemoteExecutor} and {@link RemoteRepoContentsCacheImpl}. */
class RepositoryRemoteHelpersFactoryImpl implements RepositoryRemoteHelpersFactory {

  private final CombinedCache cache;
  @Nullable private final RemoteExecutionClient remoteExecutor;
  private final String buildRequestId;
  private final String commandId;
  private final String workspaceName;

  private final String remoteInstanceName;
  private final boolean acceptCached;
  private final boolean uploadLocalResults;
  private final boolean verboseFailures;

  RepositoryRemoteHelpersFactoryImpl(
      CombinedCache cache,
      @Nullable RemoteExecutionClient remoteExecutor,
      String buildRequestId,
      String commandId,
      String workspaceName,
      String remoteInstanceName,
      boolean acceptCached,
      boolean uploadLocalResults,
      boolean verboseFailures) {
    this.cache = cache;
    this.remoteExecutor = remoteExecutor;
    this.buildRequestId = buildRequestId;
    this.commandId = commandId;
    this.workspaceName = workspaceName;
    this.remoteInstanceName = remoteInstanceName;
    this.acceptCached = acceptCached;
    this.uploadLocalResults = uploadLocalResults;
    this.verboseFailures = verboseFailures;
  }

  @Nullable
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
        workspaceName,
        remoteInstanceName,
        acceptCached);
  }

  @Nullable
  @Override
  public RemoteRepoContentsCache createRepoContentsCache() {
    return new RemoteRepoContentsCacheImpl(
        cache, buildRequestId, commandId, acceptCached, uploadLocalResults, verboseFailures);
  }
}
