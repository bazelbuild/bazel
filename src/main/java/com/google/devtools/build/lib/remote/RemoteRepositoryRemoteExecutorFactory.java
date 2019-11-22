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

import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.runtime.RepositoryRemoteExecutor;
import com.google.devtools.build.lib.runtime.RepositoryRemoteExecutorFactory;
import io.grpc.Context;

/** Factory for {@link RemoteRepositoryRemoteExecutor}. */
class RemoteRepositoryRemoteExecutorFactory implements RepositoryRemoteExecutorFactory {

  private final RemoteExecutionCache remoteExecutionCache;
  private final GrpcRemoteExecutor remoteExecutor;
  private final DigestUtil digestUtil;
  private final Context requestCtx;

  private final String remoteInstanceName;
  private final boolean acceptCached;

  RemoteRepositoryRemoteExecutorFactory(
      RemoteExecutionCache remoteExecutionCache,
      GrpcRemoteExecutor remoteExecutor,
      DigestUtil digestUtil,
      Context requestCtx,
      String remoteInstanceName,
      boolean acceptCached) {
    this.remoteExecutionCache = remoteExecutionCache;
    this.remoteExecutor = remoteExecutor;
    this.digestUtil = digestUtil;
    this.requestCtx = requestCtx;
    this.remoteInstanceName = remoteInstanceName;
    this.acceptCached = acceptCached;
  }

  @Override
  public RepositoryRemoteExecutor create() {
    return new RemoteRepositoryRemoteExecutor(
        remoteExecutionCache,
        remoteExecutor,
        digestUtil,
        requestCtx,
        remoteInstanceName,
        acceptCached);
  }
}
