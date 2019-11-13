package com.google.devtools.build.lib.remote;

import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.runtime.RepositoryRemoteExecutor;
import com.google.devtools.build.lib.runtime.RepositoryRemoteExecutorFactory;
import io.grpc.Context;

class RemoteRepositoryRemoteExecutorFactory implements RepositoryRemoteExecutorFactory {

  private final RemoteExecutionCache remoteExecutionCache;
  private final GrpcRemoteExecutor remoteExecutor;
  private final DigestUtil digestUtil;
  private final Context requestCtx;

  private final String remoteInstanceName;
  private final boolean acceptCached;

  RemoteRepositoryRemoteExecutorFactory(RemoteExecutionCache remoteExecutionCache,
      GrpcRemoteExecutor remoteExecutor, DigestUtil digestUtil, Context requestCtx,
      String remoteInstanceName, boolean acceptCached) {
    this.remoteExecutionCache = remoteExecutionCache;
    this.remoteExecutor = remoteExecutor;
    this.digestUtil = digestUtil;
    this.requestCtx = requestCtx;
    this.remoteInstanceName = remoteInstanceName;
    this.acceptCached = acceptCached;
  }

  @Override
  public RepositoryRemoteExecutor create() {
    return new RemoteRepositoryRemoteExecutor(remoteExecutionCache, remoteExecutor,
        digestUtil, requestCtx, remoteInstanceName, acceptCached);
  }
}
