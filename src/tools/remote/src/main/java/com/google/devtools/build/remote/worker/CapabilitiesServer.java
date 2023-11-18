// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.remote.worker;

import build.bazel.remote.execution.v2.ActionCacheUpdateCapabilities;
import build.bazel.remote.execution.v2.CacheCapabilities;
import build.bazel.remote.execution.v2.CapabilitiesGrpc.CapabilitiesImplBase;
import build.bazel.remote.execution.v2.DigestFunction;
import build.bazel.remote.execution.v2.ExecutionCapabilities;
import build.bazel.remote.execution.v2.GetCapabilitiesRequest;
import build.bazel.remote.execution.v2.ServerCapabilities;
import build.bazel.remote.execution.v2.SymlinkAbsolutePathStrategy;
import com.google.devtools.build.lib.remote.ApiVersion;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import io.grpc.stub.StreamObserver;

/** A basic implementation of a Capabilities service. */
final class CapabilitiesServer extends CapabilitiesImplBase {
  private final DigestUtil digestUtil;
  private final boolean execEnabled;

  public CapabilitiesServer(DigestUtil digestUtil, boolean execEnabled) {
    this.digestUtil = digestUtil;
    this.execEnabled = execEnabled;
  }

  @Override
  public void getCapabilities(
      GetCapabilitiesRequest request, StreamObserver<ServerCapabilities> responseObserver) {
    DigestFunction.Value df = digestUtil.getDigestFunction();
    ServerCapabilities.Builder response =
        ServerCapabilities.newBuilder()
            .setLowApiVersion(ApiVersion.low.toSemVer())
            .setHighApiVersion(ApiVersion.high.toSemVer())
            .setCacheCapabilities(
                CacheCapabilities.newBuilder()
                    .addDigestFunctions(df)
                    .setSymlinkAbsolutePathStrategy(SymlinkAbsolutePathStrategy.Value.DISALLOWED)
                    .setActionCacheUpdateCapabilities(
                        ActionCacheUpdateCapabilities.newBuilder().setUpdateEnabled(true).build())
                    .setMaxBatchTotalSizeBytes(CasServer.MAX_BATCH_SIZE_BYTES)
                    .build());
    if (execEnabled) {
      response.setExecutionCapabilities(
          ExecutionCapabilities.newBuilder().setDigestFunction(df).setExecEnabled(true).build());
    }
    responseObserver.onNext(response.build());
    responseObserver.onCompleted();
  }
}
