// Copyright 2017 The Bazel Authors. All rights reserved.
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


import build.bazel.remote.execution.v2.ActionCacheGrpc.ActionCacheImplBase;
import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.GetActionResultRequest;
import build.bazel.remote.execution.v2.RequestMetadata;
import build.bazel.remote.execution.v2.UpdateActionResultRequest;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.remote.common.NetworkTime;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContextImpl;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient.ActionKey;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import io.grpc.stub.StreamObserver;

/** A basic implementation of an {@link ActionCacheImplBase} service. */
final class ActionCacheServer extends ActionCacheImplBase {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private final OnDiskBlobStoreCache cache;
  private final DigestUtil digestUtil;

  public ActionCacheServer(OnDiskBlobStoreCache cache, DigestUtil digestUtil) {
    this.cache = cache;
    this.digestUtil = digestUtil;
  }

  @Override
  public void getActionResult(
      GetActionResultRequest request, StreamObserver<ActionResult> responseObserver) {
    try {
      RequestMetadata requestMetadata = TracingMetadataUtils.fromCurrentContext();
      RemoteActionExecutionContext context =
          new RemoteActionExecutionContextImpl(requestMetadata, new NetworkTime());

      ActionKey actionKey = digestUtil.asActionKey(request.getActionDigest());
      ActionResult result =
          cache.downloadActionResult(context, actionKey, /* inlineOutErr= */ false);

      if (result == null) {
        responseObserver.onError(StatusUtils.notFoundError(request.getActionDigest()));
        return;
      }

      responseObserver.onNext(result);
      responseObserver.onCompleted();
    } catch (Exception e) {
      logger.atWarning().withCause(e).log("getActionResult request failed");
      responseObserver.onError(StatusUtils.internalError(e));
    }
  }

  @Override
  public void updateActionResult(
      UpdateActionResultRequest request, StreamObserver<ActionResult> responseObserver) {
    try {
      RequestMetadata requestMetadata = TracingMetadataUtils.fromCurrentContext();
      RemoteActionExecutionContext context =
          new RemoteActionExecutionContextImpl(requestMetadata, new NetworkTime());

      ActionKey actionKey = digestUtil.asActionKey(request.getActionDigest());
      cache.uploadActionResult(context, actionKey, request.getActionResult());
      responseObserver.onNext(request.getActionResult());
      responseObserver.onCompleted();
    } catch (Exception e) {
      logger.atWarning().withCause(e).log("updateActionResult request failed");
      responseObserver.onError(StatusUtils.internalError(e));
    }
  }
}
