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

package com.google.devtools.build.lib.remote.logging;

import com.google.devtools.build.lib.remote.logging.RemoteExecutionLog.RpcCallDetails;

/**
 * An interface for building {@link RpcCallDetails}s specialized for a specific gRPC call.
 *
 * @param <ReqT> request type of the gRPC call
 * @param <RespT> response type of the gRPC call
 */
public interface LoggingHandler<ReqT, RespT> {

  /**
   * Handle logging for an issued message.
   *
   * @param message the issued request message
   */
  void handleReq(ReqT message);

  /**
   * Handle logging for a received response.
   *
   * @param message the received response message
   */
  void handleResp(RespT message);

  /**
   * Returns a {@link RpcCallDetails} based on the requests and responses handled by this handler.
   */
  RpcCallDetails getDetails();
}
