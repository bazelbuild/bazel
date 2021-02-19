// Copyright 2021 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote.grpc;

import io.grpc.CallOptions;
import io.grpc.ClientCall;
import io.grpc.MethodDescriptor;
import java.io.Closeable;
import java.io.IOException;

/**
 * A single connection to a server. RPCs are executed within the context of a connection. A {@link
 * Connection} object can consist of any number of transport connections.
 *
 * <p>Connections must be closed to ensure proper resource disposal.
 */
public interface Connection extends Closeable {

  /** Creates a new {@link ClientCall} for issuing RPC. */
  <ReqT, RespT> ClientCall<ReqT, RespT> call(
      MethodDescriptor<ReqT, RespT> method, CallOptions options);

  /** Releases any resources held by the {@link Connection}. */
  @Override
  void close() throws IOException;
}
