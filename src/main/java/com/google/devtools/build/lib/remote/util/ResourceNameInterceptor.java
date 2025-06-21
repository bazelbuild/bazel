// Copyright 2024 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.remote.util;

import com.google.bytestream.ByteStreamGrpc;
import io.grpc.CallOptions;
import io.grpc.Channel;
import io.grpc.ClientCall;
import io.grpc.ClientInterceptor;
import io.grpc.ForwardingClientCall.SimpleForwardingClientCall;
import io.grpc.Metadata;
import io.grpc.MethodDescriptor;
import io.grpc.ServerCall.Listener;

/** A gRPC client interceptor that adds a "resource-name" header to ByteStream Read/Write calls. */
public class ResourceNameInterceptor implements ClientInterceptor {
  public static final Metadata.Key<String> RESOURCE_NAME_KEY =
      Metadata.Key.of(
          "build.bazel.remote.execution.v2.resource-name", Metadata.ASCII_STRING_MARSHALLER);

  private final String resourceName;

  public ResourceNameInterceptor(String resourceName) {
    this.resourceName = resourceName;
  }

  @Override
  public <ReqT, RespT> ClientCall<ReqT, RespT> interceptCall(
      MethodDescriptor<ReqT, RespT> method, CallOptions callOptions, Channel next) {
    return new SimpleForwardingClientCall<ReqT, RespT>(next.newCall(method, callOptions)) {
      @Override
      public void start(Listener<RespT> responseListener, Metadata headers) {
        if (resourceName != null && !resourceName.isEmpty()) {
          headers.put(RESOURCE_NAME_KEY, resourceName);
        }
        super.start(responseListener, headers);
      }
    };
  }
}
