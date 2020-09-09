// Copyright 2020 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.authandtls.AuthAndTLSOptions;
import io.grpc.ClientInterceptor;
import io.grpc.ManagedChannel;
import java.io.IOException;
import java.util.List;

/** A factory interface for creating a {@link ManagedChannel}. */
public interface ChannelFactory {
  ManagedChannel newChannel(
      String target, String proxy, AuthAndTLSOptions options, List<ClientInterceptor> interceptors)
      throws IOException;
}
