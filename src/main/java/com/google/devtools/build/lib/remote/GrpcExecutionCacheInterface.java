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
package com.google.devtools.build.lib.remote;

import com.google.devtools.build.lib.remote.ExecutionCacheServiceGrpc.ExecutionCacheServiceBlockingStub;
import com.google.devtools.build.lib.remote.RemoteProtocol.ExecutionCacheReply;
import com.google.devtools.build.lib.remote.RemoteProtocol.ExecutionCacheRequest;
import com.google.devtools.build.lib.remote.RemoteProtocol.ExecutionCacheSetReply;
import com.google.devtools.build.lib.remote.RemoteProtocol.ExecutionCacheSetRequest;

/**
 * An abstraction layer between the remote execution client and gRPC to support unit testing. This
 * interface covers the execution cache RPC methods, see {@link ExecutionCacheServiceBlockingStub}.
 */
public interface GrpcExecutionCacheInterface {
  ExecutionCacheReply getCachedResult(ExecutionCacheRequest request);
  ExecutionCacheSetReply setCachedResult(ExecutionCacheSetRequest request);
}
