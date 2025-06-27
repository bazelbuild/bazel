// Copyright 2025 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe.serialization.analysis;

import com.google.common.util.concurrent.ListenableFuture;
import com.google.protobuf.ByteString;

/** Interface to the remote analysis cache. */
public interface RemoteAnalysisCacheClient {
  /** Usage statistics. */
  record Stats(long bytesSent, long bytesReceived, long requestsSent, long batches) {}

  /** Looks up an entry in the remote analysis cache based on a serialized key. */
  ListenableFuture<ByteString> lookup(ByteString key);

  /** Returns the usage statistics. */
  Stats getStats();
}
