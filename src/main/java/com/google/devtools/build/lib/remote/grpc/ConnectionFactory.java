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

import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import io.reactivex.rxjava3.core.Single;

/**
 * A {@link ConnectionFactory} represents a resource factory for connection creation. It may create
 * connections by itself, wrap a {@link ConnectionFactory}, or apply connection pooling on top of a
 * {@link ConnectionFactory}.
 *
 * <p>A {@link ConnectionFactory} uses deferred initialization and should initiate connection
 * resource allocation after subscription.
 *
 * <p>Connection creation must be cancellable. Canceling connection creation must release (“close”)
 * the connection and all associated resources.
 *
 * <p>Implementations must be thread-safe.
 */
@ThreadSafe
public interface ConnectionFactory {
  /** Creates a new {@link Connection}. */
  Single<? extends Connection> create();
}
