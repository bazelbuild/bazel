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

import io.reactivex.rxjava3.core.Single;
import java.io.Closeable;
import java.io.IOException;

/**
 * A {@link ConnectionFactory} that apply connection pooling. Connections created by {@link
 * ConnectionPool} will not be closed by {@link Connection#close()}, use {@link
 * ConnectionPool#close()} instead to close all the connections in the pool.
 *
 * <p>Connections must be closed with {@link Connection#close()} in order to be reused later.
 */
public interface ConnectionPool extends ConnectionFactory, Closeable {
  /**
   * Reuses a {@link Connection} in the pool and will potentially create a new connection depends on
   * implementation.
   */
  @Override
  Single<? extends Connection> create();

  /** Closes the connection pool and closes all the underlying connections */
  @Override
  void close() throws IOException;
}
