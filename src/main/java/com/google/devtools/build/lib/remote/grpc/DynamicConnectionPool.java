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

import com.google.devtools.build.lib.remote.grpc.SharedConnectionFactory.SharedConnection;
import io.reactivex.rxjava3.core.Single;
import java.io.IOException;
import java.util.ArrayList;
import java.util.concurrent.atomic.AtomicBoolean;
import javax.annotation.concurrent.GuardedBy;

/**
 * A {@link ConnectionPool} that creates new connection with given {@link ConnectionFactory} on
 * demand and applies rate limiting w.r.t {@code maxConcurrencyPerConnection} for one underlying
 * connection. It also uses Round-Robin algorithm to load balancing between underlying connections.
 *
 * <p>Connections must be closed with {@link Connection#close()} in order to be reused later.
 */
public class DynamicConnectionPool implements ConnectionPool {
  private final ConnectionFactory connectionFactory;
  private final int maxConcurrencyPerConnection;
  private final int maxConnections;
  private final AtomicBoolean closed = new AtomicBoolean(false);

  @GuardedBy("this")
  private final ArrayList<SharedConnectionFactory> factories;

  @GuardedBy("this")
  private int indexTicker = 0;

  public DynamicConnectionPool(
      ConnectionFactory connectionFactory, int maxConcurrencyPerConnection) {
    this(connectionFactory, maxConcurrencyPerConnection, /*maxConnections=*/ 0);
  }

  public DynamicConnectionPool(
      ConnectionFactory connectionFactory, int maxConcurrencyPerConnection, int maxConnections) {
    this.connectionFactory = connectionFactory;
    this.maxConcurrencyPerConnection = maxConcurrencyPerConnection;
    this.maxConnections = maxConnections;
    this.factories = new ArrayList<>();
  }

  public boolean isClosed() {
    return closed.get();
  }

  @Override
  public void close() throws IOException {
    if (closed.compareAndSet(false, true)) {
      synchronized (this) {
        for (SharedConnectionFactory factory : factories) {
          factory.close();
        }
        factories.clear();
      }
    }
  }

  @GuardedBy("this")
  private SharedConnectionFactory nextFactory() {
    int index = Math.abs(indexTicker % factories.size());
    indexTicker += 1;
    return factories.get(index);
  }

  /**
   * Performs a simple round robin on the list of {@link SharedConnectionFactory}.
   *
   * <p>This will try to find a factory that has available connections at this moment. If no factory
   * has available connections, and the number of factories is less than {@link #maxConnections}, it
   * will create a new {@link SharedConnectionFactory}.
   */
  private SharedConnectionFactory nextAvailableFactory() {
    if (closed.get()) {
      throw new IllegalStateException("closed");
    }

    synchronized (this) {
      for (int times = 0; times < factories.size(); ++times) {
        SharedConnectionFactory factory = nextFactory();
        if (factory.numAvailableConnections() > 0) {
          return factory;
        }
      }

      if (maxConnections <= 0 || factories.size() < maxConnections) {
        SharedConnectionFactory factory =
            new SharedConnectionFactory(connectionFactory, maxConcurrencyPerConnection);
        factories.add(factory);
        return factory;
      } else {
        return nextFactory();
      }
    }
  }

  @Override
  public Single<SharedConnection> create() {
    return Single.defer(
        () -> {
          SharedConnectionFactory factory = nextAvailableFactory();
          return factory.create();
        });
  }
}
