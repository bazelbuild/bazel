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
package com.google.devtools.build.lib.remote;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.base.Throwables;
import com.google.devtools.build.lib.remote.grpc.ChannelConnectionFactory;
import com.google.devtools.build.lib.remote.grpc.ChannelConnectionFactory.ChannelConnection;
import com.google.devtools.build.lib.remote.grpc.DynamicConnectionPool;
import com.google.devtools.build.lib.remote.grpc.SharedConnectionFactory.SharedConnection;
import io.grpc.CallOptions;
import io.grpc.Channel;
import io.grpc.ClientCall;
import io.grpc.ForwardingClientCall;
import io.grpc.ForwardingClientCallListener;
import io.grpc.Metadata;
import io.grpc.MethodDescriptor;
import io.grpc.Status;
import io.netty.util.AbstractReferenceCounted;
import io.netty.util.ReferenceCounted;
import java.io.IOException;
import java.util.concurrent.atomic.AtomicReference;
import javax.annotation.Nullable;

/**
 * A wrapper around a {@link DynamicConnectionPool} exposing {@link Channel} and a reference count.
 * When instantiated the reference count is 1. {@link DynamicConnectionPool#close()} will be called
 * on the wrapped channel when the reference count reaches 0.
 *
 * <p>See {@link ReferenceCounted} for more information about reference counting.
 */
public class ReferenceCountedChannel extends Channel implements ReferenceCounted {
  private final DynamicConnectionPool dynamicConnectionPool;
  private final AbstractReferenceCounted referenceCounted =
      new AbstractReferenceCounted() {
        @Override
        protected void deallocate() {
          try {
            dynamicConnectionPool.close();
          } catch (IOException e) {
            throw new AssertionError(e.getMessage(), e);
          }
        }

        @Override
        public ReferenceCounted touch(Object o) {
          return this;
        }
      };
  private final AtomicReference<String> authorityRef = new AtomicReference<>();

  public ReferenceCountedChannel(ChannelConnectionFactory connectionFactory) {
    this(connectionFactory, /*maxConnections=*/ 0);
  }

  public ReferenceCountedChannel(ChannelConnectionFactory connectionFactory, int maxConnections) {
    this.dynamicConnectionPool =
        new DynamicConnectionPool(
            connectionFactory, connectionFactory.maxConcurrency(), maxConnections);
  }

  public boolean isShutdown() {
    return dynamicConnectionPool.isClosed();
  }

  /** A {@link ClientCall} which call {@link SharedConnection#close()} after the RPC is closed. */
  static class ConnectionCleanupCall<ReqT, RespT>
      extends ForwardingClientCall.SimpleForwardingClientCall<ReqT, RespT> {
    private final SharedConnection connection;

    protected ConnectionCleanupCall(ClientCall<ReqT, RespT> delegate, SharedConnection connection) {
      super(delegate);
      this.connection = connection;
    }

    @Override
    public void start(Listener<RespT> responseListener, Metadata headers) {
      super.start(
          new ForwardingClientCallListener.SimpleForwardingClientCallListener<RespT>(
              responseListener) {
            @Override
            public void onClose(Status status, Metadata trailers) {
              try {
                connection.close();
              } catch (IOException e) {
                throw new AssertionError(e.getMessage(), e);
              } finally {
                super.onClose(status, trailers);
              }
            }
          },
          headers);
    }
  }

  private static class CloseOnStartClientCall<ReqT, RespT> extends ClientCall<ReqT, RespT> {
    private final Status status;

    CloseOnStartClientCall(Status status) {
      this.status = status;
    }

    @Override
    public void start(Listener<RespT> responseListener, Metadata headers) {
      responseListener.onClose(status, new Metadata());
    }

    @Override
    public void request(int numMessages) {}

    @Override
    public void cancel(@Nullable String message, @Nullable Throwable cause) {}

    @Override
    public void halfClose() {}

    @Override
    public void sendMessage(ReqT message) {}
  }

  private SharedConnection acquireSharedConnection() throws IOException, InterruptedException {
    try {
      SharedConnection sharedConnection = dynamicConnectionPool.create().blockingGet();
      ChannelConnection connection = (ChannelConnection) sharedConnection.getUnderlyingConnection();
      authorityRef.compareAndSet(null, connection.getChannel().authority());
      return sharedConnection;
    } catch (RuntimeException e) {
      Throwables.throwIfInstanceOf(e.getCause(), IOException.class);
      Throwables.throwIfInstanceOf(e.getCause(), InterruptedException.class);
      throw e;
    }
  }

  @Override
  public <RequestT, ResponseT> ClientCall<RequestT, ResponseT> newCall(
      MethodDescriptor<RequestT, ResponseT> methodDescriptor, CallOptions callOptions) {
    try {
      SharedConnection sharedConnection = acquireSharedConnection();
      return new ConnectionCleanupCall<>(
          sharedConnection.call(methodDescriptor, callOptions), sharedConnection);
    } catch (IOException e) {
      return new CloseOnStartClientCall<>(Status.UNKNOWN.withCause(e));
    } catch (InterruptedException e) {
      return new CloseOnStartClientCall<>(Status.CANCELLED.withCause(e));
    }
  }

  @Override
  public String authority() {
    String authority = authorityRef.get();
    checkNotNull(authority, "create a connection first to get the authority");
    return authority;
  }

  @Override
  public int refCnt() {
    return referenceCounted.refCnt();
  }

  @Override
  public ReferenceCountedChannel retain() {
    referenceCounted.retain();
    return this;
  }

  @Override
  public ReferenceCountedChannel retain(int increment) {
    referenceCounted.retain(increment);
    return this;
  }

  @Override
  public ReferenceCounted touch() {
    referenceCounted.touch();
    return this;
  }

  @Override
  public ReferenceCounted touch(Object hint) {
    referenceCounted.touch(hint);
    return this;
  }

  @Override
  public boolean release() {
    return referenceCounted.release();
  }

  @Override
  public boolean release(int decrement) {
    return referenceCounted.release(decrement);
  }
}
