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

import io.grpc.CallOptions;
import io.grpc.ClientCall;
import io.grpc.ManagedChannel;
import io.grpc.MethodDescriptor;
import io.netty.util.AbstractReferenceCounted;
import io.netty.util.ReferenceCounted;
import java.util.concurrent.TimeUnit;

/** A wrapper around a {@link io.grpc.ManagedChannel} exposing a reference count.
 * When instantiated the reference count is 1. {@link ManagedChannel#shutdown()} will be called
 * on the wrapped channel when the reference count reaches 0.
 *
 * See {@link ReferenceCounted} for more information about reference counting.
 */
class ReferenceCountedChannel extends ManagedChannel implements ReferenceCounted {

  private final ManagedChannel channel;
  private final AbstractReferenceCounted referenceCounted = new AbstractReferenceCounted() {
    @Override
    protected void deallocate() {
      channel.shutdown();
    }

    @Override
    public ReferenceCounted touch(Object o) {
      return this;
    }
  };

  public ReferenceCountedChannel(ManagedChannel channel) {
    this.channel = channel;
  }

  @Override
  public ManagedChannel shutdown() {
    throw new UnsupportedOperationException("Don't call shutdown() directly, but use release() "
        + "instead.");
  }

  @Override
  public boolean isShutdown() {
    return channel.isShutdown();
  }

  @Override
  public boolean isTerminated() {
    return channel.isTerminated();
  }

  @Override
  public ManagedChannel shutdownNow() {
    throw new UnsupportedOperationException("Don't call shutdownNow() directly, but use release() "
        + "instead.");
  }

  @Override
  public boolean awaitTermination(long l, TimeUnit timeUnit) throws InterruptedException {
    return channel.awaitTermination(l, timeUnit);
  }

  @Override
  public <RequestT, ResponseT> ClientCall<RequestT, ResponseT> newCall(
      MethodDescriptor<RequestT, ResponseT> methodDescriptor, CallOptions callOptions) {
    return channel.<RequestT, ResponseT>newCall(methodDescriptor, callOptions);
  }

  @Override
  public String authority() {
    return channel.authority();
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