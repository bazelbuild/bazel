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

import com.google.common.collect.ImmutableList;
import io.grpc.CallOptions;
import io.grpc.ClientCall;
import io.grpc.ManagedChannel;
import io.grpc.MethodDescriptor;
import io.netty.util.AbstractReferenceCounted;
import io.netty.util.ReferenceCounted;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * A wrapper around a {@link io.grpc.ManagedChannel} exposing a reference count and performing a
 * round-robin load balance across a list of channels. When instantiated the reference count is 1.
 * {@link ManagedChannel#shutdown()} will be called on the wrapped channel when the reference count
 * reaches 0.
 *
 * <p>See {@link ReferenceCounted} for more information about reference counting.
 */
public class ReferenceCountedChannelPool extends ReferenceCountedChannel {

  private final AtomicInteger indexTicker = new AtomicInteger();
  private final ImmutableList<ManagedChannel> channels;

  public ReferenceCountedChannelPool(ImmutableList<ManagedChannel> channels) {
    super(
        channels.get(0),
        new AbstractReferenceCounted() {
          @Override
          protected void deallocate() {
            for (ManagedChannel channel : channels) {
              channel.shutdown();
            }
          }

          @Override
          public ReferenceCounted touch(Object o) {
            return null;
          }
        });
    this.channels = channels;
  }

  @Override
  public boolean isShutdown() {
    for (ManagedChannel channel : channels) {
      if (!channel.isShutdown()) {
        return false;
      }
    }
    return true;
  }

  @Override
  public boolean isTerminated() {
    for (ManagedChannel channel : channels) {
      if (!channel.isTerminated()) {
        return false;
      }
    }
    return true;
  }

  @Override
  public boolean awaitTermination(long timeout, TimeUnit timeUnit) throws InterruptedException {
    long endTimeNanos = System.nanoTime() + timeUnit.toNanos(timeout);
    for (ManagedChannel channel : channels) {
      long awaitTimeNanos = endTimeNanos - System.nanoTime();
      if (awaitTimeNanos <= 0) {
        break;
      }
      channel.awaitTermination(awaitTimeNanos, TimeUnit.NANOSECONDS);
    }
    return isTerminated();
  }

  @Override
  public <RequestT, ResponseT> ClientCall<RequestT, ResponseT> newCall(
      MethodDescriptor<RequestT, ResponseT> methodDescriptor, CallOptions callOptions) {
    return getNextChannel().newCall(methodDescriptor, callOptions);
  }

  @Override
  public String authority() {
    // Assume all channels have the same authority.
    return channels.get(0).authority();
  }

  /**
   * Performs a simple round robin on the list of {@link ManagedChannel}s in the {@code channels}
   * list.
   *
   * @see <a href="https://github.com/grpc/grpc/issues/21386#issuecomment-564742173">Suggestion from
   *     gRPC team.</a>
   * @return A {@link ManagedChannel} that can be used for a single RPC call.
   */
  private ManagedChannel getNextChannel() {
    int index = getChannelIndex(channels.size(), indexTicker.getAndIncrement());
    return channels.get(index);
  }

  public static int getChannelIndex(int poolSize, int affinity) {
    int index = affinity % poolSize;
    return Math.abs(index);
  }
}
