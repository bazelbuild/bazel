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
 * <p>See {@link ReferenceCounted} for more information about reference counting.
 */
public class ReferenceCountedChannelPool extends ReferenceCountedChannel {

  private final AtomicInteger indexTicker = new AtomicInteger();
  private final ImmutableList<ManagedChannel> channels;

  public ReferenceCountedChannelPool(ImmutableList<ManagedChannel> channels) {
    super(channels.get(0), new AbstractReferenceCounted() {
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
    return true;  }

  @Override
  public boolean awaitTermination(long l, TimeUnit timeUnit) throws InterruptedException {
    long endTimeNanos = System.nanoTime() + timeUnit.toNanos(l);
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
   * @return A {@link ManagedChannel} that can be used for a single RPC call.
   */
  private ManagedChannel getNextChannel() {
    return getChannel(indexTicker.getAndIncrement());
  }

  private ManagedChannel getChannel(int affinity) {
    int index = affinity % channels.size();
    index = Math.abs(index);
    // If index is the most negative int, abs(index) is still negative.
    if (index < 0) {
      index = 0;
    }
    return channels.get(index);
  }
}
