package com.google.devtools.build.lib.remote;

import io.grpc.CallOptions;
import io.grpc.ClientCall;
import io.grpc.ManagedChannel;
import io.grpc.MethodDescriptor;
import io.netty.util.AbstractReferenceCounted;
import io.netty.util.ReferenceCounted;
import java.util.concurrent.TimeUnit;

class ReferenceCountedChannel extends ManagedChannel {

  private final ManagedChannel channel;
  private final AbstractReferenceCounted referenceCounted = new AbstractReferenceCounted() {
    @Override
    protected void deallocate() {
      shutdown();
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
    return channel.shutdown();
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
    return channel.shutdownNow();
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

  public ReferenceCountedChannel retain() {
    referenceCounted.retain();
    return this;
  }

  public boolean release() {
    return referenceCounted.release();
  }
}