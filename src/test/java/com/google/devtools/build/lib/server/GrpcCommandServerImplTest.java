// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.server;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.server.CommandProtos.RunRequest;
import com.google.devtools.build.lib.server.CommandProtos.RunResponse;
import com.google.devtools.build.lib.server.CommandServerGrpc.CommandServerStub;
import com.google.devtools.build.lib.server.GrpcCommandServerImpl.BlockingStreamObserver;
import com.google.protobuf.ByteString;
import io.grpc.ManagedChannel;
import io.grpc.Server;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;
import io.grpc.stub.ServerCallStreamObserver;
import io.grpc.stub.StreamObserver;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link GrpcCommandServerImpl}. */
@RunWith(JUnit4.class)
public final class GrpcCommandServerImplTest {
  @Test
  public void testBlockingStreamObserver() throws Exception {
    // This test attempts to verify that BlockingStreamObserver successfully blocks after some
    // number of onNext calls (however long it takes to fill up gRPCs internal buffers). In order to
    // trigger this behavior, we intentionally block the client after a few successful calls, then
    // wait a bit, and then check that the server has stopped prematurely. Unfortunately, we cannot
    // deterministically verify that the onNext call is blocking. A faulty implementation of
    // BlockingStreamObserver could pass this test if the sleep is too short. However, a correct
    // implementation should never fail this test. This test could start failing if gRPCs internal
    // buffer size is increased. If it fails after an upgrade of gRPC, you might want to check that.
    CountDownLatch serverDone = new CountDownLatch(1);
    CountDownLatch clientBlocks = new CountDownLatch(1);
    CountDownLatch clientUnblocks = new CountDownLatch(1);
    CountDownLatch clientDone = new CountDownLatch(1);
    AtomicInteger sentCount = new AtomicInteger();
    AtomicInteger receiveCount = new AtomicInteger();
    CommandServerGrpc.CommandServerImplBase serverImpl =
        new CommandServerGrpc.CommandServerImplBase() {
          @Override
          public void run(RunRequest request, StreamObserver<RunResponse> observer) {
            ServerCallStreamObserver<RunResponse> serverCallStreamObserver =
                (ServerCallStreamObserver<RunResponse>) observer;
            GrpcCommandServerImpl.BlockingStreamObserver<RunResponse> blockingStreamObserver =
                new GrpcCommandServerImpl.BlockingStreamObserver<>(serverCallStreamObserver);
            Thread t =
                new Thread(
                    () -> {
                      RunResponse response =
                          RunResponse.newBuilder()
                              .setStandardOutput(ByteString.copyFrom(new byte[1024]))
                              .build();
                      for (int i = 0; i < 100; i++) {
                        blockingStreamObserver.onNext(response);
                        sentCount.incrementAndGet();
                      }
                      blockingStreamObserver.onCompleted();
                      serverDone.countDown();
                    });
            t.start();
          }
        };

    String uniqueName = InProcessServerBuilder.generateName();
    // Do not use .directExecutor here, as it makes both client and server run in the same thread.
    Server server =
        InProcessServerBuilder.forName(uniqueName)
            .addService(serverImpl)
            .executor(Executors.newFixedThreadPool(4))
            .build()
            .start();
    ManagedChannel channel =
        InProcessChannelBuilder.forName(uniqueName)
            .executor(Executors.newFixedThreadPool(4))
            .build();

    CommandServerStub stub = CommandServerGrpc.newStub(channel);
    stub.run(
        RunRequest.getDefaultInstance(),
        new StreamObserver<RunResponse>() {
          @Override
          public void onNext(RunResponse value) {
            if (sentCount.get() >= 3) {
              clientBlocks.countDown();
              try {
                clientUnblocks.await();
              } catch (InterruptedException e) {
                throw new IllegalStateException(e);
              }
            }
            receiveCount.incrementAndGet();
          }

          @Override
          public void onError(Throwable t) {
            throw new IllegalStateException(t);
          }

          @Override
          public void onCompleted() {
            clientDone.countDown();
          }
        });
    clientBlocks.await();
    // Wait a bit for the server to (hopefully) block. If the server does not block, then this may
    // be flaky.
    Thread.sleep(10);
    assertThat(sentCount.get()).isLessThan(5);
    clientUnblocks.countDown();
    serverDone.await();
    clientDone.await();
    server.shutdown();
    server.awaitTermination();
  }

  @Test
  public void testBlockingStreamObserverClientCancel() throws Exception {
    // This test attempts to verify that FlowControl unblocks if the client prematurely closes the
    // connection. In that case, FlowControl should observe the onCancel event and interrupt the
    // calling thread. I have observed this test failing with an intentionally introduced bug in
    // FlowControl.
    CountDownLatch serverDone = new CountDownLatch(1);
    CountDownLatch clientDone = new CountDownLatch(1);
    AtomicInteger sentCount = new AtomicInteger();
    AtomicInteger receiveCount = new AtomicInteger();
    CommandServerGrpc.CommandServerImplBase serverImpl =
        new CommandServerGrpc.CommandServerImplBase() {
          @Override
          public void run(RunRequest request, StreamObserver<RunResponse> observer) {
            ServerCallStreamObserver<RunResponse> serverCallStreamObserver =
                (ServerCallStreamObserver<RunResponse>) observer;
            GrpcCommandServerImpl.BlockingStreamObserver<RunResponse> blockingStreamObserver =
                new GrpcCommandServerImpl.BlockingStreamObserver<>(serverCallStreamObserver);
            Thread t =
                new Thread(
                    () -> {
                      RunResponse response =
                          RunResponse.newBuilder()
                              .setStandardOutput(ByteString.copyFrom(new byte[1024]))
                              .build();
                      for (int i = 0; i < 100; i++) {
                        blockingStreamObserver.onNext(response);
                        sentCount.incrementAndGet();
                      }
                      // FlowControl should have interrupted the current thread after learning of
                      // the server
                      // cancel.
                      assertThat(Thread.currentThread().isInterrupted()).isTrue();
                      blockingStreamObserver.onCompleted();
                      serverDone.countDown();
                    });
            t.start();
          }
        };

    String uniqueName = InProcessServerBuilder.generateName();
    // Do not use .directExecutor here, as it makes both client and server run in the same thread.
    Server server =
        InProcessServerBuilder.forName(uniqueName)
            .addService(serverImpl)
            .executor(Executors.newFixedThreadPool(4))
            .build()
            .start();
    ManagedChannel channel =
        InProcessChannelBuilder.forName(uniqueName)
            .executor(Executors.newFixedThreadPool(4))
            .build();

    CommandServerStub stub = CommandServerGrpc.newStub(channel);
    stub.run(
        RunRequest.getDefaultInstance(),
        new StreamObserver<RunResponse>() {
          @Override
          public void onNext(RunResponse value) {
            if (receiveCount.get() > 3) {
              channel.shutdownNow();
            }
            receiveCount.incrementAndGet();
          }

          @Override
          public void onError(Throwable t) {
            clientDone.countDown();
          }

          @Override
          public void onCompleted() {
            clientDone.countDown();
          }
        });
    serverDone.await();
    clientDone.await();
    server.shutdown();
    server.awaitTermination();
  }

  @Test
  public void testBlockingStreamObserverInterrupt() throws Exception {
    // This test attempts to verify that BlockingStreamObserver does not hang if the current thread
    // is interrupted. The initial implementation of BlockingStreamObserver (which was never
    // submitted) would go into an infinite loop holding the lock on BlockingStreamObserver. This
    // would prevent any other thread from obtaining the lock on BlockingStreamObserver, and hang
    // the entire process. I have confirmed that this test fails with the original faulty
    // implementation of BlockingStreamObserver.
    CountDownLatch serverDone = new CountDownLatch(1);
    CountDownLatch clientDone = new CountDownLatch(1);
    AtomicInteger sentCount = new AtomicInteger();
    AtomicInteger receiveCount = new AtomicInteger();
    CommandServerGrpc.CommandServerImplBase serverImpl =
        new CommandServerGrpc.CommandServerImplBase() {
          @Override
          public void run(RunRequest request, StreamObserver<RunResponse> observer) {
            ServerCallStreamObserver<RunResponse> serverCallStreamObserver =
                (ServerCallStreamObserver<RunResponse>) observer;
            BlockingStreamObserver<RunResponse> blockingStreamObserver =
                new BlockingStreamObserver<>(serverCallStreamObserver);
            Thread t =
                new Thread(
                    () -> {
                      RunResponse response =
                          RunResponse.newBuilder()
                              .setStandardOutput(ByteString.copyFrom(new byte[1024]))
                              .build();
                      // We want to trigger isReady() -> false, and we use sentCount to control
                      // whether to sleep on the client side. Therefore, we only set sentCount after
                      // isReady() changes.
                      int sent = 0;
                      while (serverCallStreamObserver.isReady()) {
                        blockingStreamObserver.onNext(response);
                        sent++;
                      }
                      sentCount.set(sent);
                      // If the current thread is interrupted, the subsequent onNext calls should
                      // not hang, but complete eventually (they may block on flow control).
                      Thread.currentThread().interrupt();
                      for (int i = 0; i < 10; i++) {
                        blockingStreamObserver.onNext(response);
                        sentCount.incrementAndGet();
                      }
                      blockingStreamObserver.onCompleted();
                      serverDone.countDown();
                    });
            t.start();
          }
        };

    String uniqueName = InProcessServerBuilder.generateName();
    // Do not use .directExecutor here, as it makes both client and server run in the same thread.
    Server server =
        InProcessServerBuilder.forName(uniqueName)
            .addService(serverImpl)
            .executor(Executors.newFixedThreadPool(4))
            .build()
            .start();
    ManagedChannel channel =
        InProcessChannelBuilder.forName(uniqueName)
            .executor(Executors.newFixedThreadPool(4))
            .build();

    CommandServerStub stub = CommandServerGrpc.newStub(channel);
    stub.run(
        RunRequest.getDefaultInstance(),
        new StreamObserver<RunResponse>() {
          @Override
          public void onNext(RunResponse value) {
            if (sentCount.get() == 0) {
              try {
                Thread.sleep(1);
              } catch (InterruptedException e) {
                throw new IllegalStateException(e);
              }
            }
            receiveCount.incrementAndGet();
          }

          @Override
          public void onError(Throwable t) {
            throw new IllegalStateException(t);
          }

          @Override
          public void onCompleted() {
            clientDone.countDown();
          }
        });
    serverDone.await();
    clientDone.await();
    assertThat(sentCount.get()).isEqualTo(receiveCount.get());
    server.shutdown();
    server.awaitTermination();
  }
}
