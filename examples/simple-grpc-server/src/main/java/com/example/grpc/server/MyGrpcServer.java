/*
 * Copyright 2016 Google, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.example.grpc.server;

import com.example.grpc.GreetingServiceGrpc;
import com.example.grpc.HelloRequest;
import com.example.grpc.HelloResponse;
import com.example.grpc.HelloResponseOrBuilder;
import io.grpc.Server;
import io.grpc.ServerBuilder;
import io.grpc.stub.StreamObserver;
import io.grpc.netty.NettyServerBuilder;
import java.io.IOException;
import java.net.InetSocketAddress;
import io.netty.util.NetUtil;

/**
 * Created by rayt on 5/16/16.
 */
public class MyGrpcServer {
  static public void main(String [] args) throws IOException, InterruptedException {

    if (NetUtil.isIpV4StackPreferred()) {
      throw new IOException("ipv4 is preferred on the system.");
    }

    InetSocketAddress address = new InetSocketAddress("[::1]", 8080);
//    Server server = ServerBuilder.forPort(8080).addService(new GreetingServiceImpl()).build();
    Server server = NettyServerBuilder.forAddress(address).addService(new GreetingServiceImpl()).directExecutor().build();
    System.out.println("Starting server...");
    server.start();
    System.out.println("Server started!");
    server.awaitTermination();
  }

  public static class GreetingServiceImpl extends GreetingServiceGrpc.GreetingServiceImplBase {
    @Override
    public void greeting(HelloRequest request, StreamObserver<HelloResponse> responseObserver) {
      System.out.println(request);

      String greeting = "Hello there, " + request.getName();

      HelloResponse response = HelloResponse.newBuilder().setGreeting(greeting).build();

      responseObserver.onNext(response);
      responseObserver.onCompleted();
    }
  }
}
