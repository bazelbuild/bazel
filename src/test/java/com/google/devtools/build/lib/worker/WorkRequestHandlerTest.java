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

package com.google.devtools.build.lib.worker;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.worker.WorkRequestHandler.RequestInfo;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkRequest;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkResponse;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.MockitoAnnotations;

/** Tests for the WorkRequestHandler */
@RunWith(JUnit4.class)
public class WorkRequestHandlerTest {

  @Before
  public void init() {
    MockitoAnnotations.initMocks(this);
  }

  @Test
  public void testNormalWorkRequest() throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    WorkRequestHandler handler =
        new WorkRequestHandler(
            (args, err) -> 1,
            new PrintStream(new ByteArrayOutputStream()),
            new ProtoWorkerMessageProcessor(new ByteArrayInputStream(new byte[0]), out));

    List<String> args = Arrays.asList("--sources", "A.java");
    WorkRequest request = WorkRequest.newBuilder().addAllArguments(args).build();
    handler.respondToRequest(request, new RequestInfo());

    WorkResponse response =
        WorkResponse.parseDelimitedFrom(new ByteArrayInputStream(out.toByteArray()));
    assertThat(response.getRequestId()).isEqualTo(0);
    assertThat(response.getExitCode()).isEqualTo(1);
    assertThat(response.getOutput()).isEmpty();
  }

  @Test
  public void testMultiplexWorkRequest() throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    WorkRequestHandler handler =
        new WorkRequestHandler(
            (args, err) -> 0,
            new PrintStream(new ByteArrayOutputStream()),
            new ProtoWorkerMessageProcessor(new ByteArrayInputStream(new byte[0]), out));

    List<String> args = Arrays.asList("--sources", "A.java");
    WorkRequest request = WorkRequest.newBuilder().addAllArguments(args).setRequestId(42).build();
    handler.respondToRequest(request, new RequestInfo());

    WorkResponse response =
        WorkResponse.parseDelimitedFrom(new ByteArrayInputStream(out.toByteArray()));
    assertThat(response.getRequestId()).isEqualTo(42);
    assertThat(response.getExitCode()).isEqualTo(0);
    assertThat(response.getOutput()).isEmpty();
  }

  @Test
  public void testOutput() throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    WorkRequestHandler handler =
        new WorkRequestHandler(
            (args, err) -> {
              err.println("Failed!");
              return 1;
            },
            new PrintStream(new ByteArrayOutputStream()),
            new ProtoWorkerMessageProcessor(new ByteArrayInputStream(new byte[0]), out));

    List<String> args = Arrays.asList("--sources", "A.java");
    WorkRequest request = WorkRequest.newBuilder().addAllArguments(args).build();
    handler.respondToRequest(request, new RequestInfo());

    WorkResponse response =
        WorkResponse.parseDelimitedFrom(new ByteArrayInputStream(out.toByteArray()));
    assertThat(response.getRequestId()).isEqualTo(0);
    assertThat(response.getExitCode()).isEqualTo(1);
    assertThat(response.getOutput()).contains("Failed!");
  }

  @Test
  public void testException() throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    WorkRequestHandler handler =
        new WorkRequestHandler(
            (args, err) -> {
              throw new RuntimeException("Exploded!");
            },
            new PrintStream(new ByteArrayOutputStream()),
            new ProtoWorkerMessageProcessor(new ByteArrayInputStream(new byte[0]), out));

    List<String> args = Arrays.asList("--sources", "A.java");
    WorkRequest request = WorkRequest.newBuilder().addAllArguments(args).build();
    handler.respondToRequest(request, new RequestInfo());

    WorkResponse response =
        WorkResponse.parseDelimitedFrom(new ByteArrayInputStream(out.toByteArray()));
    assertThat(response.getRequestId()).isEqualTo(0);
    assertThat(response.getExitCode()).isEqualTo(1);
    assertThat(response.getOutput()).startsWith("java.lang.RuntimeException: Exploded!");
  }
}
