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

package com.google.devtools.build.buildjar;

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doReturn;

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
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;

/** Tests for the BazelJavaBuilder */
@RunWith(JUnit4.class)
public class BazelJavaBuilderTest {

  @Before
  public void init() {
    MockitoAnnotations.initMocks(this);
  }

  @Test
  public void testNormalWorkRequest() throws IOException {
    BazelJavaBuilder builder = new BazelJavaBuilder();
    BazelJavaBuilder builderSpy = Mockito.spy(builder);
    doReturn(1).when(builderSpy).parseAndProcessRequest(any(), any());

    ByteArrayOutputStream out = new ByteArrayOutputStream();
    List<String> args = Arrays.asList("--sources", "A.java");
    WorkRequest request = WorkRequest.newBuilder().addAllArguments(args).build();
    builderSpy.respondToRequest(request, new PrintStream(out));

    WorkResponse response =
        WorkResponse.parseDelimitedFrom(new ByteArrayInputStream(out.toByteArray()));
    assertThat(response.getRequestId()).isEqualTo(0);
    assertThat(response.getExitCode()).isEqualTo(1);
    assertThat(response.getOutput()).isEmpty();
  }

  @Test
  public void testMultiplexWorkRequest() throws IOException {
    BazelJavaBuilder builder = new BazelJavaBuilder();
    BazelJavaBuilder builderSpy = Mockito.spy(builder);
    doReturn(0).when(builderSpy).parseAndProcessRequest(any(), any());

    ByteArrayOutputStream out = new ByteArrayOutputStream();
    List<String> args = Arrays.asList("--sources", "A.java");
    WorkRequest request = WorkRequest.newBuilder().addAllArguments(args).setRequestId(42).build();
    builderSpy.respondToRequest(request, new PrintStream(out));

    WorkResponse response =
        WorkResponse.parseDelimitedFrom(new ByteArrayInputStream(out.toByteArray()));
    assertThat(response.getRequestId()).isEqualTo(42);
    assertThat(response.getExitCode()).isEqualTo(0);
    assertThat(response.getOutput()).isEmpty();
  }

  @Test
  public void testBrokenWorkRequest() throws IOException {
    BazelJavaBuilder builder = new BazelJavaBuilder();

    List<String> args = Arrays.asList("Foo", "Bar");
    WorkRequest request = WorkRequest.newBuilder().addAllArguments(args).build();
    ByteArrayOutputStream out = new ByteArrayOutputStream();

    builder.respondToRequest(request, new PrintStream(out));
    WorkResponse response =
        WorkResponse.parseDelimitedFrom(new ByteArrayInputStream(out.toByteArray()));

    assertThat(response.getRequestId()).isEqualTo(0);
    assertThat(response.getExitCode()).isEqualTo(1);
    assertThat(response.getOutput())
        .contains("BazelJavaBuilder threw exception: unknown option : 'Foo'");
  }
}
