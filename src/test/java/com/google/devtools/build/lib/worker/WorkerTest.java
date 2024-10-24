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
import static com.google.devtools.build.lib.actions.ExecutionRequirements.WorkerProtocolFormat.JSON;
import static com.google.devtools.build.lib.actions.ExecutionRequirements.WorkerProtocolFormat.PROTO;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertThrows;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.actions.ExecutionRequirements.WorkerProtocolFormat;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxInputs;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxOutputs;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.lib.worker.WorkerProtocol.Input;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkRequest;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkResponse;
import com.google.devtools.build.lib.worker.WorkerTestUtils.FakeSubprocess;
import com.google.devtools.build.lib.worker.WorkerTestUtils.TestWorker;
import com.google.protobuf.ByteString;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import org.junit.After;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link SingleplexWorker}. */
@RunWith(JUnit4.class)
public final class WorkerTest {
  final FileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);

  private TestWorker workerForCleanup = null;
  private final WorkerOptions options = new WorkerOptions();

  @After
  public void destroyWorker() throws IOException {
    if (workerForCleanup != null) {
      workerForCleanup.destroy();
      workerForCleanup = null;
    }
  }

  private static byte[] serializeResponseToProtoBytes(WorkResponse response) throws IOException {
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    response.writeDelimitedTo(baos);
    return baos.toByteArray();
  }

  private TestWorker createTestWorker(byte[] outputStreamBytes, WorkerProtocolFormat protocolFormat)
      throws IOException, InterruptedException, UserExecException {
    Preconditions.checkState(
        workerForCleanup == null, "createTestWorker can only be called once per test");

    WorkerKey key = WorkerTestUtils.createWorkerKey(protocolFormat, fs);

    FakeSubprocess fakeSubprocess = new FakeSubprocess(outputStreamBytes);

    Path workerBaseDir = fs.getPath("/outputbase/bazel-workers");
    int workerId = 1;
    Path logFile = workerBaseDir.getRelative("test-log-file.log");

    TestWorker worker =
        new TestWorker(key, workerId, key.getExecRoot(), logFile, fakeSubprocess, options);

    SandboxInputs sandboxInputs = null;
    SandboxOutputs sandboxOutputs = null;
    worker.prepareExecution(
        sandboxInputs, sandboxOutputs, key.getWorkerFilesWithDigests().keySet());

    workerForCleanup = worker;

    return worker;
  }

  @Test
  public void testPutRequest_success() throws IOException, InterruptedException, UserExecException {
    WorkRequest request = WorkRequest.getDefaultInstance();

    TestWorker testWorker = createTestWorker(new byte[0], PROTO);
    testWorker.putRequest(request);

    OutputStream stdout = testWorker.getFakeSubprocess().getOutputStream();
    WorkRequest requestFromStdout =
        WorkRequest.parseDelimitedFrom(new ByteArrayInputStream(stdout.toString().getBytes(UTF_8)));

    assertThat(requestFromStdout).isEqualTo(request);
  }

  @Test
  public void testGetResponse_success()
      throws IOException, InterruptedException, UserExecException {
    WorkResponse response = WorkResponse.getDefaultInstance();

    TestWorker testWorker = createTestWorker(serializeResponseToProtoBytes(response), PROTO);
    WorkResponse readResponse = testWorker.getResponse(0);

    assertThat(readResponse).isEqualTo(response);
  }

  @Test
  public void testPutRequest_json_success()
      throws IOException, InterruptedException, UserExecException {
    TestWorker testWorker = createTestWorker(new byte[0], JSON);
    testWorker.putRequest(WorkRequest.getDefaultInstance());

    OutputStream stdout = testWorker.getFakeSubprocess().getOutputStream();
    assertThat(stdout.toString()).isEqualTo("{}" + System.lineSeparator());
  }

  @Test
  public void testGetResponse_json_success()
      throws IOException, InterruptedException, UserExecException {
    TestWorker testWorker = createTestWorker(("{}" + System.lineSeparator()).getBytes(UTF_8), JSON);
    WorkResponse readResponse = testWorker.getResponse(0);
    WorkResponse response = WorkResponse.getDefaultInstance();

    assertThat(readResponse).isEqualTo(response);
  }

  @Test
  public void testPutRequest_json_populatedFields_success()
      throws IOException, InterruptedException, UserExecException {
    WorkRequest request =
        WorkRequest.newBuilder()
            .addArguments("testRequest")
            .addInputs(
                Input.newBuilder()
                    .setPath("testPath")
                    .setDigest(ByteString.copyFromUtf8("testDigest"))
                    .build())
            .setRequestId(1)
            .setVerbosity(11)
            .build();

    TestWorker testWorker = createTestWorker(new byte[0], JSON);
    testWorker.putRequest(request);

    OutputStream stdout = testWorker.getFakeSubprocess().getOutputStream();
    String requestJsonString =
        "{\"arguments\":[\"testRequest\"],\"inputs\":"
            + "[{\"path\":\"testPath\",\"digest\":\"dGVzdERpZ2VzdA==\"}],\"requestId\":1,\"verbosity\":11}"
            + System.lineSeparator();
    assertThat(stdout.toString()).isEqualTo(requestJsonString);
  }

  @Test
  public void testGetResponse_json_populatedFields_success()
      throws IOException, InterruptedException, UserExecException {
    TestWorker testWorker =
        createTestWorker(
            "{\"exitCode\":1,\"output\":\"test output\",\"requestId\":1}".getBytes(UTF_8), JSON);
    WorkResponse readResponse = testWorker.getResponse(1);
    WorkResponse response =
        WorkResponse.newBuilder().setExitCode(1).setOutput("test output").setRequestId(1).build();

    assertThat(readResponse).isEqualTo(response);
  }

  private void verifyGetResponseFailure(String responseString, String expectedError)
      throws IOException, InterruptedException, UserExecException {
    TestWorker testWorker =
        createTestWorker((responseString + System.lineSeparator()).getBytes(UTF_8), JSON);
    IOException ex = assertThrows(IOException.class, () -> testWorker.getResponse(0));
    assertThat(ex).hasMessageThat().contains(expectedError);
  }

  @Test
  public void testGetResponse_badJson_throws()
      throws IOException, InterruptedException, UserExecException {
    verifyGetResponseFailure(
        "{ \"output\": \"I'm missing a bracket\"", "Could not parse json work request correctly");
  }

  @Test
  public void testGetResponse_json_multipleExitCode_fails()
      throws IOException, InterruptedException, UserExecException {
    verifyGetResponseFailure(
        "{\"exitCode\":1,\"exitCode\":1}", "Work response cannot have more than one exit code");
  }

  @Test
  public void testGetResponse_json_multipleOutput_fails()
      throws IOException, InterruptedException, UserExecException {
    verifyGetResponseFailure(
        "{\"output\":\"\",\"output\":\"\"}", "Work response cannot have more than one output");
  }

  @Test
  public void testGetResponse_json_multipleRequestId_fails()
      throws IOException, InterruptedException, UserExecException {
    verifyGetResponseFailure(
        "{\"requestId\":0,\"requestId\":0}", "Work response cannot have more than one requestId");
  }

  @Test
  public void testGetResponse_json_unknownFieldsIgnored()
      throws IOException, InterruptedException, UserExecException {
    TestWorker testWorker =
        createTestWorker(
            "{\"exitCode\":1,\"output\":\"test output\",\"requestId\":1,\"unknown\":{1:['a']}}"
                .getBytes(UTF_8),
            JSON);
    WorkResponse readResponse = testWorker.getResponse(1);
    WorkResponse response =
        WorkResponse.newBuilder().setExitCode(1).setOutput("test output").setRequestId(1).build();

    assertThat(readResponse).isEqualTo(response);
  }
}
