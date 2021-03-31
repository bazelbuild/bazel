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
import static com.google.devtools.build.lib.worker.TestUtils.createWorkerKey;
import static org.junit.Assert.assertThrows;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionRequirements.WorkerProtocolFormat;
import com.google.devtools.build.lib.actions.MetadataProvider;
import com.google.devtools.build.lib.actions.ResourceManager;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnMetrics;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.exec.SpawnRunner.ProgressStatus;
import com.google.devtools.build.lib.exec.SpawnRunner.SpawnExecutionContext;
import com.google.devtools.build.lib.exec.local.LocalEnvProvider;
import com.google.devtools.build.lib.sandbox.SandboxHelpers;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxInputs;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxOutputs;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkResponse;
import java.io.IOException;
import org.apache.commons.pool2.PooledObject;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnit;
import org.mockito.junit.MockitoRule;

/** Unit tests for the WorkerSpawnRunner. */
@RunWith(JUnit4.class)
public class WorkerSpawnRunnerTest {
  final FileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
  @Rule public final MockitoRule mockito = MockitoJUnit.rule();
  @Mock ExtendedEventHandler reporter;
  @Mock LocalEnvProvider localEnvProvider;
  @Mock ResourceManager resourceManager;
  @Mock SpawnMetrics.Builder spawnMetrics;
  @Mock Spawn spawn;
  @Mock SpawnExecutionContext context;
  @Mock MetadataProvider inputFileCache;
  @Mock Worker worker;
  @Mock WorkerOptions options;

  @Before
  public void setUp() {
    when(spawn.getInputFiles()).thenReturn(NestedSetBuilder.emptySet(Order.COMPILE_ORDER));
    when(context.getArtifactExpander()).thenReturn((artifact, output) -> {});
  }

  private WorkerPool createWorkerPool() {
    return new WorkerPool(
        new WorkerFactory(options, fs.getPath("/workerBase")) {
          @Override
          public Worker create(WorkerKey key) {
            return worker;
          }

          @Override
          public boolean validateObject(WorkerKey key, PooledObject<Worker> p) {
            return true;
          }
        },
        ImmutableMap.of(),
        ImmutableMap.of(),
        ImmutableList.of());
  }

  @Test
  public void testExecInWorker_happyPath() throws ExecException, InterruptedException, IOException {
    WorkerSpawnRunner runner =
        new WorkerSpawnRunner(
            new SandboxHelpers(false),
            fs.getPath("/execRoot"),
            createWorkerPool(),
            /* multiplex */ false,
            reporter,
            localEnvProvider,
            /* binTools */ null,
            resourceManager,
            /* runfilestTreeUpdater */ null,
            new WorkerOptions());
    WorkerKey key = createWorkerKey(fs, "mnem", false);
    Path logFile = fs.getPath("/worker.log");
    when(worker.getLogFile()).thenReturn(logFile);
    when(worker.getResponse(0))
        .thenReturn(WorkResponse.newBuilder().setExitCode(0).setOutput("out").build());
    WorkResponse response =
        runner.execInWorker(
            spawn,
            key,
            context,
            new SandboxInputs(ImmutableMap.of(), ImmutableSet.of(), ImmutableMap.of()),
            SandboxOutputs.create(ImmutableSet.of(), ImmutableSet.of()),
            ImmutableList.of(),
            inputFileCache,
            spawnMetrics);

    assertThat(response).isNotNull();
    assertThat(response.getExitCode()).isEqualTo(0);
    assertThat(response.getRequestId()).isEqualTo(0);
    assertThat(response.getOutput()).isEqualTo("out");
    assertThat(logFile.exists()).isFalse();
    verify(context, times(1)).report(ProgressStatus.EXECUTING, "worker");
  }

  @Test
  public void testExecInWorker_noMultiplexWithDynamic()
      throws ExecException, InterruptedException, IOException {
    WorkerOptions workerOptions = new WorkerOptions();
    workerOptions.workerMultiplex = true;
    when(context.speculating()).thenReturn(true);
    WorkerSpawnRunner runner =
        new WorkerSpawnRunner(
            new SandboxHelpers(false),
            fs.getPath("/execRoot"),
            createWorkerPool(),
            /* multiplex */ true,
            reporter,
            localEnvProvider,
            /* binTools */ null,
            resourceManager,
            /* runfilestTreeUpdater */ null,
            workerOptions);
    // This worker key just so happens to be multiplex and require sandboxing.
    WorkerKey key = createWorkerKey(WorkerProtocolFormat.JSON, fs);
    Path logFile = fs.getPath("/worker.log");
    when(worker.getLogFile()).thenReturn(logFile);
    when(worker.getResponse(0))
        .thenReturn(
            WorkResponse.newBuilder().setExitCode(0).setRequestId(0).setOutput("out").build());
    WorkResponse response =
        runner.execInWorker(
            spawn,
            key,
            context,
            new SandboxInputs(ImmutableMap.of(), ImmutableSet.of(), ImmutableMap.of()),
            SandboxOutputs.create(ImmutableSet.of(), ImmutableSet.of()),
            ImmutableList.of(),
            inputFileCache,
            spawnMetrics);

    assertThat(response).isNotNull();
    assertThat(response.getExitCode()).isEqualTo(0);
    assertThat(response.getRequestId()).isEqualTo(0);
    assertThat(response.getOutput()).isEqualTo("out");
    assertThat(logFile.exists()).isFalse();
    verify(context, times(1)).report(ProgressStatus.EXECUTING, "worker");
  }

  private void assertRecordedResponsethrowsException(String recordedResponse, String exceptionText)
      throws Exception {
    WorkerSpawnRunner runner =
        new WorkerSpawnRunner(
            new SandboxHelpers(false),
            fs.getPath("/execRoot"),
            createWorkerPool(),
            /* multiplex */ false,
            reporter,
            localEnvProvider,
            /* binTools */ null,
            resourceManager,
            /* runfilestTreeUpdater */ null,
            new WorkerOptions());
    WorkerKey key = createWorkerKey(fs, "mnem", false);
    Path logFile = fs.getPath("/worker.log");
    when(worker.getLogFile()).thenReturn(logFile);
    when(worker.getResponse(0)).thenThrow(new IOException("Bad protobuf"));
    when(worker.getRecordingStreamMessage()).thenReturn(recordedResponse);
    String workerLog = "Log from worker\n";
    FileSystemUtils.writeIsoLatin1(logFile, workerLog);
    UserExecException execException =
        assertThrows(
            UserExecException.class,
            () ->
                runner.execInWorker(
                    spawn,
                    key,
                    context,
                    new SandboxInputs(ImmutableMap.of(), ImmutableSet.of(), ImmutableMap.of()),
                    SandboxOutputs.create(ImmutableSet.of(), ImmutableSet.of()),
                    ImmutableList.of(),
                    inputFileCache,
                    spawnMetrics));

    assertThat(execException).hasMessageThat().contains(exceptionText);
    if (!recordedResponse.isEmpty()) {
      assertThat(execException)
          .hasMessageThat()
          .contains(logMarker("Exception details") + "java.io.IOException: Bad protobuf");

      assertThat(execException)
          .hasMessageThat()
          .contains(
              logMarker("Start of response") + recordedResponse + logMarker("End of response"));
    }
    assertThat(execException)
        .hasMessageThat()
        .contains(logMarker("Start of log, file at " + logFile.getPathString()) + workerLog);
  }

  @Test
  public void testExecInWorker_showsLogFileInException() throws Exception {
    assertRecordedResponsethrowsException("Some text", "unparseable WorkResponse!\n");
  }

  @Test
  public void testExecInWorker_throwsWithEmptyResponse() throws Exception {
    assertRecordedResponsethrowsException("", "did not return a WorkResponse");
  }

  private static String logMarker(String text) {
    return "---8<---8<--- " + text + " ---8<---8<---\n";
  }
}
