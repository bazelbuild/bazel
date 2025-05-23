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

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyBoolean;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.ExecuteResponse;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedMap;
import com.google.devtools.build.lib.remote.CombinedCache.CachedActionResult;
import com.google.devtools.build.lib.remote.common.RemoteExecutionClient;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.runtime.RepositoryRemoteExecutor.ExecutionResult;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.protobuf.ByteString;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

/** Tests for {@link com.google.devtools.build.lib.remote.RemoteRepositoryRemoteExecutor}. */
@RunWith(JUnit4.class)
public class RemoteRepositoryRemoteExecutorTest {
  public static final DigestUtil DIGEST_UTIL =
      new DigestUtil(SyscallCache.NO_CACHE, DigestHashFunction.SHA256);

  @Mock public RemoteExecutionCache remoteCache;

  @Mock public RemoteExecutionClient remoteExecutor;

  private RemoteRepositoryRemoteExecutor repoExecutor;

  @Before
  public void setup() {
    MockitoAnnotations.initMocks(this);
    repoExecutor =
        new RemoteRepositoryRemoteExecutor(
            remoteCache,
            remoteExecutor,
            DIGEST_UTIL,
            "none",
            "none",
            /* remoteInstanceName= */ "foo",
            /* acceptCached= */ true);
  }

  @Test
  public void testZeroExitCodeFromCache() throws IOException, InterruptedException {
    // Test that an ActionResult with exit code zero is accepted as cached.

    ActionResult cachedResult = ActionResult.newBuilder().setExitCode(0).build();
    when(remoteCache.downloadActionResult(
            any(),
            any(),
            /* inlineOutErr= */ eq(true),
            /* inlineOutputFiles= */ eq(ImmutableSet.of())))
        .thenReturn(CachedActionResult.remote(cachedResult));

    ExecutionResult executionResult =
        repoExecutor.execute(
            ImmutableList.of("/bin/bash", "-c", "exit 0"),
            /* inputFiles= */ ImmutableSortedMap.of(),
            /* executionProperties= */ ImmutableMap.of(),
            /* environment= */ ImmutableMap.of(),
            /* workingDirectory= */ null,
            /* timeout= */ Duration.ZERO);

    verify(remoteCache)
        .downloadActionResult(
            any(), any(), anyBoolean(), /* inlineOutputFiles= */ eq(ImmutableSet.of()));
    // Don't fallback to execution
    verify(remoteExecutor, never()).executeRemotely(any(), any(), any());

    assertThat(executionResult.exitCode()).isEqualTo(0);
  }

  @Test
  public void testNoneZeroExitCodeFromCache() throws IOException, InterruptedException {
    // Test that an ActionResult with a none-zero exit code is not accepted as cached.

    ActionResult cachedResult = ActionResult.newBuilder().setExitCode(1).build();
    when(remoteCache.downloadActionResult(
            any(),
            any(),
            /* inlineOutErr= */ eq(true),
            /* inlineOutputFiles= */ eq(ImmutableSet.of())))
        .thenReturn(CachedActionResult.remote(cachedResult));

    ExecuteResponse response = ExecuteResponse.newBuilder().setResult(cachedResult).build();
    when(remoteExecutor.executeRemotely(any(), any(), any())).thenReturn(response);

    ExecutionResult executionResult =
        repoExecutor.execute(
            ImmutableList.of("/bin/bash", "-c", "exit 1"),
            /* inputFiles= */ ImmutableSortedMap.of(),
            /* executionProperties= */ ImmutableMap.of(),
            /* environment= */ ImmutableMap.of(),
            /* workingDirectory= */ null,
            /* timeout= */ Duration.ZERO);

    verify(remoteCache)
        .downloadActionResult(
            any(), any(), anyBoolean(), /* inlineOutputFiles= */ eq(ImmutableSet.of()));
    // Fallback to execution
    verify(remoteExecutor).executeRemotely(any(), any(), any());

    assertThat(executionResult.exitCode()).isEqualTo(1);
  }

  @Test
  public void testInlineStdoutStderr() throws IOException, InterruptedException {
    // Test that inline stdout/stderr responses are returned in execution results.

    byte[] stdout = "hello".getBytes(StandardCharsets.UTF_8);
    byte[] stderr = "world".getBytes(StandardCharsets.UTF_8);
    ActionResult cachedResult =
        ActionResult.newBuilder()
            .setExitCode(0)
            .setStdoutRaw(ByteString.copyFrom(stdout))
            .setStderrRaw(ByteString.copyFrom(stderr))
            .build();
    when(remoteCache.downloadActionResult(
            any(),
            any(),
            /* inlineOutErr= */ eq(true),
            /* inlineOutputFiles= */ eq(ImmutableSet.of())))
        .thenReturn(CachedActionResult.remote(cachedResult));

    ExecuteResponse response = ExecuteResponse.newBuilder().setResult(cachedResult).build();
    when(remoteExecutor.executeRemotely(any(), any(), any())).thenReturn(response);

    ExecutionResult executionResult =
        repoExecutor.execute(
            ImmutableList.of("/bin/bash", "-c", "echo hello"),
            /* inputFiles= */ ImmutableSortedMap.of(),
            /* executionProperties= */ ImmutableMap.of(),
            /* environment= */ ImmutableMap.of(),
            /* workingDirectory= */ null,
            /* timeout= */ Duration.ZERO);

    verify(remoteCache)
        .downloadActionResult(
            any(),
            any(),
            /* inlineOutErr= */ eq(true),
            /* inlineOutputFiles= */ eq(ImmutableSet.of()));

    assertThat(executionResult.exitCode()).isEqualTo(0);
    assertThat(executionResult.stdout()).isEqualTo(stdout);
    assertThat(executionResult.stderr()).isEqualTo(stderr);
  }
}
