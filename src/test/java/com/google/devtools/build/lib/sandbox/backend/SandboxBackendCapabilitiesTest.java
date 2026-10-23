// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.sandbox.backend;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import build.bazel.remote.execution.v2.CacheCapabilities;
import build.bazel.remote.execution.v2.DigestFunction;
import build.bazel.remote.execution.v2.ExecutionCapabilities;
import build.bazel.remote.execution.v2.ServerCapabilities;
import build.bazel.semver.SemVer;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.SyscallCache;
import io.grpc.Status;
import java.io.IOException;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for the check of what a sandbox backend says it supports. */
@RunWith(JUnit4.class)
public final class SandboxBackendCapabilitiesTest {
  private final DigestUtil digestUtil =
      new DigestUtil(SyscallCache.NO_CACHE, DigestHashFunction.SHA256);
  private FakeBackend backend;

  @Before
  public void setUp() throws Exception {
    backend = new FakeBackend();
  }

  @After
  public void tearDown() {
    backend.close();
  }

  private static SemVer version(int major, int minor) {
    return SemVer.newBuilder().setMajor(major).setMinor(minor).build();
  }

  private static ServerCapabilities.Builder capabilities() {
    return FakeBackend.defaultCapabilities().toBuilder();
  }

  private SandboxBackendSession start(DigestUtil digestUtil) throws IOException {
    return SandboxBackendSession.forChannel("fake", backend::newChannel, digestUtil);
  }

  @Test
  public void supportedBackend_isAskedOncePerSession() throws Exception {
    try (SandboxBackendSession session = start(digestUtil)) {
      assertThat(session.isAlive()).isTrue();
    }

    assertThat(backend.capabilitiesCalls.get()).isEqualTo(1);
  }

  @Test
  public void execDisabled_fails() {
    backend.setCapabilities(
        capabilities()
            .setExecutionCapabilities(
                ExecutionCapabilities.newBuilder()
                    .setExecEnabled(false)
                    .addDigestFunctions(DigestFunction.Value.SHA256))
            .build());

    IOException e = assertThrows(IOException.class, () -> start(digestUtil));

    assertThat(e).hasMessageThat().contains("sandbox backend 'fake' does not enable execution");
  }

  @Test
  public void digestFunctionNotSupported_fails() {
    DigestUtil sha1 = new DigestUtil(SyscallCache.NO_CACHE, DigestHashFunction.SHA1);

    IOException e = assertThrows(IOException.class, () -> start(sha1));

    assertThat(e)
        .hasMessageThat()
        .isEqualTo("sandbox backend 'fake' does not support digest function SHA1");
  }

  @Test
  public void apiVersionOutOfRange_fails() {
    backend.setCapabilities(
        capabilities().setLowApiVersion(version(3, 0)).setHighApiVersion(version(3, 1)).build());

    IOException e = assertThrows(IOException.class, () -> start(digestUtil));

    assertThat(e).hasMessageThat().contains("supports Remote Execution API versions 3.0 to 3.1");
  }

  @Test
  public void unimplemented_fails() {
    backend.failCapabilities(Status.UNIMPLEMENTED);

    IOException e = assertThrows(IOException.class, () -> start(digestUtil));

    assertThat(e).hasMessageThat().contains("does not implement GetCapabilities");
  }

  @Test
  public void otherError_failsWithStatus() {
    backend.failCapabilities(Status.UNAVAILABLE.withDescription("starting up"));

    IOException e = assertThrows(IOException.class, () -> start(digestUtil));

    assertThat(e).hasMessageThat().contains("failed GetCapabilities");
    assertThat(e).hasMessageThat().contains("starting up");
  }

  @Test
  public void check_digestFunctionOfOlderBackends() {
    // Only the singular field is set.
    ServerCapabilities singular =
        capabilities()
            .setExecutionCapabilities(
                ExecutionCapabilities.newBuilder()
                    .setExecEnabled(true)
                    .setDigestFunction(DigestFunction.Value.BLAKE3))
            .build();
    assertThat(SandboxBackendCapabilities.check("b", singular, DigestFunction.Value.BLAKE3))
        .isNull();
    assertThat(SandboxBackendCapabilities.check("b", singular, DigestFunction.Value.SHA256))
        .isEqualTo("sandbox backend 'b' does not support digest function SHA256");

    // Nothing is set, which means SHA-256.
    ServerCapabilities none =
        capabilities()
            .setExecutionCapabilities(ExecutionCapabilities.newBuilder().setExecEnabled(true))
            .build();
    assertThat(SandboxBackendCapabilities.check("b", none, DigestFunction.Value.SHA256)).isNull();
    assertThat(SandboxBackendCapabilities.check("b", none, DigestFunction.Value.BLAKE3))
        .isEqualTo("sandbox backend 'b' does not support digest function BLAKE3");
  }

  @Test
  public void check_cacheDigestFunctionsAreHonoredWhenDeclared() {
    ServerCapabilities caps =
        capabilities()
            .setCacheCapabilities(
                CacheCapabilities.newBuilder().addDigestFunctions(DigestFunction.Value.SHA1))
            .build();

    assertThat(SandboxBackendCapabilities.check("b", caps, DigestFunction.Value.SHA256))
        .isEqualTo("sandbox backend 'b' does not support digest function SHA256");
  }

  @Test
  public void check_versionRangeOnlyHasToOverlap() {
    assertThat(
            SandboxBackendCapabilities.check(
                "b",
                capabilities()
                    .setLowApiVersion(version(2, 5))
                    .setHighApiVersion(version(2, 99))
                    .build(),
                DigestFunction.Value.SHA256))
        .isNull();
    // A backend that doesn't say anything speaks version 0.
    assertThat(
            SandboxBackendCapabilities.check(
                "b",
                capabilities().clearLowApiVersion().clearHighApiVersion().build(),
                DigestFunction.Value.SHA256))
        .contains("supports Remote Execution API versions 0.0 to 0.0");
  }
}
