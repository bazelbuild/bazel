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

import com.google.devtools.build.lib.util.OS;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link SandboxBackendTransport}. */
@RunWith(JUnit4.class)
public final class SandboxBackendTransportTest {
  @Test
  public void isSupported_notOnWindows() {
    assertThat(SandboxBackendTransport.isSupported(OS.WINDOWS)).isFalse();
    assertThat(SandboxBackendTransport.isSupported(OS.LINUX)).isTrue();
    assertThat(SandboxBackendTransport.isSupported(OS.DARWIN)).isTrue();
  }

  @Test
  public void create_isConsistentWithIsSupported() throws Exception {
    if (SandboxBackendTransport.isSupported()) {
      try (SandboxBackendTransport transport = SandboxBackendTransport.create()) {
        assertThat(transport.socketPath()).endsWith("backend.sock");
      }
    }
  }
}
