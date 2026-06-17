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
package com.google.devtools.build.lib.sandbox;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.sandbox.proto.SandboxProto.Manifest;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link SandboxBackendManifest}. The input tree is now built by {@code MerkleTreeComputer}; the
 * only logic left here is the {@code --sandbox_debug} text dump.
 */
@RunWith(JUnit4.class)
public final class SandboxBackendManifestTest {

  @Test
  public void toDebugString_rendersManifestFields() {
    Manifest manifest =
        Manifest.newBuilder()
            .setMnemonic("CppCompile")
            .setHashFunction("SHA-256")
            .setExecRoot("/out/execroot")
            .putLocations("ws/bin.runfiles/x", "ws/x") // synthetic file leaf (runfiles)
            .putLocations("ws/pkg/treeart", "ws/pkg/treeart") // whole-dir entry
            .build();

    String text = SandboxBackendManifest.toDebugString(manifest);

    assertThat(text).contains("CppCompile");
    assertThat(text).contains("SHA-256");
    assertThat(text).contains("locations");
    assertThat(text).contains("ws/pkg/treeart");
  }
}
