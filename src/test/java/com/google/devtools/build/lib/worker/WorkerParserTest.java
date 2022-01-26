// Copyright 2021 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.worker;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.actions.ExecutionRequirements.SUPPORTS_MULTIPLEX_SANDBOXING;

import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link WorkerParser}. */
@RunWith(JUnit4.class)
public class WorkerParserTest {

  final FileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);

  @Test
  public void workerKeyComputationCheck() {
    WorkerKey keyNomultiNoSandboxedNoDynamic = TestUtils.createWorkerKey(fs, false, false, false);
    assertThat(keyNomultiNoSandboxedNoDynamic.isMultiplex()).isFalse();
    assertThat(keyNomultiNoSandboxedNoDynamic.isSandboxed()).isFalse();
    assertThat(keyNomultiNoSandboxedNoDynamic.getWorkerTypeName()).isEqualTo("worker");

    WorkerKey keyMultiNoSandboxedNoDynamic = TestUtils.createWorkerKey(fs, true, false, false);
    assertThat(keyMultiNoSandboxedNoDynamic.isMultiplex()).isTrue();
    assertThat(keyMultiNoSandboxedNoDynamic.isSandboxed()).isFalse();
    assertThat(keyMultiNoSandboxedNoDynamic.getWorkerTypeName()).isEqualTo("multiplex-worker");

    WorkerKey keyNomultiSandboxedNoDynamic = TestUtils.createWorkerKey(fs, false, true, false);
    assertThat(keyNomultiSandboxedNoDynamic.isMultiplex()).isFalse();
    assertThat(keyNomultiSandboxedNoDynamic.isSandboxed()).isTrue();
    assertThat(keyNomultiSandboxedNoDynamic.getWorkerTypeName()).isEqualTo("worker");

    WorkerKey keyMultiSandboxedNoDynamic = TestUtils.createWorkerKey(fs, true, true, false);
    assertThat(keyMultiSandboxedNoDynamic.isMultiplex()).isTrue();
    assertThat(keyMultiSandboxedNoDynamic.isSandboxed()).isFalse();
    assertThat(keyMultiSandboxedNoDynamic.getWorkerTypeName()).isEqualTo("multiplex-worker");

    WorkerKey keyNomultiNoSandboxedDynamic = TestUtils.createWorkerKey(fs, false, false, true);
    assertThat(keyNomultiNoSandboxedDynamic.isMultiplex()).isFalse();
    assertThat(keyNomultiNoSandboxedDynamic.isSandboxed()).isTrue();
    assertThat(keyNomultiNoSandboxedDynamic.getWorkerTypeName()).isEqualTo("worker");

    WorkerKey keyMultiNoSandboxedDynamic = TestUtils.createWorkerKey(fs, true, false, true);
    assertThat(keyMultiNoSandboxedDynamic.isMultiplex()).isFalse();
    assertThat(keyMultiNoSandboxedDynamic.isSandboxed()).isTrue();
    assertThat(keyMultiNoSandboxedDynamic.getWorkerTypeName()).isEqualTo("worker");

    WorkerKey keyNomultiSandboxedDynamic = TestUtils.createWorkerKey(fs, false, true, true);
    assertThat(keyNomultiSandboxedDynamic.isMultiplex()).isFalse();
    assertThat(keyNomultiSandboxedDynamic.isSandboxed()).isTrue();
    assertThat(keyNomultiSandboxedDynamic.getWorkerTypeName()).isEqualTo("worker");

    WorkerKey keyMultiSandboxedDynamic = TestUtils.createWorkerKey(fs, true, true, true);
    assertThat(keyMultiSandboxedDynamic.isMultiplex()).isFalse();
    assertThat(keyMultiSandboxedDynamic.isSandboxed()).isTrue();
    assertThat(keyMultiSandboxedDynamic.getWorkerTypeName()).isEqualTo("worker");
  }

  @Test
  public void createWorkerKey_understandsMultiplexSandboxing() {
    WorkerOptions options = new WorkerOptions();
    options.multiplexSandboxing = false;
    options.workerMultiplex = true;

    WorkerKey keyNoMultiplexSandboxing =
        TestUtils.createWorkerKeyWithRequirements(fs.getPath("/outputbase"), options, "Nom", false);
    assertThat(keyNoMultiplexSandboxing.isMultiplex()).isTrue();
    assertThat(keyNoMultiplexSandboxing.isSandboxed()).isFalse();
    assertThat(keyNoMultiplexSandboxing.getWorkerTypeName()).isEqualTo("multiplex-worker");

    WorkerKey keyForcedSandboxedDynamic =
        TestUtils.createWorkerKeyWithRequirements(fs.getPath("/outputbase"), options, "Nom", true);
    assertThat(keyForcedSandboxedDynamic.isMultiplex()).isFalse();
    assertThat(keyForcedSandboxedDynamic.isSandboxed()).isTrue();
    assertThat(keyForcedSandboxedDynamic.getWorkerTypeName()).isEqualTo("worker");

    WorkerKey keyForcedeMultiplexSandboxing =
        TestUtils.createWorkerKeyWithRequirements(
            fs.getPath("/outputbase"), options, "Nom", true, SUPPORTS_MULTIPLEX_SANDBOXING);
    assertThat(keyForcedeMultiplexSandboxing.isMultiplex()).isFalse();
    assertThat(keyForcedeMultiplexSandboxing.isSandboxed()).isTrue();
    assertThat(keyForcedeMultiplexSandboxing.getWorkerTypeName()).isEqualTo("worker");

    options.multiplexSandboxing = true;

    WorkerKey keyBaseMultiplexNoSandbox =
        TestUtils.createWorkerKeyWithRequirements(fs.getPath("/outputbase"), options, "Nom", false);
    assertThat(keyBaseMultiplexNoSandbox.isMultiplex()).isTrue();
    assertThat(keyBaseMultiplexNoSandbox.isSandboxed()).isFalse();
    assertThat(keyBaseMultiplexNoSandbox.getWorkerTypeName()).isEqualTo("multiplex-worker");

    WorkerKey keyBaseMultiplexSandboxing =
        TestUtils.createWorkerKeyWithRequirements(
            fs.getPath("/outputbase"), options, "Nom", false, SUPPORTS_MULTIPLEX_SANDBOXING);
    assertThat(keyBaseMultiplexSandboxing.isMultiplex()).isTrue();
    assertThat(keyBaseMultiplexSandboxing.isSandboxed()).isTrue();
    assertThat(keyBaseMultiplexSandboxing.getWorkerTypeName()).isEqualTo("multiplex-worker");

    WorkerKey keyDynamicMultiplexSandboxing =
        TestUtils.createWorkerKeyWithRequirements(
            fs.getPath("/outputbase"), options, "Nom", true, SUPPORTS_MULTIPLEX_SANDBOXING);
    assertThat(keyDynamicMultiplexSandboxing.isMultiplex()).isTrue();
    assertThat(keyDynamicMultiplexSandboxing.isSandboxed()).isTrue();
    assertThat(keyDynamicMultiplexSandboxing.getWorkerTypeName()).isEqualTo("multiplex-worker");
  }
}
