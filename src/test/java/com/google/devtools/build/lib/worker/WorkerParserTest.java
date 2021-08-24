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

import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link WorkerParser}. */
@RunWith(JUnit4.class)
public class WorkerParserTest {

  final FileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);

  @Test
  public void workerKeyComputationCheck() throws ExecException, IOException, InterruptedException {
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
    assertThat(keyMultiSandboxedNoDynamic.isMultiplex()).isFalse();
    assertThat(keyMultiSandboxedNoDynamic.isSandboxed()).isTrue();
    assertThat(keyMultiSandboxedNoDynamic.getWorkerTypeName()).isEqualTo("worker");

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
}
