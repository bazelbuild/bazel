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
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.util.ArrayList;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link WorkerParser}. */
@RunWith(JUnit4.class)
public class WorkerParserTest {

  final FileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);

  @Test
  public void workerKeyComputationCheck() {
    WorkerKey keyNomultiNoSandboxedNoDynamic =
        WorkerTestUtils.createWorkerKey(fs, false, false, false);
    assertThat(keyNomultiNoSandboxedNoDynamic.isMultiplex()).isFalse();
    assertThat(keyNomultiNoSandboxedNoDynamic.isSandboxed()).isFalse();
    assertThat(keyNomultiNoSandboxedNoDynamic.getWorkerTypeName()).isEqualTo("worker");

    WorkerKey keyMultiNoSandboxedNoDynamic =
        WorkerTestUtils.createWorkerKey(fs, true, false, false);
    assertThat(keyMultiNoSandboxedNoDynamic.isMultiplex()).isTrue();
    assertThat(keyMultiNoSandboxedNoDynamic.isSandboxed()).isFalse();
    assertThat(keyMultiNoSandboxedNoDynamic.getWorkerTypeName()).isEqualTo("multiplex-worker");

    WorkerKey keyNomultiSandboxedNoDynamic =
        WorkerTestUtils.createWorkerKey(fs, false, true, false);
    assertThat(keyNomultiSandboxedNoDynamic.isMultiplex()).isFalse();
    assertThat(keyNomultiSandboxedNoDynamic.isSandboxed()).isTrue();
    assertThat(keyNomultiSandboxedNoDynamic.getWorkerTypeName()).isEqualTo("worker");

    WorkerKey keyMultiSandboxedNoDynamic = WorkerTestUtils.createWorkerKey(fs, true, true, false);
    assertThat(keyMultiSandboxedNoDynamic.isMultiplex()).isTrue();
    assertThat(keyMultiSandboxedNoDynamic.isSandboxed()).isFalse();
    assertThat(keyMultiSandboxedNoDynamic.getWorkerTypeName()).isEqualTo("multiplex-worker");

    WorkerKey keyNomultiNoSandboxedDynamic =
        WorkerTestUtils.createWorkerKey(fs, false, false, true);
    assertThat(keyNomultiNoSandboxedDynamic.isMultiplex()).isFalse();
    assertThat(keyNomultiNoSandboxedDynamic.isSandboxed()).isTrue();
    assertThat(keyNomultiNoSandboxedDynamic.getWorkerTypeName()).isEqualTo("worker");

    WorkerKey keyMultiNoSandboxedDynamic = WorkerTestUtils.createWorkerKey(fs, true, false, true);
    assertThat(keyMultiNoSandboxedDynamic.isMultiplex()).isFalse();
    assertThat(keyMultiNoSandboxedDynamic.isSandboxed()).isTrue();
    assertThat(keyMultiNoSandboxedDynamic.getWorkerTypeName()).isEqualTo("worker");

    WorkerKey keyNomultiSandboxedDynamic = WorkerTestUtils.createWorkerKey(fs, false, true, true);
    assertThat(keyNomultiSandboxedDynamic.isMultiplex()).isFalse();
    assertThat(keyNomultiSandboxedDynamic.isSandboxed()).isTrue();
    assertThat(keyNomultiSandboxedDynamic.getWorkerTypeName()).isEqualTo("worker");

    WorkerKey keyMultiSandboxedDynamic = WorkerTestUtils.createWorkerKey(fs, true, true, true);
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
        WorkerTestUtils.createWorkerKeyWithRequirements(
            fs.getPath("/outputbase"), options, "Nom", false);
    assertThat(keyNoMultiplexSandboxing.isMultiplex()).isTrue();
    assertThat(keyNoMultiplexSandboxing.isSandboxed()).isFalse();
    assertThat(keyNoMultiplexSandboxing.getWorkerTypeName()).isEqualTo("multiplex-worker");

    WorkerKey keyForcedSandboxedDynamic =
        WorkerTestUtils.createWorkerKeyWithRequirements(
            fs.getPath("/outputbase"), options, "Nom", true);
    assertThat(keyForcedSandboxedDynamic.isMultiplex()).isFalse();
    assertThat(keyForcedSandboxedDynamic.isSandboxed()).isTrue();
    assertThat(keyForcedSandboxedDynamic.getWorkerTypeName()).isEqualTo("worker");

    WorkerKey keyForcedeMultiplexSandboxing =
        WorkerTestUtils.createWorkerKeyWithRequirements(
            fs.getPath("/outputbase"), options, "Nom", true, SUPPORTS_MULTIPLEX_SANDBOXING);
    assertThat(keyForcedeMultiplexSandboxing.isMultiplex()).isFalse();
    assertThat(keyForcedeMultiplexSandboxing.isSandboxed()).isTrue();
    assertThat(keyForcedeMultiplexSandboxing.getWorkerTypeName()).isEqualTo("worker");

    options.multiplexSandboxing = true;

    WorkerKey keyBaseMultiplexNoSandbox =
        WorkerTestUtils.createWorkerKeyWithRequirements(
            fs.getPath("/outputbase"), options, "Nom", false);
    assertThat(keyBaseMultiplexNoSandbox.isMultiplex()).isTrue();
    assertThat(keyBaseMultiplexNoSandbox.isSandboxed()).isFalse();
    assertThat(keyBaseMultiplexNoSandbox.getWorkerTypeName()).isEqualTo("multiplex-worker");

    WorkerKey keyBaseMultiplexSandboxing =
        WorkerTestUtils.createWorkerKeyWithRequirements(
            fs.getPath("/outputbase"), options, "Nom", false, SUPPORTS_MULTIPLEX_SANDBOXING);
    assertThat(keyBaseMultiplexSandboxing.isMultiplex()).isTrue();
    assertThat(keyBaseMultiplexSandboxing.isSandboxed()).isTrue();
    assertThat(keyBaseMultiplexSandboxing.getWorkerTypeName()).isEqualTo("multiplex-worker");

    WorkerKey keyDynamicMultiplexSandboxing =
        WorkerTestUtils.createWorkerKeyWithRequirements(
            fs.getPath("/outputbase"), options, "Nom", true, SUPPORTS_MULTIPLEX_SANDBOXING);
    assertThat(keyDynamicMultiplexSandboxing.isMultiplex()).isTrue();
    assertThat(keyDynamicMultiplexSandboxing.isSandboxed()).isTrue();
    assertThat(keyDynamicMultiplexSandboxing.getWorkerTypeName()).isEqualTo("multiplex-worker");
  }

  @Test
  public void splitSpawnArgsIntoWorkerArgsAndFlagFiles_splitsArgsBasicCase()
      throws UserExecException {
    WorkerOptions options = new WorkerOptions();
    options.workerExtraFlags = ImmutableList.of();
    WorkerParser parser = new WorkerParser(null, options, null, null);

    Spawn spawn = WorkerTestUtils.createSpawn(ImmutableList.of("--foo", "@bar"), ImmutableMap.of());
    List<String> flagFiles = new ArrayList<>();
    ImmutableList<String> args = parser.splitSpawnArgsIntoWorkerArgsAndFlagFiles(spawn, flagFiles);
    assertThat(args).containsExactly("--foo", "--persistent_worker").inOrder();
    assertThat(flagFiles).containsExactly("@bar");
  }

  @Test
  public void splitSpawnArgsIntoWorkerArgsAndFlagFiles_addsExtras() throws UserExecException {
    WorkerOptions options = new WorkerOptions();
    options.workerExtraFlags =
        ImmutableList.of(
            Maps.immutableEntry("Null", "--qux"),
            Maps.immutableEntry("Other action", "--should_not_appear"),
            Maps.immutableEntry("Null", "--quxify"));
    WorkerParser parser = new WorkerParser(null, options, null, null);
    Spawn spawn = WorkerTestUtils.createSpawn(ImmutableList.of("--foo", "@bar"), ImmutableMap.of());

    List<String> flagFiles = new ArrayList<>();
    ImmutableList<String> args = parser.splitSpawnArgsIntoWorkerArgsAndFlagFiles(spawn, flagFiles);

    assertThat(args).containsExactly("--foo", "--persistent_worker", "--qux", "--quxify").inOrder();
    assertThat(flagFiles).containsExactly("@bar");
  }

  @Test
  public void splitSpawnArgsIntoWorkerArgsAndFlagFiles_addsFlagFiles() throws UserExecException {
    WorkerOptions options = new WorkerOptions();
    options.workerExtraFlags = ImmutableList.of();
    options.strictFlagfiles = false;
    WorkerParser parser = new WorkerParser(null, options, null, null);
    Spawn spawn =
        WorkerTestUtils.createSpawn(
            ImmutableList.of("--foo", "--flagfile=bar", "@@escaped", "@bar", "@bartoo", "--final"),
            ImmutableMap.of());

    List<String> flagFiles = new ArrayList<>();
    ImmutableList<String> args = parser.splitSpawnArgsIntoWorkerArgsAndFlagFiles(spawn, flagFiles);

    assertThat(args).containsExactly("--foo", "--final", "--persistent_worker").inOrder();
    // Yes, the legacy implementation allows multiple flagfiles and ignores escape sequences.
    assertThat(flagFiles)
        .containsExactly("--flagfile=bar", "@@escaped", "@bar", "@bartoo")
        .inOrder();
  }

  @Test
  public void splitSpawnArgsIntoWorkerArgsAndFlagFiles_addsFlagFilesStrict()
      throws UserExecException {
    WorkerOptions options = new WorkerOptions();
    options.workerExtraFlags = ImmutableList.of();
    options.strictFlagfiles = true;
    WorkerParser parser = new WorkerParser(null, options, null, null);
    Spawn spawn =
        WorkerTestUtils.createSpawn(
            ImmutableList.of("--foo", "@@escaped", "--final", "@bar"), ImmutableMap.of());

    List<String> flagFiles = new ArrayList<>();
    ImmutableList<String> args = parser.splitSpawnArgsIntoWorkerArgsAndFlagFiles(spawn, flagFiles);

    assertThat(args)
        .containsExactly("--foo", "@@escaped", "--final", "--persistent_worker")
        .inOrder();
    assertThat(flagFiles).containsExactly("@bar");
  }

  @Test
  public void splitSpawnArgsIntoWorkerArgsAndFlagFiles_strictFlagFiles() throws UserExecException {
    assertIllegalFlags("Must have args");
    assertIllegalFlags("Must have a flagfile", "--foo", "--final");
    assertIllegalFlags("Flagfile must be at the end", "@earlyFile", "--final");
    assertIllegalFlags("Only one flagfile allowed", "@earlyFile", "--final", "@lateFile");
    assertIllegalFlags(
        "Only one flagfile allowed, regardless of syntax",
        "--flagfile=foo",
        "--final",
        "@lateFile");
  }

  private void assertIllegalFlags(String message, String... args) {
    WorkerOptions options = new WorkerOptions();
    options.workerExtraFlags = ImmutableList.of();
    options.strictFlagfiles = true;
    WorkerParser parser = new WorkerParser(null, options, null, null);
    Spawn spawn = WorkerTestUtils.createSpawn(ImmutableList.copyOf(args), ImmutableMap.of());

    assertThrows(
        message,
        UserExecException.class,
        () -> parser.splitSpawnArgsIntoWorkerArgsAndFlagFiles(spawn, new ArrayList<>()));
  }
}
