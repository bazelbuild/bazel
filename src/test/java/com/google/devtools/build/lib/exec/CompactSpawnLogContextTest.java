// Copyright 2023 The Bazel Authors. All Rights Reserved.
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
package com.google.devtools.build.lib.exec;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.truth.Truth.assertThat;

import com.github.luben.zstd.ZstdInputStream;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.exec.Protos.Digest;
import com.google.devtools.build.lib.exec.Protos.ExecLogEntry;
import com.google.devtools.build.lib.exec.Protos.File;
import com.google.devtools.build.lib.exec.Protos.SpawnExec;
import com.google.devtools.build.lib.exec.util.SpawnBuilder;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.common.options.Options;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.SortedMap;
import java.util.TreeMap;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for {@link CompactSpawnLogContext}. */
@RunWith(TestParameterInjector.class)
public final class CompactSpawnLogContextTest extends SpawnLogContextTestBase {
  private final Path logPath = fs.getPath("/log");

  @Test
  public void testTransitiveNestedSet(@TestParameter InputsMode inputsMode) throws Exception {
    Artifact file1 = ActionsTestUtil.createArtifact(rootDir, "file1");
    Artifact file2 = ActionsTestUtil.createArtifact(rootDir, "file2");
    Artifact file3 = ActionsTestUtil.createArtifact(rootDir, "file3");

    writeFile(file1, "abc");
    writeFile(file2, "def");
    writeFile(file3, "ghi");

    NestedSet<ActionInput> inputs =
        NestedSetBuilder.<ActionInput>stableOrder()
            .add(file1)
            .addTransitive(
                NestedSetBuilder.<ActionInput>stableOrder().add(file2).add(file3).build())
            .build();

    assertThat(inputs.getLeaves()).hasSize(1);
    assertThat(inputs.getNonLeaves()).hasSize(1);

    SpawnBuilder spawn = defaultSpawnBuilder().withInputs(inputs);
    if (inputsMode.isTool()) {
      spawn = spawn.withTools(inputs);
    }

    SpawnLogContext context = createSpawnLogContext();

    context.logSpawn(
        spawn.build(),
        createInputMetadataProvider(file1, file2, file3),
        createInputMap(file1, file2, file3),
        fs,
        defaultTimeout(),
        defaultSpawnResult());

    closeAndAssertLog(
        context,
        defaultSpawnExecBuilder()
            .addInputs(
                File.newBuilder()
                    .setPath("file1")
                    .setDigest(getDigest("abc"))
                    .setIsTool(inputsMode.isTool()))
            .addInputs(
                File.newBuilder()
                    .setPath("file2")
                    .setDigest(getDigest("def"))
                    .setIsTool(inputsMode.isTool()))
            .addInputs(
                File.newBuilder()
                    .setPath("file3")
                    .setDigest(getDigest("ghi"))
                    .setIsTool(inputsMode.isTool()))
            .build());
  }

  @Override
  protected SpawnLogContext createSpawnLogContext(ImmutableMap<String, String> platformProperties)
      throws IOException {
    RemoteOptions remoteOptions = Options.getDefaults(RemoteOptions.class);
    remoteOptions.remoteDefaultExecProperties = platformProperties.entrySet().asList();

    return new CompactSpawnLogContext(
        logPath,
        execRoot.asFragment(),
        remoteOptions,
        DigestHashFunction.SHA256,
        SyscallCache.NO_CACHE);
  }

  @Override
  protected void closeAndAssertLog(SpawnLogContext context, SpawnExec... expected)
      throws IOException {
    context.close();

    HashMap<Integer, ExecLogEntry> entryMap = new HashMap<>();

    ArrayList<SpawnExec> actual = new ArrayList<>();
    try (InputStream in = new ZstdInputStream(logPath.getInputStream())) {
      while (in.available() > 0) {
        ExecLogEntry e = ExecLogEntry.parseDelimitedFrom(in);
        entryMap.put(e.getId(), e);
        if (e.hasSpawn()) {
          actual.add(reconstructSpawnExec(e.getSpawn(), entryMap));
        }
      }
    }

    assertThat(actual).containsExactlyElementsIn(expected).inOrder();
  }

  private SpawnExec reconstructSpawnExec(
      ExecLogEntry.Spawn entry, Map<Integer, ExecLogEntry> entryMap) {
    SpawnExec.Builder builder =
        SpawnExec.newBuilder()
            .addAllCommandArgs(entry.getArgsList())
            .addAllEnvironmentVariables(entry.getEnvVarsList())
            .setTargetLabel(entry.getTargetLabel())
            .setMnemonic(entry.getMnemonic())
            .setExitCode(entry.getExitCode())
            .setStatus(entry.getStatus())
            .setRunner(entry.getRunner())
            .setCacheHit(entry.getCacheHit())
            .setRemotable(entry.getRemotable())
            .setCacheable(entry.getCacheable())
            .setRemoteCacheable(entry.getRemoteCacheable())
            .setTimeoutMillis(entry.getTimeoutMillis())
            .setMetrics(entry.getMetrics());

    if (entry.hasPlatform()) {
      builder.setPlatform(entry.getPlatform());
    }

    SortedMap<String, File> inputs = reconstructInputs(entry.getInputSetId(), entryMap);
    SortedMap<String, File> toolInputs = reconstructInputs(entry.getToolSetId(), entryMap);

    for (Map.Entry<String, File> e : inputs.entrySet()) {
      File file = e.getValue();
      if (toolInputs.containsKey(e.getKey())) {
        file = file.toBuilder().setIsTool(true).build();
      }
      builder.addInputs(file);
    }

    for (ExecLogEntry.Output output : entry.getOutputsList()) {
      switch (output.getTypeCase()) {
        case FILE_ID:
          ExecLogEntry.File file = checkNotNull(entryMap.get(output.getFileId())).getFile();
          builder.addListedOutputs(file.getPath());
          builder.addActualOutputs(reconstructFile(/* parentDir= */ null, file));
          break;
        case DIRECTORY_ID:
          ExecLogEntry.Directory dir =
              checkNotNull(entryMap.get(output.getDirectoryId())).getDirectory();
          builder.addListedOutputs(dir.getPath());
          for (ExecLogEntry.File dirFile : dir.getFilesList()) {
            builder.addActualOutputs(reconstructFile(dir, dirFile));
          }
          break;
        case MISSING_PATH:
          builder.addListedOutputs(output.getMissingPath());
          break;
        case TYPE_NOT_SET:
          throw new AssertionError("malformed log entry");
      }
    }

    if (!entry.getDigest().isEmpty()) {
      builder.setDigest(Digest.newBuilder().setHash(entry.getDigest()).build());
    }

    return builder.build();
  }

  private SortedMap<String, File> reconstructInputs(
      int setId, Map<Integer, ExecLogEntry> entryMap) {
    TreeMap<String, File> inputs = new TreeMap<>();
    ArrayDeque<Integer> setsToVisit = new ArrayDeque<>();
    if (setId != 0) {
      setsToVisit.addLast(setId);
    }
    while (!setsToVisit.isEmpty()) {
      ExecLogEntry.InputSet set =
          checkNotNull(entryMap.get(setsToVisit.removeFirst())).getInputSet();
      for (int fileId : set.getFileIdsList()) {
        ExecLogEntry.File file = checkNotNull(entryMap.get(fileId)).getFile();
        inputs.put(file.getPath(), reconstructFile(null, file));
      }
      for (int dirId : set.getDirectoryIdsList()) {
        ExecLogEntry.Directory dir = checkNotNull(entryMap.get(dirId)).getDirectory();
        for (ExecLogEntry.File dirFile : dir.getFilesList()) {
          inputs.put(dirFile.getPath(), reconstructFile(dir, dirFile));
        }
      }
      setsToVisit.addAll(set.getTransitiveSetIdsList());
    }
    return inputs;
  }

  private File reconstructFile(@Nullable ExecLogEntry.Directory parentDir, ExecLogEntry.File file) {
    String path = parentDir != null ? parentDir.getPath() + "/" + file.getPath() : file.getPath();
    File.Builder builder = File.newBuilder().setPath(path);
    if (file.getSizeBytes() > 0) {
      builder.setDigest(
          Digest.newBuilder()
              .setSizeBytes(file.getSizeBytes())
              .setHash(file.getDigest())
              .setHashFunctionName(digestHashFunction.toString()));
    }
    return builder.build();
  }
}
