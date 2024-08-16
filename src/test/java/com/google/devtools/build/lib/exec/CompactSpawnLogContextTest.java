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

import static com.google.common.truth.Truth.assertThat;

import com.github.luben.zstd.ZstdInputStream;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.BuildConfigurationEvent;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.actions.SymlinkAction;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.exec.Protos.File;
import com.google.devtools.build.lib.exec.Protos.SpawnExec;
import com.google.devtools.build.lib.exec.util.SpawnBuilder;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.common.options.Options;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import net.starlark.java.syntax.Location;
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

  @Test
  public void testSymlinkAction() throws IOException, InterruptedException {
    Artifact source = ActionsTestUtil.createArtifact(rootDir, "source");
    Artifact target = ActionsTestUtil.createArtifact(rootDir, "target");
    ActionOwner owner =
        ActionOwner.createDummy(
            Label.parseCanonicalUnchecked("//pkg:symlink"),
            new Location("dummy-file", 0, 0),
            "some_rule",
            "configurationMnemonic",
            /* configurationChecksum= */ "configurationChecksum",
            new BuildConfigurationEvent(
                BuildEventStreamProtos.BuildEventId.getDefaultInstance(),
                BuildEventStreamProtos.BuildEvent.getDefaultInstance()),
            /* isToolConfiguration= */ false,
            /* executionPlatform= */ null,
            /* aspectDescriptors= */ ImmutableList.of(),
            /* execProperties= */ ImmutableMap.of());
    SymlinkAction symlinkAction =
        SymlinkAction.toArtifact(owner, source, target, "Creating symlink");

    SpawnLogContext context = createSpawnLogContext();
    context.logSymlinkAction(symlinkAction);
    context.close();

    var entries = new ArrayList<Protos.ExecLogEntry>();
    try (InputStream in = logPath.getInputStream();
        ZstdInputStream zstdIn = new ZstdInputStream(in)) {
      Protos.ExecLogEntry entry;
      while ((entry = Protos.ExecLogEntry.parseDelimitedFrom(zstdIn)) != null) {
        entries.add(entry);
      }
    }

    assertThat(entries)
        .containsExactly(
            Protos.ExecLogEntry.newBuilder()
                .setId(1)
                .setInvocation(
                    Protos.ExecLogEntry.Invocation.newBuilder().setHashFunctionName("SHA-256"))
                .build(),
            Protos.ExecLogEntry.newBuilder()
                .setId(2)
                .setSymlinkAction(
                    Protos.ExecLogEntry.SymlinkAction.newBuilder()
                        .setInputPath("source")
                        .setOutputPath("target")
                        .setMnemonic("Symlink")
                        .setTargetLabel("//pkg:symlink"))
                .build());
  }

  @Override
  protected SpawnLogContext createSpawnLogContext(ImmutableMap<String, String> platformProperties)
      throws IOException, InterruptedException {
    RemoteOptions remoteOptions = Options.getDefaults(RemoteOptions.class);
    remoteOptions.remoteDefaultExecProperties = platformProperties.entrySet().asList();

    return new CompactSpawnLogContext(
        logPath,
        execRoot.asFragment(),
        TestConstants.WORKSPACE_NAME,
        remoteOptions,
        DigestHashFunction.SHA256,
        SyscallCache.NO_CACHE);
  }

  @Override
  protected void closeAndAssertLog(SpawnLogContext context, SpawnExec... expected)
      throws IOException, InterruptedException {
    context.close();

    ArrayList<SpawnExec> actual = new ArrayList<>();
    try (SpawnLogReconstructor reconstructor =
        new SpawnLogReconstructor(logPath.getInputStream())) {
      SpawnExec ex;
      while ((ex = reconstructor.read()) != null) {
        actual.add(ex);
      }
    }

    assertThat(actual).containsExactlyElementsIn(expected).inOrder();
  }
}
