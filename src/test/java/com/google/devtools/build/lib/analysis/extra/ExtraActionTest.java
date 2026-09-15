// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.extra;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.NULL_ACTION_OWNER;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.NULL_LABEL;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.ensureMemoizedIsInitializedIsSet;
import static com.google.devtools.build.lib.skyframe.serialization.testutils.Dumper.dumpStructureWithEquivalenceReduction;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionContext;
import com.google.devtools.build.lib.actions.ActionEnvironment;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionContext.LostInputsCheck;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.ActionResult;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.BuildConfigurationEvent;
import com.google.devtools.build.lib.actions.CommandLine;
import com.google.devtools.build.lib.actions.DiscoveredModulesPruner;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.SpawnStrategy;
import com.google.devtools.build.lib.actions.ThreadStateReceiver;
import com.google.devtools.build.lib.actions.extra.ExtraActionInfo;
import com.google.devtools.build.lib.actions.extra.JavaCompileInfo;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil.InputDiscoveringNullAction;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil.NullAction;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.exec.BlazeExecutor;
import com.google.devtools.build.lib.exec.util.FakeActionInputFileCache;
import com.google.devtools.build.lib.exec.util.TestExecutorBuilder;
import com.google.devtools.build.lib.skyframe.serialization.ArrayCodec;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationDepsUtils;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.SyscallCache;
import java.util.HashMap;
import java.util.Map;
import net.starlark.java.syntax.Location;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.ArgumentMatchers;
import org.mockito.Mockito;

/**
 * Unit tests for ExtraAction class.
 */
@RunWith(JUnit4.class)
public class ExtraActionTest extends FoundationTestCase {

  private final ActionKeyContext actionKeyContext = new ActionKeyContext();

  private static class SpecifiedInfoAction extends NullAction {
    private final ExtraActionInfo info;

    private SpecifiedInfoAction(ExtraActionInfo info) {
      this.info = info;
    }

    @Override
    public ExtraActionInfo.Builder getExtraActionInfo(ActionKeyContext actionKeyContext) {
      return info.toBuilder();
    }
  }

  @Test
  public void testExtraActionInfoAffectsMnemonic() throws Exception {
    ExtraActionInfo infoOne = ExtraActionInfo.newBuilder()
        .setExtension(
            JavaCompileInfo.javaCompileInfo,
            JavaCompileInfo.newBuilder().addSourceFile("one").build())
        .build();

    ExtraActionInfo infoTwo = ExtraActionInfo.newBuilder()
        .setExtension(
            JavaCompileInfo.javaCompileInfo,
            JavaCompileInfo.newBuilder().addSourceFile("two").build())
        .build();

    Path execRoot = scratch.getFileSystem().getPath("/");
    ArtifactRoot root = ArtifactRoot.asDerivedRoot(execRoot, RootType.OUTPUT, "out");
    Artifact output = ActionsTestUtil.createArtifact(root, scratch.file("/out/test.out"));
    Action actionOne = new ExtraActionInfoFileWriteAction(ActionsTestUtil.NULL_ACTION_OWNER, output,
        new SpecifiedInfoAction(infoOne));
    Action actionTwo = new ExtraActionInfoFileWriteAction(ActionsTestUtil.NULL_ACTION_OWNER, output,
        new SpecifiedInfoAction(infoTwo));

    assertThat(actionOne.getKey(actionKeyContext, /* inputMetadataProvider= */ null))
        .isNotEqualTo(actionTwo.getKey(actionKeyContext, /* inputMetadataProvider= */ null));
  }

  /**
   * Regression test. The Spawn created for extra actions needs to pass the environment of the extra
   * action by getting the result of SpawnAction.getEnvironment() method instead of relying on the
   * default field value of BaseSpawn.environment.
   */
  @Test
  public void testEnvironmentPassedOnOverwrite() throws Exception {
    Path execRoot = scratch.getFileSystem().getPath("/");
    ArtifactRoot out = ArtifactRoot.asDerivedRoot(execRoot, RootType.OUTPUT, "out");
    ExtraAction extraAction =
        new ExtraAction(
            NULL_ACTION_OWNER,
            NestedSetBuilder.emptySet(Order.STABLE_ORDER),
            ImmutableSet.of(
                (Artifact.DerivedArtifact)
                    ActionsTestUtil.createArtifact(out, scratch.file("/out/test.out"))),
            new NullAction(),
            false,
            CommandLine.of(ImmutableList.of("one", "two", "thee")),
            ActionEnvironment.create(ImmutableMap.of("TEST", "TEST_VALUE")),
            ImmutableMap.of(),
            "Executing extra action bla bla",
            "bla bla");

    final Map<String, String> spawnEnvironment = new HashMap<>();
    SpawnStrategy fakeSpawnStrategy =
        new SpawnStrategy() {
          @Override
          public ImmutableList<SpawnResult> exec(
              Spawn spawn, ActionExecutionContext actionExecutionContext) {
            spawnEnvironment.putAll(spawn.getEnvironment());
            return ImmutableList.of();
          }

          @Override
          public boolean canExec(
              Spawn spawn, ActionContext.ActionContextRegistry actionContextRegistry) {
            return true;
          }
        };

    BlazeExecutor testExecutor =
        new TestExecutorBuilder(fileSystem, execRoot)
            .addStrategy(fakeSpawnStrategy, "fake")
            .setDefaultStrategies("fake")
            .build();

    ActionResult actionResult =
        extraAction.execute(
            new ActionExecutionContext(
                testExecutor,
                new FakeActionInputFileCache(),
                ActionInputPrefetcher.NONE,
                actionKeyContext,
                /* outputMetadataStore= */ null,
                /* rewindingEnabled= */ false,
                LostInputsCheck.NONE,
                /* fileOutErr= */ null,
                /* eventHandler= */ null,
                /* clientEnv= */ ImmutableMap.of(),
                /* actionFileSystem= */ null,
                DiscoveredModulesPruner.DEFAULT,
                SyscallCache.NO_CACHE,
                ThreadStateReceiver.NULL_INSTANCE));
    assertThat(actionResult.spawnResults()).isEmpty();
    assertThat(spawnEnvironment.get("TEST")).isNotNull();
    assertThat(spawnEnvironment).containsEntry("TEST", "TEST_VALUE");
  }

  @Test
  public void testUpdateInputsNotPassedToShadowedAction() throws Exception {
    Path execRoot = scratch.getFileSystem().getPath("/");
    ArtifactRoot out = ArtifactRoot.asDerivedRoot(execRoot, RootType.OUTPUT, "out");
    ArtifactRoot src = ArtifactRoot.asSourceRoot(Root.fromPath(scratch.dir("/src")));
    Artifact extraIn = ActionsTestUtil.createArtifact(src, scratch.file("/src/extra.in"));
    Artifact discoveredIn = ActionsTestUtil.createArtifact(src, scratch.file("/src/discovered.in"));
    Action shadowedAction = mock(Action.class);
    when(shadowedAction.discoversInputs()).thenReturn(true);
    when(shadowedAction.getInputs()).thenReturn(NestedSetBuilder.emptySet(Order.STABLE_ORDER));
    when(shadowedAction.inputsKnown()).thenReturn(true);
    ExtraAction extraAction =
        new ExtraAction(
            NULL_ACTION_OWNER,
            NestedSetBuilder.create(Order.STABLE_ORDER, extraIn),
            ImmutableSet.of(
                (Artifact.DerivedArtifact)
                    ActionsTestUtil.createArtifact(out, scratch.file("/out/test.out"))),
            shadowedAction,
            false,
            CommandLine.of(ImmutableList.of()),
            ActionEnvironment.EMPTY,
            ImmutableMap.of(),
            "Executing extra action bla bla",
            "bla bla");
    extraAction.updateInputs(NestedSetBuilder.create(Order.STABLE_ORDER, extraIn, discoveredIn));
    verify(shadowedAction, Mockito.never()).updateInputs(ArgumentMatchers.any());
  }

  @Test
  public void testSerializationRoundTrip_resetsInputs() throws Exception {
    Path execRoot = scratch.getFileSystem().getPath("/");
    ArtifactRoot out = ArtifactRoot.asDerivedRoot(execRoot, RootType.OUTPUT, "out");
    ArtifactRoot src = ArtifactRoot.asSourceRoot(Root.fromPath(scratch.dir("/src")));
    Artifact extraInput = ActionsTestUtil.createArtifact(src, scratch.file("/src/extra.in"));
    Artifact discoveredInput =
        ActionsTestUtil.createArtifact(src, scratch.file("/src/discovered.in"));
    var output =
        (Artifact.DerivedArtifact)
            ActionsTestUtil.createArtifact(out, scratch.file("/out/test.out"));
    output.setGeneratingActionKey(ActionsTestUtil.NULL_ACTION_LOOKUP_DATA);
    // Note that this differs from NULL_ACTION_OWNER in that it has non-empty execProperties, which
    // are important for testing.
    var dummyActionOwner =
        ActionOwner.createDummy(
            NULL_LABEL,
            new Location("dummy-file", 0, 0),
            /* targetKind= */ "dummy-kind",
            /* buildConfigurationMnemonic= */ "dummy-configuration-mnemonic",
            /* configurationChecksum= */ "dummy-configuration",
            new BuildConfigurationEvent(
                BuildEventStreamProtos.BuildEventId.getDefaultInstance(),
                BuildEventStreamProtos.BuildEvent.getDefaultInstance()),
            /* isToolConfiguration= */ false,
            /* executionPlatform= */ PlatformInfo.EMPTY_PLATFORM_INFO,
            /* aspectDescriptors= */ ImmutableList.of(),
            /* execProperties= */ ImmutableMap.of("property1", "value1", "property2", "value2"));
    ExtraAction extraAction =
        new ExtraAction(
            dummyActionOwner,
            NestedSetBuilder.create(Order.STABLE_ORDER, extraInput),
            ImmutableSet.of(output),
            /* shadowedAction= */ new InputDiscoveringNullAction(),
            /* createDummyOutput= */ false,
            CommandLine.of(ImmutableList.of()),
            ActionEnvironment.EMPTY,
            /* executionInfo= */ ImmutableMap.of("xyz", "2", "abc", "1"),
            "Executing extra action bla bla",
            "bla bla");
    ensureMemoizedIsInitializedIsSet(extraAction);
    String originalStructure = dumpStructureWithEquivalenceReduction(extraAction);

    extraAction.updateInputs(
        NestedSetBuilder.create(Order.STABLE_ORDER, extraInput, discoveredInput));

    new SerializationTester(extraAction)
        .makeMemoizingAndAllowFutureBlocking(/* allowFutureBlocking= */ true)
        .addCodec(ArrayCodec.forComponentType(Artifact.class))
        .setVerificationFunction(
            (unusedInput, deserialized) ->
                assertThat(dumpStructureWithEquivalenceReduction(deserialized))
                    .isEqualTo(originalStructure))
        .addDependencies(getCommonSerializationDependencies())
        .addDependencies(SerializationDepsUtils.SERIALIZATION_DEPS_FOR_TEST)
        .runTests();
  }
}
