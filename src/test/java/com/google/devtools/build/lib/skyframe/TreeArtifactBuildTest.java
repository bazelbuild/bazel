// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.actions.ActionInputHelper.treeFileArtifact;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.collect.Collections2;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.hash.Hashing;
import com.google.common.util.concurrent.Runnables;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.ActionLookupValue;
import com.google.devtools.build.lib.actions.ActionResult;
import com.google.devtools.build.lib.actions.Actions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifactType;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.actions.MetadataProvider;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.actions.cache.MetadataHandler;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.actions.util.TestAction;
import com.google.devtools.build.lib.actions.util.TestAction.DummyAction;
import com.google.devtools.build.lib.analysis.actions.SpawnActionTemplate;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.skyframe.ActionTemplateExpansionValue.ActionTemplateExpansionKey;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import javax.annotation.Nullable;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Timestamp builder tests for TreeArtifacts. */
@RunWith(JUnit4.class)
public class TreeArtifactBuildTest extends TimestampBuilderTestCase {

  private static final Predicate<Event> IS_ERROR_EVENT = new Predicate<Event>() {
    @Override
    public boolean apply(Event event) {
      return event.getKind().equals(EventKind.ERROR);
    }
  };

  // Common Artifacts, TreeFileArtifact, and Buttons. These aren't all used in all tests, but
  // they're used often enough that we can save ourselves a lot of copy-pasted code by creating them
  // in setUp().

  Artifact in;

  SpecialArtifact outOne;
  TreeFileArtifact outOneFileOne;
  TreeFileArtifact outOneFileTwo;
  Button buttonOne = new Button();

  SpecialArtifact outTwo;
  TreeFileArtifact outTwoFileOne;
  TreeFileArtifact outTwoFileTwo;
  Button buttonTwo = new Button();
  @Before
  public void setUp() throws Exception {
    in = createSourceArtifact("input");
    writeFile(in, "input_content");

    outOne = createTreeArtifact("outputOne");
    outOneFileOne =
        ActionsTestUtil.createTreeFileArtifactWithNoGeneratingAction(outOne, "out_one_file_one");
    outOneFileTwo =
        ActionsTestUtil.createTreeFileArtifactWithNoGeneratingAction(outOne, "out_one_file_two");

    outTwo = createTreeArtifact("outputTwo");
    outTwoFileOne =
        ActionsTestUtil.createTreeFileArtifactWithNoGeneratingAction(outTwo, "out_one_file_one");
    outTwoFileTwo =
        ActionsTestUtil.createTreeFileArtifactWithNoGeneratingAction(outTwo, "out_one_file_two");
  }

  @Test
  public void testCodec() throws Exception {
    SpecialArtifact parent = createTreeArtifact("parent");
    parent.setGeneratingActionKey(ActionLookupData.create(ACTION_LOOKUP_KEY, 0));
    new SerializationTester(parent, ActionInputHelper.treeFileArtifact(parent, "child"))
        .addDependency(FileSystem.class, scratch.getFileSystem())
        .runTests();
  }

  /** Simple smoke test. If this isn't passing, something is very wrong... */
  @Test
  public void testTreeArtifactSimpleCase() throws Exception {
    TouchingTestAction action = new TouchingTestAction(outOneFileOne, outOneFileTwo);
    registerAction(action);
    buildArtifact(action.getSoleOutput());

    assertThat(outOneFileOne.getPath().exists()).isTrue();
    assertThat(outOneFileTwo.getPath().exists()).isTrue();
  }

  /** Simple test for the case with dependencies. */
  @Test
  public void testDependentTreeArtifacts() throws Exception {
    TouchingTestAction actionOne = new TouchingTestAction(outOneFileOne, outOneFileTwo);
    registerAction(actionOne);

    CopyTreeAction actionTwo = new CopyTreeAction(
        ImmutableList.of(outOneFileOne, outOneFileTwo),
        ImmutableList.of(outTwoFileOne, outTwoFileTwo));
    registerAction(actionTwo);

    buildArtifact(outTwo);

    assertThat(outOneFileOne.getPath().exists()).isTrue();
    assertThat(outOneFileTwo.getPath().exists()).isTrue();
    assertThat(outTwoFileOne.getPath().exists()).isTrue();
    assertThat(outTwoFileTwo.getPath().exists()).isTrue();
  }

  @Test
  public void testInputTreeArtifactMetadataProvider() throws Exception {
    TouchingTestAction actionOne = new TouchingTestAction(outOneFileOne, outOneFileTwo);
    registerAction(actionOne);

    final Artifact normalOutput = createDerivedArtifact("normal/out");
    Action testAction =
        new TestAction(
            TestAction.NO_EFFECT,
            NestedSetBuilder.create(Order.STABLE_ORDER, outOne),
            ImmutableSet.of(normalOutput)) {
          @Override
          public ActionResult execute(ActionExecutionContext actionExecutionContext) {
            try {
              // Check the metadata provider for input TreeFileArtifacts.
              MetadataProvider metadataProvider = actionExecutionContext.getMetadataProvider();
              assertThat(metadataProvider.getMetadata(outOneFileOne).getType().isFile()).isTrue();
              assertThat(metadataProvider.getMetadata(outOneFileTwo).getType().isFile()).isTrue();

              // Touch the action output.
              touchFile(normalOutput);
            } catch (Exception e) {
              throw new RuntimeException(e);
            }
            return ActionResult.EMPTY;
          }
        };

    registerAction(testAction);
    buildArtifact(normalOutput);
  }

  /** Unchanged TreeArtifact outputs should not cause reexecution. */
  @Test
  public void testCacheCheckingForTreeArtifactsDoesNotCauseReexecution() throws Exception {
    SpecialArtifact outOne = createTreeArtifact("outputOne");
    Button buttonOne = new Button();

    SpecialArtifact outTwo = createTreeArtifact("outputTwo");
    Button buttonTwo = new Button();

    TouchingTestAction actionOne = new TouchingTestAction(
        buttonOne, outOne, "file_one", "file_two");
    registerAction(actionOne);

    CopyTreeAction actionTwo = new CopyTreeAction(
        buttonTwo, outOne, outTwo, "file_one", "file_two");
    registerAction(actionTwo);

    buttonOne.pressed = buttonTwo.pressed = false;
    buildArtifact(outTwo);
    assertThat(buttonOne.pressed).isTrue(); // built
    assertThat(buttonTwo.pressed).isTrue(); // built

    buttonOne.pressed = buttonTwo.pressed = false;
    buildArtifact(outTwo);
    assertThat(buttonOne.pressed).isFalse(); // not built
    assertThat(buttonTwo.pressed).isFalse(); // not built
  }

  /**
   * Test rebuilding TreeArtifacts for inputs, outputs, and dependents.
   * Also a test for caching.
   */
  @Test
  public void testTransitiveReexecutionForTreeArtifacts() throws Exception {
    WriteInputToFilesAction actionOne = new WriteInputToFilesAction(
        buttonOne,
        in,
        outOneFileOne, outOneFileTwo);
    registerAction(actionOne);

    CopyTreeAction actionTwo = new CopyTreeAction(
        buttonTwo,
        ImmutableList.of(outOneFileOne, outOneFileTwo),
        ImmutableList.of(outTwoFileOne, outTwoFileTwo));
    registerAction(actionTwo);

    buttonOne.pressed = buttonTwo.pressed = false;
    buildArtifact(outTwo);
    assertThat(buttonOne.pressed).isTrue(); // built
    assertThat(buttonTwo.pressed).isTrue(); // built

    buttonOne.pressed = buttonTwo.pressed = false;
    writeFile(in, "modified_input");
    buildArtifact(outTwo);
    assertThat(buttonOne.pressed).isTrue(); // built
    assertThat(buttonTwo.pressed).isTrue(); // not built

    buttonOne.pressed = buttonTwo.pressed = false;
    writeFile(outOneFileOne, "modified_output");
    buildArtifact(outTwo);
    assertThat(buttonOne.pressed).isTrue(); // built
    assertThat(buttonTwo.pressed).isFalse(); // should have been cached

    buttonOne.pressed = buttonTwo.pressed = false;
    writeFile(outTwoFileOne, "more_modified_output");
    buildArtifact(outTwo);
    assertThat(buttonOne.pressed).isFalse(); // not built
    assertThat(buttonTwo.pressed).isTrue(); // built
  }

  /** Tests that changing a TreeArtifact directory should cause reexeuction. */
  @Test
  public void testDirectoryContentsCachingForTreeArtifacts() throws Exception {
    WriteInputToFilesAction actionOne = new WriteInputToFilesAction(
        buttonOne,
        in,
        outOneFileOne, outOneFileTwo);
    registerAction(actionOne);

    CopyTreeAction actionTwo = new CopyTreeAction(
        buttonTwo,
        ImmutableList.of(outOneFileOne, outOneFileTwo),
        ImmutableList.of(outTwoFileOne, outTwoFileTwo));
    registerAction(actionTwo);

    buttonOne.pressed = buttonTwo.pressed = false;
    buildArtifact(outTwo);
    // just a smoke test--if these aren't built we have bigger problems!
    assertThat(buttonOne.pressed).isTrue();
    assertThat(buttonTwo.pressed).isTrue();

    // Adding a file to a directory should cause reexecution.
    buttonOne.pressed = buttonTwo.pressed = false;
    Path spuriousOutputOne = outOne.getPath().getRelative("spuriousOutput");
    touchFile(spuriousOutputOne);
    buildArtifact(outTwo);
    // Should re-execute, and delete spurious output
    assertThat(spuriousOutputOne.exists()).isFalse();
    assertThat(buttonOne.pressed).isTrue();
    assertThat(buttonTwo.pressed).isFalse(); // should have been cached

    buttonOne.pressed = buttonTwo.pressed = false;
    Path spuriousOutputTwo = outTwo.getPath().getRelative("anotherSpuriousOutput");
    touchFile(spuriousOutputTwo);
    buildArtifact(outTwo);
    assertThat(spuriousOutputTwo.exists()).isFalse();
    assertThat(buttonOne.pressed).isFalse();
    assertThat(buttonTwo.pressed).isTrue();

    // Deleting should cause reexecution.
    buttonOne.pressed = buttonTwo.pressed = false;
    deleteFile(outOneFileOne);
    buildArtifact(outTwo);
    assertThat(outOneFileOne.getPath().exists()).isTrue();
    assertThat(buttonOne.pressed).isTrue();
    assertThat(buttonTwo.pressed).isFalse(); // should have been cached

    buttonOne.pressed = buttonTwo.pressed = false;
    deleteFile(outTwoFileOne);
    buildArtifact(outTwo);
    assertThat(outTwoFileOne.getPath().exists()).isTrue();
    assertThat(buttonOne.pressed).isFalse();
    assertThat(buttonTwo.pressed).isTrue();
  }

  /** TreeArtifacts don't care about mtime, even when the file is empty. */
  @Test
  public void testMTimeForTreeArtifactsDoesNotMatter() throws Exception {
    // For this test, we only touch the input file.
    Artifact in = createSourceArtifact("touchable_input");
    touchFile(in);

    WriteInputToFilesAction actionOne = new WriteInputToFilesAction(
        buttonOne,
        in,
        outOneFileOne, outOneFileTwo);
    registerAction(actionOne);

    CopyTreeAction actionTwo = new CopyTreeAction(
        buttonTwo,
        ImmutableList.of(outOneFileOne, outOneFileTwo),
        ImmutableList.of(outTwoFileOne, outTwoFileTwo));
    registerAction(actionTwo);

    buttonOne.pressed = buttonTwo.pressed = false;
    buildArtifact(outTwo);
    assertThat(buttonOne.pressed).isTrue(); // built
    assertThat(buttonTwo.pressed).isTrue(); // built

    buttonOne.pressed = buttonTwo.pressed = false;
    touchFile(in);
    buildArtifact(outTwo);
    // mtime does not matter.
    assertThat(buttonOne.pressed).isFalse();
    assertThat(buttonTwo.pressed).isFalse();

    // None of the below following should result in anything being built.
    buttonOne.pressed = buttonTwo.pressed = false;
    touchFile(outOneFileOne);
    buildArtifact(outTwo);
    // Nothing should be built.
    assertThat(buttonOne.pressed).isFalse();
    assertThat(buttonTwo.pressed).isFalse();

    buttonOne.pressed = buttonTwo.pressed = false;
    touchFile(outOneFileTwo);
    buildArtifact(outTwo);
    // Nothing should be built.
    assertThat(buttonOne.pressed).isFalse();
    assertThat(buttonTwo.pressed).isFalse();
  }

  /** Tests that the declared order of TreeArtifact contents does not matter. */
  @Test
  public void testOrderIndependenceOfTreeArtifactContents() throws Exception {
    WriteInputToFilesAction actionOne = new WriteInputToFilesAction(
        in,
        // The design of WritingTestAction is s.t.
        // these files will be registered in the given order.
        outOneFileTwo, outOneFileOne);
    registerAction(actionOne);

    CopyTreeAction actionTwo = new CopyTreeAction(
        ImmutableList.of(outOneFileOne, outOneFileTwo),
        ImmutableList.of(outTwoFileOne, outTwoFileTwo));
    registerAction(actionTwo);

    buildArtifact(outTwo);
  }

  @Test
  public void testActionExpansion() throws Exception {
    WriteInputToFilesAction action = new WriteInputToFilesAction(in, outOneFileOne, outOneFileTwo);

    CopyTreeAction actionTwo =
        new CopyTreeAction(
            ImmutableList.of(outOneFileOne, outOneFileTwo),
            ImmutableList.of(outTwoFileOne, outTwoFileTwo)) {
          @Override
          public void executeTestBehavior(ActionExecutionContext actionExecutionContext)
              throws ActionExecutionException {
            super.executeTestBehavior(actionExecutionContext);

            Collection<ActionInput> expanded =
                ActionInputHelper.expandArtifacts(
                    NestedSetBuilder.create(Order.STABLE_ORDER, outOne),
                    actionExecutionContext.getArtifactExpander());
            // Only files registered should show up here.
            assertThat(expanded).containsExactly(outOneFileOne, outOneFileTwo);
          }
        };

    registerAction(action);
    registerAction(actionTwo);

    buildArtifact(outTwo); // should not fail
  }

  @Test
  public void testInvalidOutputRegistrations() throws Exception {
    // Failure expected
    StoredEventHandler storingEventHandler = new StoredEventHandler();
    reporter.removeHandler(failFastHandler);
    reporter.addHandler(storingEventHandler);

    SpecialArtifact outOne = createTreeArtifact("outputOne");
    TreeFileArtifact outOneFileOne =
        ActionInputHelper.treeFileArtifactWithNoGeneratingActionSet(
            outOne, PathFragment.create("out_one_file_one"), ACTION_LOOKUP_KEY);
    TreeFileArtifact outOneFileTwo =
        ActionInputHelper.treeFileArtifactWithNoGeneratingActionSet(
            outOne, PathFragment.create("out_one_file_two"), ACTION_LOOKUP_KEY);
    TreeArtifactTestAction failureOne = new TreeArtifactTestAction(
        Runnables.doNothing(), outOneFileOne, outOneFileTwo) {
      @Override
      public void executeTestBehavior(ActionExecutionContext actionExecutionContext) {
        try {
          writeFile(outOneFileOne, "one");
          writeFile(outOneFileTwo, "two");
          // In this test case, we only register one output. This will fail.
          registerOutput(actionExecutionContext, "one");
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
      }
    };

    registerAction(failureOne);
    outOneFileOne.setGeneratingActionKey(outOne.getGeneratingActionKey());
    outOneFileTwo.setGeneratingActionKey(outOne.getGeneratingActionKey());
    assertThrows(BuildFailedException.class, () -> buildArtifact(outOne));
    // not all outputs were created
    List<Event> errors =
        ImmutableList.copyOf(Iterables.filter(storingEventHandler.getEvents(), IS_ERROR_EVENT));
    assertThat(errors).hasSize(2);
    assertThat(errors.get(0).getMessage()).contains("not present on disk");
    assertThat(errors.get(1).getMessage()).contains("not all outputs were created or valid");

    SpecialArtifact outTwo = createTreeArtifact("outputTwo");
    TreeFileArtifact outTwoFileOne =
        ActionInputHelper.treeFileArtifactWithNoGeneratingActionSet(
            outTwo, PathFragment.create("out_two_file_one"), ACTION_LOOKUP_KEY);
    TreeFileArtifact outTwoFileTwo =
        ActionInputHelper.treeFileArtifactWithNoGeneratingActionSet(
            outTwo, PathFragment.create("out_two_file_two"), ACTION_LOOKUP_KEY);
    TreeArtifactTestAction failureTwo = new TreeArtifactTestAction(
        Runnables.doNothing(), outTwoFileOne, outTwoFileTwo) {
      @Override
      public void executeTestBehavior(ActionExecutionContext actionExecutionContext) {
        try {
          writeFile(outTwoFileOne, "one");
          writeFile(outTwoFileTwo, "two");
          // In this test case, register too many outputs. This will fail.
          registerOutput(actionExecutionContext, "one");
          registerOutput(actionExecutionContext, "two");
          registerOutput(actionExecutionContext, "three");
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
      }
    };

    registerAction(failureTwo);
    outTwoFileOne.setGeneratingActionKey(outTwo.getGeneratingActionKey());
    outTwoFileTwo.setGeneratingActionKey(outTwo.getGeneratingActionKey());
    storingEventHandler.clear();
    assertThrows(BuildFailedException.class, () -> buildArtifact(outTwo));
    errors =
        ImmutableList.copyOf(Iterables.filter(storingEventHandler.getEvents(), IS_ERROR_EVENT));
    assertThat(errors).hasSize(2);
    assertThat(errors.get(0).getMessage()).contains("not present on disk");
    assertThat(errors.get(1).getMessage()).contains("not all outputs were created or valid");
  }

  private static void checkDirectoryPermissions(Path path) throws IOException {
    assertThat(path.isDirectory()).isTrue();
    assertThat(path.isExecutable()).isTrue();
    assertThat(path.isReadable()).isTrue();
    assertThat(path.isWritable()).isFalse();
  }

  private static void checkFilePermissions(Path path) throws IOException {
    assertThat(path.isDirectory()).isFalse();
    assertThat(path.isExecutable()).isTrue();
    assertThat(path.isReadable()).isTrue();
    assertThat(path.isWritable()).isFalse();
  }

  @Test
  public void testOutputsAreReadOnlyAndExecutable() throws Exception {
    final SpecialArtifact out = createTreeArtifact("output");

    TreeArtifactTestAction action =
        new TreeArtifactTestAction(out) {
          @Override
          public ActionResult execute(ActionExecutionContext actionExecutionContext) {
            try {
              writeFile(out.getPath().getChild("one"), "one");
              writeFile(out.getPath().getChild("two"), "two");
              writeFile(out.getPath().getChild("three").getChild("four"), "three/four");
              registerOutput(actionExecutionContext, "one");
              registerOutput(actionExecutionContext, "two");
              registerOutput(actionExecutionContext, "three/four");
            } catch (Exception e) {
              throw new RuntimeException(e);
            }
            return ActionResult.EMPTY;
          }
        };

    registerAction(action);

    buildArtifact(action.getSoleOutput());

    checkDirectoryPermissions(out.getPath());
    checkFilePermissions(out.getPath().getChild("one"));
    checkFilePermissions(out.getPath().getChild("two"));
    checkDirectoryPermissions(out.getPath().getChild("three"));
    checkFilePermissions(out.getPath().getChild("three").getChild("four"));
  }

  @Test
  public void testValidRelativeSymlinkAccepted() throws Exception {
    final SpecialArtifact out = createTreeArtifact("output");

    TreeArtifactTestAction action =
        new TreeArtifactTestAction(out) {
          @Override
          public ActionResult execute(ActionExecutionContext actionExecutionContext) {
            try {
              writeFile(out.getPath().getChild("one"), "one");
              writeFile(out.getPath().getChild("two"), "two");
              FileSystemUtils.ensureSymbolicLink(
                  out.getPath().getChild("links").getChild("link"), "../one");
            } catch (Exception e) {
              throw new RuntimeException(e);
            }
            return ActionResult.EMPTY;
          }
        };

    registerAction(action);

    buildArtifact(action.getSoleOutput());
  }

  @Test
  public void testInvalidSymlinkRejected() throws Exception {
    // Failure expected
    StoredEventHandler storingEventHandler = new StoredEventHandler();
    reporter.removeHandler(failFastHandler);
    reporter.addHandler(storingEventHandler);

    final SpecialArtifact out = createTreeArtifact("output");

    TreeArtifactTestAction action =
        new TreeArtifactTestAction(out) {
          @Override
          public ActionResult execute(ActionExecutionContext actionExecutionContext) {
            try {
              writeFile(out.getPath().getChild("one"), "one");
              writeFile(out.getPath().getChild("two"), "two");
              FileSystemUtils.ensureSymbolicLink(
                  out.getPath().getChild("links").getChild("link"), "../invalid");
            } catch (Exception e) {
              throw new RuntimeException(e);
            }
            return ActionResult.EMPTY;
          }
        };

    registerAction(action);

    assertThrows(BuildFailedException.class, () -> buildArtifact(action.getSoleOutput()));
    List<Event> errors =
        ImmutableList.copyOf(Iterables.filter(storingEventHandler.getEvents(), IS_ERROR_EVENT));
      assertThat(errors).hasSize(2);
      assertThat(errors.get(0).getMessage()).contains(
          "Failed to resolve relative path links/link");
    assertThat(errors.get(1).getMessage()).contains("not all outputs were created or valid");
  }

  @Test
  public void testAbsoluteSymlinkBadTargetRejected() throws Exception {
    // Failure expected
    StoredEventHandler storingEventHandler = new StoredEventHandler();
    reporter.removeHandler(failFastHandler);
    reporter.addHandler(storingEventHandler);

    final SpecialArtifact out = createTreeArtifact("output");

    TreeArtifactTestAction action =
        new TreeArtifactTestAction(out) {
          @Override
          public ActionResult execute(ActionExecutionContext actionExecutionContext) {
            try {
              writeFile(out.getPath().getChild("one"), "one");
              writeFile(out.getPath().getChild("two"), "two");
              FileSystemUtils.ensureSymbolicLink(
                  out.getPath().getChild("links").getChild("link"), "/random/pointer");
            } catch (Exception e) {
              throw new RuntimeException(e);
            }
            return ActionResult.EMPTY;
          }
        };

    registerAction(action);

    assertThrows(BuildFailedException.class, () -> buildArtifact(action.getSoleOutput()));
    List<Event> errors =
        ImmutableList.copyOf(Iterables.filter(storingEventHandler.getEvents(), IS_ERROR_EVENT));
      assertThat(errors).hasSize(2);
      assertThat(errors.get(0).getMessage()).contains("Failed to resolve relative path links/link");
    assertThat(errors.get(1).getMessage()).contains("not all outputs were created or valid");
  }

  @Test
  public void testAbsoluteSymlinkAccepted() throws Exception {
    scratch.overwriteFile("/random/pointer");

    final SpecialArtifact out = createTreeArtifact("output");

    TreeArtifactTestAction action =
        new TreeArtifactTestAction(out) {
          @Override
          public ActionResult execute(ActionExecutionContext actionExecutionContext) {
            try {
              writeFile(out.getPath().getChild("one"), "one");
              writeFile(out.getPath().getChild("two"), "two");
              FileSystemUtils.ensureSymbolicLink(
                  out.getPath().getChild("links").getChild("link"), "/random/pointer");
            } catch (Exception e) {
              throw new RuntimeException(e);
            }
            return ActionResult.EMPTY;
          }
        };

    registerAction(action);

    buildArtifact(action.getSoleOutput());
  }

  @Test
  public void testRelativeSymlinkTraversingOutsideOfTreeArtifactRejected() throws Exception {
    // Failure expected
    StoredEventHandler storingEventHandler = new StoredEventHandler();
    reporter.removeHandler(failFastHandler);
    reporter.addHandler(storingEventHandler);

    final SpecialArtifact out = createTreeArtifact("output");

    TreeArtifactTestAction action =
        new TreeArtifactTestAction(out) {
          @Override
          public ActionResult execute(ActionExecutionContext actionExecutionContext) {
            try {
              writeFile(out.getPath().getChild("one"), "one");
              writeFile(out.getPath().getChild("two"), "two");
              FileSystemUtils.ensureSymbolicLink(
                  out.getPath().getChild("links").getChild("link"), "../../output/random/pointer");
            } catch (Exception e) {
              throw new RuntimeException(e);
            }
            return ActionResult.EMPTY;
          }
        };

    registerAction(action);

    assertThrows(BuildFailedException.class, () -> buildArtifact(action.getSoleOutput()));
    List<Event> errors =
        ImmutableList.copyOf(Iterables.filter(storingEventHandler.getEvents(), IS_ERROR_EVENT));
      assertThat(errors).hasSize(2);
      assertThat(errors.get(0).getMessage()).contains(
          "A TreeArtifact may not contain relative symlinks whose target paths traverse "
          + "outside of the TreeArtifact");
    assertThat(errors.get(1).getMessage()).contains("not all outputs were created or valid");
  }

  @Test
  public void testRelativeSymlinkTraversingToDirOutsideOfTreeArtifactRejected() throws Exception {
    // Failure expected
    StoredEventHandler storingEventHandler = new StoredEventHandler();
    reporter.removeHandler(failFastHandler);
    reporter.addHandler(storingEventHandler);

    final SpecialArtifact out = createTreeArtifact("output");

    // Create a valid directory that can be referenced
    scratch.dir(out.getRoot().getRoot().getRelative("some/dir").getPathString());

    TreeArtifactTestAction action =
        new TreeArtifactTestAction(out) {
          @Override
          public ActionResult execute(ActionExecutionContext actionExecutionContext) {
            try {
              writeFile(out.getPath().getChild("one"), "one");
              writeFile(out.getPath().getChild("two"), "two");
              FileSystemUtils.ensureSymbolicLink(
                  out.getPath().getChild("links").getChild("link"), "../../some/dir");
            } catch (Exception e) {
              throw new RuntimeException(e);
            }
            return ActionResult.EMPTY;
          }
        };

    registerAction(action);

    assertThrows(BuildFailedException.class, () -> buildArtifact(action.getSoleOutput()));
    List<Event> errors =
        ImmutableList.copyOf(Iterables.filter(storingEventHandler.getEvents(), IS_ERROR_EVENT));
      assertThat(errors).hasSize(2);
      assertThat(errors.get(0).getMessage()).contains(
          "A TreeArtifact may not contain relative symlinks whose target paths traverse "
              + "outside of the TreeArtifact");
    assertThat(errors.get(1).getMessage()).contains("not all outputs were created or valid");
  }

  // This is more a smoke test than anything, because it turns out that:
  // 1) there is no easy way to turn fast digests on/off for these test cases, and
  // 2) injectDigest() doesn't really complain if you inject bad digests or digests
  // for nonexistent files. Instead some weird error shows up down the line.
  // In fact, there are no tests for injectDigest anywhere in the codebase.
  // So all we're really testing here is that injectDigest() doesn't throw a weird exception.
  // TODO(bazel-team): write real tests for injectDigest, here and elsewhere.
  @Test
  public void testDigestInjection() throws Exception {
    TreeArtifactTestAction action =
        new TreeArtifactTestAction(outOne) {
          @Override
          public ActionResult execute(ActionExecutionContext actionExecutionContext)
              throws ActionExecutionException {
            try {
              writeFile(outOneFileOne, "one");
              writeFile(outOneFileTwo, "two");

              MetadataHandler md = actionExecutionContext.getMetadataHandler();
              FileStatus stat = outOneFileOne.getPath().stat(Symlinks.NOFOLLOW);
              md.injectDigest(
                  outOneFileOne,
                  stat,
                  Hashing.md5().hashString("one", Charset.forName("UTF-8")).asBytes());

              stat = outOneFileTwo.getPath().stat(Symlinks.NOFOLLOW);
              md.injectDigest(
                  outOneFileTwo,
                  stat,
                  Hashing.md5().hashString("two", Charset.forName("UTF-8")).asBytes());
            } catch (Exception e) {
              throw new RuntimeException(e);
            }
            return ActionResult.EMPTY;
          }
        };

    registerAction(action);
    buildArtifact(action.getSoleOutput());
  }

  @Test
  public void testExpandedActionsBuildInActionTemplate() throws Throwable {
    // artifact1 is a tree artifact generated by a TouchingTestAction.
    SpecialArtifact artifact1 = createTreeArtifact("treeArtifact1");
    TreeFileArtifact treeFileArtifactA =
        ActionInputHelper.treeFileArtifactWithNoGeneratingActionSet(
            artifact1, PathFragment.create("child1"), ACTION_LOOKUP_KEY);
    TreeFileArtifact treeFileArtifactB =
        ActionInputHelper.treeFileArtifactWithNoGeneratingActionSet(
            artifact1, PathFragment.create("child2"), ACTION_LOOKUP_KEY);
    registerAction(new TouchingTestAction(treeFileArtifactA, treeFileArtifactB));
    treeFileArtifactA.setGeneratingActionKey(artifact1.getGeneratingActionKey());
    treeFileArtifactB.setGeneratingActionKey(artifact1.getGeneratingActionKey());

    // artifact2 is a tree artifact generated by an action template.
    SpecialArtifact artifact2 = createTreeArtifact("treeArtifact2");
    SpawnActionTemplate actionTemplate = ActionsTestUtil.createDummySpawnActionTemplate(
        artifact1, artifact2);
    registerAction(actionTemplate);

    // We mock out the action template function to expand into two actions that just touch the
    // output files.
    ActionTemplateExpansionKey secondOwner = ActionTemplateExpansionKey.of(ACTION_LOOKUP_KEY, 1);
    TreeFileArtifact expectedOutputTreeFileArtifact1 =
        ActionInputHelper.treeFileArtifactWithNoGeneratingActionSet(
            artifact2, PathFragment.createAlreadyNormalized("child1"), secondOwner);
    TreeFileArtifact expectedOutputTreeFileArtifact2 =
        ActionInputHelper.treeFileArtifactWithNoGeneratingActionSet(
            artifact2, PathFragment.createAlreadyNormalized("child2"), secondOwner);
    Action generateOutputAction =
        new DummyAction(
            NestedSetBuilder.create(Order.STABLE_ORDER, treeFileArtifactA),
            expectedOutputTreeFileArtifact1);
    Action noGenerateOutputAction =
        new DummyAction(
            NestedSetBuilder.create(Order.STABLE_ORDER, treeFileArtifactB),
            expectedOutputTreeFileArtifact2);

    actionTemplateExpansionFunction =
        new DummyActionTemplateExpansionFunction(
            actionKeyContext, ImmutableList.of(generateOutputAction, noGenerateOutputAction));

    buildArtifact(artifact2);
  }

  @Test
  public void testExpandedActionDoesNotGenerateOutputInActionTemplate() throws Throwable {
    // expect errors
    reporter.removeHandler(failFastHandler);

    // artifact1 is a tree artifact generated by a TouchingTestAction.
    SpecialArtifact artifact1 = createTreeArtifact("treeArtifact1");
    TreeFileArtifact treeFileArtifactA =
        ActionInputHelper.treeFileArtifactWithNoGeneratingActionSet(
            artifact1, PathFragment.create("child1"), ACTION_LOOKUP_KEY);
    TreeFileArtifact treeFileArtifactB =
        ActionInputHelper.treeFileArtifactWithNoGeneratingActionSet(
            artifact1, PathFragment.create("child2"), ACTION_LOOKUP_KEY);
    registerAction(new TouchingTestAction(treeFileArtifactA, treeFileArtifactB));
    treeFileArtifactA.setGeneratingActionKey(artifact1.getGeneratingActionKey());
    treeFileArtifactB.setGeneratingActionKey(artifact1.getGeneratingActionKey());

    // artifact2 is a tree artifact generated by an action template.
    SpecialArtifact artifact2 = createTreeArtifact("treeArtifact2");
    SpawnActionTemplate actionTemplate = ActionsTestUtil.createDummySpawnActionTemplate(
        artifact1, artifact2);
    registerAction(actionTemplate);

    // We mock out the action template function to expand into two actions:
    // One Action that touches the output file.
    // The other action that does not generate the output file.
    ActionTemplateExpansionKey secondOwner = ActionTemplateExpansionKey.of(ACTION_LOOKUP_KEY, 1);
    TreeFileArtifact expectedOutputTreeFileArtifact1 =
        ActionInputHelper.treeFileArtifactWithNoGeneratingActionSet(
            artifact2, PathFragment.createAlreadyNormalized("child1"), secondOwner);
    TreeFileArtifact expectedOutputTreeFileArtifact2 =
        ActionInputHelper.treeFileArtifactWithNoGeneratingActionSet(
            artifact2, PathFragment.createAlreadyNormalized("child2"), secondOwner);
    Action generateOutputAction =
        new DummyAction(
            NestedSetBuilder.create(Order.STABLE_ORDER, treeFileArtifactA),
            expectedOutputTreeFileArtifact1);
    Action noGenerateOutputAction =
        new NoOpDummyAction(
            NestedSetBuilder.create(Order.STABLE_ORDER, treeFileArtifactB),
            ImmutableSet.of(expectedOutputTreeFileArtifact2));

    actionTemplateExpansionFunction =
        new DummyActionTemplateExpansionFunction(
            actionKeyContext, ImmutableList.of(generateOutputAction, noGenerateOutputAction));

    BuildFailedException e =
        assertThrows(BuildFailedException.class, () -> buildArtifact(artifact2));
    assertThat(e).hasMessageThat().contains("not all outputs were created or valid");
  }

  @Test
  public void testOneExpandedActionThrowsInActionTemplate() throws Throwable {
    // expect errors
    reporter.removeHandler(failFastHandler);

    // artifact1 is a tree artifact generated by a TouchingTestAction.
    SpecialArtifact artifact1 = createTreeArtifact("treeArtifact1");
    TreeFileArtifact treeFileArtifactA =
        ActionInputHelper.treeFileArtifactWithNoGeneratingActionSet(
            artifact1, PathFragment.create("child1"), ACTION_LOOKUP_KEY);
    TreeFileArtifact treeFileArtifactB =
        ActionInputHelper.treeFileArtifactWithNoGeneratingActionSet(
            artifact1, PathFragment.create("child2"), ACTION_LOOKUP_KEY);
    registerAction(new TouchingTestAction(treeFileArtifactA, treeFileArtifactB));
    treeFileArtifactA.setGeneratingActionKey(artifact1.getGeneratingActionKey());
    treeFileArtifactB.setGeneratingActionKey(artifact1.getGeneratingActionKey());

    // artifact2 is a tree artifact generated by an action template.
    SpecialArtifact artifact2 = createTreeArtifact("treeArtifact2");
    SpawnActionTemplate actionTemplate = ActionsTestUtil.createDummySpawnActionTemplate(
        artifact1, artifact2);
    registerAction(actionTemplate);

    // We mock out the action template function to expand into two actions:
    // One Action that touches the output file.
    // The other action that just throws when executed.
    TreeFileArtifact expectedOutputTreeFileArtifact1 =
        ActionInputHelper.treeFileArtifactWithNoGeneratingActionSet(
            artifact2,
            PathFragment.createAlreadyNormalized("child1"),
            ActionTemplateExpansionKey.of(artifact1.getArtifactOwner(), 1));
    TreeFileArtifact expectedOutputTreeFileArtifact2 =
        ActionInputHelper.treeFileArtifactWithNoGeneratingActionSet(
            artifact2,
            PathFragment.createAlreadyNormalized("child2"),
            ActionTemplateExpansionKey.of(artifact1.getArtifactOwner(), 1));
    Action generateOutputAction =
        new DummyAction(
            NestedSetBuilder.create(Order.STABLE_ORDER, treeFileArtifactA),
            expectedOutputTreeFileArtifact1);
    Action throwingAction =
        new ThrowingDummyAction(
            NestedSetBuilder.create(Order.STABLE_ORDER, treeFileArtifactB),
            ImmutableSet.of(expectedOutputTreeFileArtifact2));

    actionTemplateExpansionFunction =
        new DummyActionTemplateExpansionFunction(
            actionKeyContext, ImmutableList.of(generateOutputAction, throwingAction));

    BuildFailedException e =
        assertThrows(BuildFailedException.class, () -> buildArtifact(artifact2));
    assertThat(e).hasMessageThat().contains("Throwing dummy action");
  }

  @Test
  public void testAllExpandedActionsThrowInActionTemplate() throws Throwable {
    // expect errors
    reporter.removeHandler(failFastHandler);

    // artifact1 is a tree artifact generated by a TouchingTestAction.
    SpecialArtifact artifact1 = createTreeArtifact("treeArtifact1");
    TreeFileArtifact treeFileArtifactA =
        ActionInputHelper.treeFileArtifactWithNoGeneratingActionSet(
            artifact1, PathFragment.create("child1"), ACTION_LOOKUP_KEY);
    TreeFileArtifact treeFileArtifactB =
        ActionInputHelper.treeFileArtifactWithNoGeneratingActionSet(
            artifact1, PathFragment.create("child2"), ACTION_LOOKUP_KEY);
    registerAction(new TouchingTestAction(treeFileArtifactA, treeFileArtifactB));
    treeFileArtifactA.setGeneratingActionKey(artifact1.getGeneratingActionKey());
    treeFileArtifactB.setGeneratingActionKey(artifact1.getGeneratingActionKey());

    // artifact2 is a tree artifact generated by an action template.
    SpecialArtifact artifact2 = createTreeArtifact("treeArtifact2");
    SpawnActionTemplate actionTemplate = ActionsTestUtil.createDummySpawnActionTemplate(
        artifact1, artifact2);
    registerAction(actionTemplate);

    // We mock out the action template function to expand into two actions that throw when executed.
    ActionTemplateExpansionKey secondOwner = ActionTemplateExpansionKey.of(ACTION_LOOKUP_KEY, 1);
    TreeFileArtifact expectedOutputTreeFileArtifact1 =
        ActionInputHelper.treeFileArtifactWithNoGeneratingActionSet(
            artifact2, PathFragment.createAlreadyNormalized("child1"), secondOwner);
    TreeFileArtifact expectedOutputTreeFileArtifact2 =
        ActionInputHelper.treeFileArtifactWithNoGeneratingActionSet(
            artifact2, PathFragment.createAlreadyNormalized("child2"), secondOwner);
    Action throwingAction =
        new ThrowingDummyAction(
            NestedSetBuilder.create(Order.STABLE_ORDER, treeFileArtifactA),
            ImmutableSet.of(expectedOutputTreeFileArtifact1));
    Action anotherThrowingAction =
        new ThrowingDummyAction(
            NestedSetBuilder.create(Order.STABLE_ORDER, treeFileArtifactB),
            ImmutableSet.of(expectedOutputTreeFileArtifact2));

    actionTemplateExpansionFunction =
        new DummyActionTemplateExpansionFunction(
            actionKeyContext, ImmutableList.of(throwingAction, anotherThrowingAction));

    BuildFailedException e =
        assertThrows(BuildFailedException.class, () -> buildArtifact(artifact2));
    assertThat(e).hasMessageThat().contains("Throwing dummy action");
  }

  @Test
  public void testInputTreeArtifactCreationFailedInActionTemplate() throws Throwable {
    // expect errors
    reporter.removeHandler(failFastHandler);

    // artifact1 is created by a action that throws.
    SpecialArtifact artifact1 = createTreeArtifact("treeArtifact1");
    registerAction(
        new ThrowingDummyAction(
            NestedSetBuilder.emptySet(Order.STABLE_ORDER), ImmutableSet.of(artifact1)));

    // artifact2 is a tree artifact generated by an action template.
    SpecialArtifact artifact2 = createTreeArtifact("treeArtifact2");
    SpawnActionTemplate actionTemplate = ActionsTestUtil.createDummySpawnActionTemplate(
        artifact1, artifact2);
    registerAction(actionTemplate);

    BuildFailedException e =
        assertThrows(BuildFailedException.class, () -> buildArtifact(artifact2));
    assertThat(e).hasMessageThat().contains("Throwing dummy action");
  }

  @Test
  public void testEmptyInputAndOutputTreeArtifactInActionTemplate() throws Throwable {
    // artifact1 is an empty tree artifact which is generated by a single no-op dummy action.
    SpecialArtifact artifact1 = createTreeArtifact("treeArtifact1");
    registerAction(
        new NoOpDummyAction(
            NestedSetBuilder.emptySet(Order.STABLE_ORDER), ImmutableSet.of(artifact1)));

    // artifact2 is a tree artifact generated by an action template that takes artifact1 as input.
    SpecialArtifact artifact2 = createTreeArtifact("treeArtifact2");
    SpawnActionTemplate actionTemplate = ActionsTestUtil.createDummySpawnActionTemplate(
        artifact1, artifact2);
    registerAction(actionTemplate);

    buildArtifact(artifact2);

    assertThat(artifact1.getPath().exists()).isTrue();
    assertThat(artifact1.getPath().getDirectoryEntries()).isEmpty();
    assertThat(artifact2.getPath().exists()).isTrue();
    assertThat(artifact2.getPath().getDirectoryEntries()).isEmpty();
  }

  /**
   * A generic test action that takes at most one input TreeArtifact,
   * exactly one output TreeArtifact, and some path fragment inputs/outputs.
   */
  private abstract static class TreeArtifactTestAction extends TestAction {
    final Iterable<TreeFileArtifact> inputFiles;
    final Iterable<TreeFileArtifact> outputFiles;

    TreeArtifactTestAction(final SpecialArtifact output, final String... subOutputs) {
      this(Runnables.doNothing(),
          null,
          ImmutableList.<TreeFileArtifact>of(),
          output,
          Collections2.transform(
              Arrays.asList(subOutputs),
              new Function<String, TreeFileArtifact>() {
                @Nullable
                @Override
                public TreeFileArtifact apply(String s) {
                  return ActionInputHelper.treeFileArtifact(output, s);
                }
              }));
    }

    TreeArtifactTestAction(Runnable effect, TreeFileArtifact... outputFiles) {
      this(effect, Arrays.asList(outputFiles));
    }

    TreeArtifactTestAction(Runnable effect, Collection<TreeFileArtifact> outputFiles) {
      this(effect, null, ImmutableList.<TreeFileArtifact>of(),
          outputFiles.iterator().next().getParent(), outputFiles);
    }

    TreeArtifactTestAction(Runnable effect, Artifact inputFile,
        Collection<TreeFileArtifact> outputFiles) {
      this(effect, inputFile, ImmutableList.<TreeFileArtifact>of(),
          outputFiles.iterator().next().getParent(), outputFiles);
    }

    TreeArtifactTestAction(Runnable effect, Collection<TreeFileArtifact> inputFiles,
        Collection<TreeFileArtifact> outputFiles) {
      this(effect, inputFiles.iterator().next().getParent(), inputFiles,
          outputFiles.iterator().next().getParent(), outputFiles);
    }

    TreeArtifactTestAction(
        Runnable effect,
        @Nullable Artifact input,
        Collection<TreeFileArtifact> inputFiles,
        Artifact output,
        Collection<TreeFileArtifact> outputFiles) {
      super(
          effect,
          input == null
              ? NestedSetBuilder.emptySet(Order.STABLE_ORDER)
              : NestedSetBuilder.create(Order.STABLE_ORDER, input),
          ImmutableSet.of(output));
      Preconditions.checkArgument(
          inputFiles.isEmpty() || (input != null && input.isTreeArtifact()));
      Preconditions.checkArgument(output == null || output.isTreeArtifact());
      this.inputFiles = ImmutableList.copyOf(inputFiles);
      this.outputFiles = ImmutableList.copyOf(outputFiles);
      for (TreeFileArtifact inputFile : inputFiles) {
        Preconditions.checkState(inputFile.getParent().equals(input));
      }
      for (TreeFileArtifact outputFile : outputFiles) {
        Preconditions.checkState(outputFile.getParent().equals(output));
      }
    }

    @Override
    public ActionResult execute(ActionExecutionContext actionExecutionContext)
        throws ActionExecutionException {
      if (!getInputs().isEmpty()) {
        // Sanity check--verify all inputs exist.
        Artifact input = getInputs().getSingleton();
        if (!input.getPath().exists()) {
          throw new IllegalStateException("action's input Artifact does not exist: "
              + input.getPath());
        }
        for (Artifact inputFile : inputFiles) {
          if (!inputFile.getPath().exists()) {
            throw new IllegalStateException("action's input does not exist: " + inputFile);
          }
        }
      }

      Artifact output = getSoleOutput();
      assertThat(output.getPath().exists()).isTrue();
      try {
        effect.call();
        executeTestBehavior(actionExecutionContext);
        for (TreeFileArtifact outputFile : outputFiles) {
          actionExecutionContext.getMetadataHandler().addExpandedTreeOutput(outputFile);
        }
      } catch (RuntimeException e) {
        throw new RuntimeException(e);
      } catch (Exception e) {
        throw new ActionExecutionException(
            "TestAction failed due to exception: " + e.getMessage(), e, this, false);
      }
      return ActionResult.EMPTY;
    }

    void executeTestBehavior(ActionExecutionContext c) throws ActionExecutionException {
      // Default: do nothing
    }

    /** Checks there's exactly one output, and returns it. */
    SpecialArtifact getSoleOutput() {
      Iterator<Artifact> it = getOutputs().iterator();
      SpecialArtifact r = (SpecialArtifact) it.next();
      Preconditions.checkNotNull(r);
      Preconditions.checkState(!it.hasNext());
      Preconditions.checkState(r.equals(getPrimaryOutput()));
      return r;
    }

    void registerOutput(ActionExecutionContext context, String outputName) throws IOException {
      context.getMetadataHandler().addExpandedTreeOutput(
          treeFileArtifact(getSoleOutput(), PathFragment.create(outputName)));
    }

    static List<TreeFileArtifact> asTreeFileArtifacts(
        final SpecialArtifact parent, String... files) {
      return Lists.transform(
          Arrays.asList(files),
          new Function<String, TreeFileArtifact>() {
            @Nullable
            @Override
            public TreeFileArtifact apply(String s) {
              return ActionInputHelper.treeFileArtifactWithNoGeneratingActionSet(
                  parent, PathFragment.create(s), parent.getArtifactOwner());
            }
          });
    }
  }

  /** An action that touches some output TreeFileArtifacts. Takes no inputs. */
  private static class TouchingTestAction extends TreeArtifactTestAction {
    TouchingTestAction(TreeFileArtifact... outputPaths) {
      super(Runnables.doNothing(), outputPaths);
    }

    TouchingTestAction(Runnable effect, SpecialArtifact output, String... outputPaths) {
      super(effect, asTreeFileArtifacts(output, outputPaths));
    }

    @Override
    public void executeTestBehavior(ActionExecutionContext actionExecutionContext)
        throws ActionExecutionException {
      try {
        for (Artifact file : outputFiles) {
          touchFile(file);
        }
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }
  }

  /** Takes an input file and populates several copies inside a TreeArtifact. */
  private static class WriteInputToFilesAction extends TreeArtifactTestAction {
    WriteInputToFilesAction(Artifact input, TreeFileArtifact... outputs) {
      this(Runnables.doNothing(), input, outputs);
    }

    WriteInputToFilesAction(
        Runnable effect,
        Artifact input,
        TreeFileArtifact... outputs) {
      super(effect, input, Arrays.asList(outputs));
      Preconditions.checkArgument(!input.isTreeArtifact());
    }

    @Override
    public void executeTestBehavior(ActionExecutionContext actionExecutionContext)
        throws ActionExecutionException {
      try {
        for (Artifact file : outputFiles) {
          FileSystemUtils.createDirectoryAndParents(file.getPath().getParentDirectory());
          FileSystemUtils.copyFile(getInputs().getSingleton().getPath(), file.getPath());
        }
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }
  }

  /** Copies the given TreeFileArtifact inputs to the given outputs, in respective order. */
  private static class CopyTreeAction extends TreeArtifactTestAction {

    CopyTreeAction(
        Runnable effect, SpecialArtifact input, SpecialArtifact output, String... sourcesAndDests) {
      super(effect, input, asTreeFileArtifacts(input, sourcesAndDests), output,
          asTreeFileArtifacts(output, sourcesAndDests));
    }

    CopyTreeAction(
        Collection<TreeFileArtifact> inputPaths,
        Collection<TreeFileArtifact> outputPaths) {
      super(Runnables.doNothing(), inputPaths, outputPaths);
    }

    CopyTreeAction(
        Runnable effect,
        Collection<TreeFileArtifact> inputPaths,
        Collection<TreeFileArtifact> outputPaths) {
      super(effect, inputPaths, outputPaths);
    }

    @Override
    public void executeTestBehavior(ActionExecutionContext actionExecutionContext)
        throws ActionExecutionException {
      Iterator<TreeFileArtifact> inputIterator = inputFiles.iterator();
      Iterator<TreeFileArtifact> outputIterator = outputFiles.iterator();

      try {
        while (inputIterator.hasNext() || outputIterator.hasNext()) {
          Artifact input = inputIterator.next();
          Artifact output = outputIterator.next();
          FileSystemUtils.createDirectoryAndParents(output.getPath().getParentDirectory());
          FileSystemUtils.copyFile(input.getPath(), output.getPath());
        }
      } catch (IOException e) {
        throw new RuntimeException(e);
      }

      // both iterators must be of the same size
      assertThat(inputIterator.hasNext()).isFalse();
      assertThat(inputIterator.hasNext()).isFalse();
    }
  }

  private SpecialArtifact createTreeArtifact(String name) {
    FileSystem fs = scratch.getFileSystem();
    Path execRoot =
        fs.getPath(TestUtils.tmpDir()).getRelative("execroot").getRelative("default-exec-root");
    PathFragment execPath = PathFragment.create("out").getRelative(name);
    return new SpecialArtifact(
        ArtifactRoot.asDerivedRoot(execRoot, execRoot.getRelative("out")),
        execPath,
        ACTION_LOOKUP_KEY,
        SpecialArtifactType.TREE);
  }

  private void buildArtifact(Artifact artifact) throws Exception {
    buildArtifacts(cachingBuilder(), artifact);
  }

  private static void writeFile(Path path, String contents) throws IOException {
    FileSystemUtils.createDirectoryAndParents(path.getParentDirectory());
    // sometimes we write read-only files
    if (path.exists()) {
      path.setWritable(true);
    }
    FileSystemUtils.writeContentAsLatin1(path, contents);
  }

  private static void writeFile(Artifact file, String contents) throws IOException {
    writeFile(file.getPath(), contents);
  }

  private static void touchFile(Path path) throws IOException {
    FileSystemUtils.createDirectoryAndParents(path.getParentDirectory());
    path.getParentDirectory().setWritable(true);
    FileSystemUtils.touchFile(path);
  }

  private static void touchFile(Artifact file) throws IOException {
    touchFile(file.getPath());
  }

  private static void deleteFile(Artifact file) throws IOException {
    Path path = file.getPath();
    // sometimes we write read-only files
    if (path.exists()) {
      path.setWritable(true);
      // work around the sticky bit (this might depend on the behavior of the OS?)
      path.getParentDirectory().setWritable(true);
      path.delete();
    }
  }

  /** A dummy action template expansion function that just returns the injected actions */
  private static class DummyActionTemplateExpansionFunction implements SkyFunction {
    private final ActionKeyContext actionKeyContext;
    private final ImmutableList<ActionAnalysisMetadata> actions;

    DummyActionTemplateExpansionFunction(
        ActionKeyContext actionKeyContext, ImmutableList<ActionAnalysisMetadata> actions) {
      this.actionKeyContext = actionKeyContext;
      this.actions = actions;
    }

    @Override
    public SkyValue compute(SkyKey skyKey, Environment env) {
      try {
        return new ActionTemplateExpansionValue(
            Actions.assignOwnersAndFilterSharedActionsAndThrowActionConflict(
                actionKeyContext,
                actions,
                (ActionLookupValue.ActionLookupKey) skyKey,
                /*outputFiles=*/ null));
      } catch (ActionConflictException e) {
        throw new IllegalStateException(e);
      }
    }

    @Override
    public String extractTag(SkyKey skyKey) {
      return null;
    }
  }

  /** No-op action that does not generate the action outputs. */
  private static class NoOpDummyAction extends TestAction {
    public NoOpDummyAction(NestedSet<Artifact> inputs, ImmutableSet<Artifact> outputs) {
      super(NO_EFFECT, inputs, outputs);
    }

    /** Do nothing */
    @Override
    public ActionResult execute(ActionExecutionContext actionExecutionContext)
        throws ActionExecutionException {
      return ActionResult.EMPTY;
    }
  }

  /** No-op action that throws when executed */
  private static class ThrowingDummyAction extends TestAction {
    public ThrowingDummyAction(NestedSet<Artifact> inputs, ImmutableSet<Artifact> outputs) {
      super(NO_EFFECT, inputs, outputs);
    }

    /** Throws */
    @Override
    public ActionResult execute(ActionExecutionContext actionExecutionContext)
        throws ActionExecutionException {
      throw new ActionExecutionException("Throwing dummy action", this, true);
    }
  }
}
