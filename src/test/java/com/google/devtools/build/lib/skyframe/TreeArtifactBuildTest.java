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

import static com.google.common.base.Throwables.getRootCause;
import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.actions.ActionInputHelper.artifactFile;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.collect.Collections2;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.common.hash.Hashing;
import com.google.common.util.concurrent.Runnables;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifactType;
import com.google.devtools.build.lib.actions.ArtifactFile;
import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.actions.TestExecException;
import com.google.devtools.build.lib.actions.cache.InjectedStat;
import com.google.devtools.build.lib.actions.cache.MetadataHandler;
import com.google.devtools.build.lib.actions.util.TestAction;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.IOException;
import java.nio.charset.Charset;
import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;

import javax.annotation.Nullable;

/** Timestamp builder tests for TreeArtifacts. */
@RunWith(JUnit4.class)
public class TreeArtifactBuildTest extends TimestampBuilderTestCase {
  // Common Artifacts, ArtifactFiles, and Buttons. These aren't all used in all tests, but they're
  // used often enough that we can save ourselves a lot of copy-pasted code by creating them
  // in setUp().

  Artifact in;

  Artifact outOne;
  ArtifactFile outOneFileOne;
  ArtifactFile outOneFileTwo;
  Button buttonOne = new Button();

  Artifact outTwo;
  ArtifactFile outTwoFileOne;
  ArtifactFile outTwoFileTwo;
  Button buttonTwo = new Button();

  @Before
  public void setUp() throws Exception {
    in = createSourceArtifact("input");
    writeFile(in, "input_content");

    outOne = createTreeArtifact("outputOne");
    outOneFileOne = artifactFile(outOne, "out_one_file_one");
    outOneFileTwo = artifactFile(outOne, "out_one_file_two");

    outTwo = createTreeArtifact("outputTwo");
    outTwoFileOne = artifactFile(outTwo, "out_one_file_one");
    outTwoFileTwo = artifactFile(outTwo, "out_one_file_two");
  }

  /** Simple smoke test. If this isn't passing, something is very wrong... */
  @Test
  public void testTreeArtifactSimpleCase() throws Exception {
    TouchingTestAction action = new TouchingTestAction(outOneFileOne, outOneFileTwo);
    registerAction(action);
    buildArtifact(action.getSoleOutput());

    assertTrue(outOneFileOne.getPath().exists());
    assertTrue(outOneFileTwo.getPath().exists());
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

    assertTrue(outOneFileOne.getPath().exists());
    assertTrue(outOneFileTwo.getPath().exists());
    assertTrue(outTwoFileOne.getPath().exists());
    assertTrue(outTwoFileTwo.getPath().exists());
  }

  /** Unchanged TreeArtifact outputs should not cause reexecution. */
  @Test
  public void testCacheCheckingForTreeArtifactsDoesNotCauseReexecution() throws Exception {
    Artifact outOne = createTreeArtifact("outputOne");
    Button buttonOne = new Button();

    Artifact outTwo = createTreeArtifact("outputTwo");
    Button buttonTwo = new Button();

    TouchingTestAction actionOne = new TouchingTestAction(
        buttonOne, outOne, "file_one", "file_two");
    registerAction(actionOne);

    CopyTreeAction actionTwo = new CopyTreeAction(
        buttonTwo, outOne, outTwo, "file_one", "file_two");
    registerAction(actionTwo);

    buttonOne.pressed = buttonTwo.pressed = false;
    buildArtifact(outTwo);
    assertTrue(buttonOne.pressed); // built
    assertTrue(buttonTwo.pressed); // built

    buttonOne.pressed = buttonTwo.pressed = false;
    buildArtifact(outTwo);
    assertFalse(buttonOne.pressed); // not built
    assertFalse(buttonTwo.pressed); // not built
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
    assertTrue(buttonOne.pressed); // built
    assertTrue(buttonTwo.pressed); // built

    buttonOne.pressed = buttonTwo.pressed = false;
    writeFile(in, "modified_input");
    buildArtifact(outTwo);
    assertTrue(buttonOne.pressed); // built
    assertTrue(buttonTwo.pressed); // not built

    buttonOne.pressed = buttonTwo.pressed = false;
    writeFile(outOneFileOne, "modified_output");
    buildArtifact(outTwo);
    assertTrue(buttonOne.pressed); // built
    assertFalse(buttonTwo.pressed); // should have been cached

    buttonOne.pressed = buttonTwo.pressed = false;
    writeFile(outTwoFileOne, "more_modified_output");
    buildArtifact(outTwo);
    assertFalse(buttonOne.pressed); // not built
    assertTrue(buttonTwo.pressed); // built
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
    assertTrue(buttonOne.pressed);
    assertTrue(buttonTwo.pressed);

    // Adding a file to a directory should cause reexecution.
    buttonOne.pressed = buttonTwo.pressed = false;
    Path spuriousOutputOne = outOne.getPath().getRelative("spuriousOutput");
    touchFile(spuriousOutputOne);
    buildArtifact(outTwo);
    // Should re-execute, and delete spurious output
    assertFalse(spuriousOutputOne.exists());
    assertTrue(buttonOne.pressed);
    assertFalse(buttonTwo.pressed); // should have been cached

    buttonOne.pressed = buttonTwo.pressed = false;
    Path spuriousOutputTwo = outTwo.getPath().getRelative("anotherSpuriousOutput");
    touchFile(spuriousOutputTwo);
    buildArtifact(outTwo);
    assertFalse(spuriousOutputTwo.exists());
    assertFalse(buttonOne.pressed);
    assertTrue(buttonTwo.pressed);

    // Deleting should cause reexecution.
    buttonOne.pressed = buttonTwo.pressed = false;
    deleteFile(outOneFileOne);
    buildArtifact(outTwo);
    assertTrue(outOneFileOne.getPath().exists());
    assertTrue(buttonOne.pressed);
    assertFalse(buttonTwo.pressed); // should have been cached

    buttonOne.pressed = buttonTwo.pressed = false;
    deleteFile(outTwoFileOne);
    buildArtifact(outTwo);
    assertTrue(outTwoFileOne.getPath().exists());
    assertFalse(buttonOne.pressed);
    assertTrue(buttonTwo.pressed);
  }

  /**
   * TreeArtifacts don't care about mtime, even when the file is empty.
   * However, actions taking input non-Tree artifacts still care about mtime
   * (although this behavior should go away).
   */
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
    assertTrue(buttonOne.pressed); // built
    assertTrue(buttonTwo.pressed); // built

    buttonOne.pressed = buttonTwo.pressed = false;
    touchFile(in);
    buildArtifact(outTwo);
    // Per existing behavior, mtime matters for empty file Artifacts.
    assertTrue(buttonOne.pressed);
    // But this should be cached.
    assertFalse(buttonTwo.pressed);

    // None of the below following should result in anything being built.
    buttonOne.pressed = buttonTwo.pressed = false;
    touchFile(outOneFileOne);
    buildArtifact(outTwo);
    // Nothing should be built.
    assertFalse(buttonOne.pressed);
    assertFalse(buttonTwo.pressed);

    buttonOne.pressed = buttonTwo.pressed = false;
    touchFile(outOneFileTwo);
    buildArtifact(outTwo);
    // Nothing should be built.
    assertFalse(buttonOne.pressed);
    assertFalse(buttonTwo.pressed);
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

    CopyTreeAction actionTwo = new CopyTreeAction(
        ImmutableList.of(outOneFileOne, outOneFileTwo),
        ImmutableList.of(outTwoFileOne, outTwoFileTwo)) {
      @Override
      public void executeTestBehavior(ActionExecutionContext actionExecutionContext)
          throws ActionExecutionException {
        super.executeTestBehavior(actionExecutionContext);

        Collection<ActionInput> expanded =
            ActionInputHelper.expandArtifacts(ImmutableList.of(outOne),
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
    TreeArtifactTestAction failureOne = new TreeArtifactTestAction(
        Runnables.doNothing(), outOneFileOne, outOneFileTwo) {
      @Override
      public void executeTestBehavior(ActionExecutionContext actionExecutionContext)
          throws ActionExecutionException {
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
    try {
      buildArtifact(outOne);
      fail(); // Should have thrown
    } catch (Exception e) {
      assertThat(getRootCause(e).getMessage()).contains("not present on disk");
    }

    TreeArtifactTestAction failureTwo = new TreeArtifactTestAction(
        Runnables.doNothing(), outTwoFileOne, outTwoFileTwo) {
      @Override
      public void executeTestBehavior(ActionExecutionContext actionExecutionContext)
          throws ActionExecutionException {
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
    try {
      buildArtifact(outTwo);
      fail(); // Should have thrown
    } catch (Exception e) {
      assertThat(getRootCause(e).getMessage()).contains("not present on disk");
    }
  }

  private static void checkDirectoryPermissions(Path path) throws IOException {
    assertTrue(path.isDirectory());
    assertTrue(path.isExecutable());
    assertTrue(path.isReadable());
    assertFalse(path.isWritable());
  }

  private static void checkFilePermissions(Path path) throws IOException {
    assertFalse(path.isDirectory());
    assertTrue(path.isExecutable());
    assertTrue(path.isReadable());
    assertFalse(path.isWritable());
  }

  @Test
  public void testOutputsAreReadOnlyAndExecutable() throws Exception {
    final Artifact out = createTreeArtifact("output");

    TreeArtifactTestAction action = new TreeArtifactTestAction(out) {
      @Override
      public void execute(ActionExecutionContext actionExecutionContext)
          throws ActionExecutionException {
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

  // This is more a smoke test than anything, because it turns out that:
  // 1) there is no easy way to turn fast digests on/off for these test cases, and
  // 2) injectDigest() doesn't really complain if you inject bad digests or digests
  // for nonexistent files. Instead some weird error shows up down the line.
  // In fact, there are no tests for injectDigest anywhere in the codebase.
  // So all we're really testing here is that injectDigest() doesn't throw a weird exception.
  // TODO(bazel-team): write real tests for injectDigest, here and elsewhere.
  @Test
  public void testDigestInjection() throws Exception {
    TreeArtifactTestAction action = new TreeArtifactTestAction(outOne) {
      @Override
      public void execute(ActionExecutionContext actionExecutionContext)
          throws ActionExecutionException {
        try {
          writeFile(outOneFileOne, "one");
          writeFile(outOneFileTwo, "two");

          MetadataHandler md = actionExecutionContext.getMetadataHandler();
          FileStatus stat = outOneFileOne.getPath().stat(Symlinks.NOFOLLOW);
          md.injectDigest(outOneFileOne,
              new InjectedStat(stat.getLastModifiedTime(), stat.getSize(), stat.getNodeId()),
              Hashing.md5().hashString("one", Charset.forName("UTF-8")).asBytes());

          stat = outOneFileTwo.getPath().stat(Symlinks.NOFOLLOW);
          md.injectDigest(outOneFileTwo,
              new InjectedStat(stat.getLastModifiedTime(), stat.getSize(), stat.getNodeId()),
              Hashing.md5().hashString("two", Charset.forName("UTF-8")).asBytes());
        } catch (Exception e) {
          throw new RuntimeException(e);
        }
      }
    };

    registerAction(action);
    buildArtifact(action.getSoleOutput());
  }

  /**
   * A generic test action that takes at most one input TreeArtifact,
   * exactly one output TreeArtifact, and some path fragment inputs/outputs.
   */
  private abstract static class TreeArtifactTestAction extends TestAction {
    final Iterable<ArtifactFile> inputFiles;
    final Iterable<ArtifactFile> outputFiles;

    TreeArtifactTestAction(final Artifact output, final String... subOutputs) {
      this(Runnables.doNothing(),
          null,
          ImmutableList.<ArtifactFile>of(),
          output,
          Collections2.transform(
              Arrays.asList(subOutputs),
              new Function<String, ArtifactFile>() {
                @Nullable
                @Override
                public ArtifactFile apply(String s) {
                  return ActionInputHelper.artifactFile(output, s);
                }
              }));
    }

    TreeArtifactTestAction(Runnable effect, ArtifactFile... outputFiles) {
      this(effect, Arrays.asList(outputFiles));
    }

    TreeArtifactTestAction(Runnable effect, Collection<ArtifactFile> outputFiles) {
      this(effect, null, ImmutableList.<ArtifactFile>of(),
          outputFiles.iterator().next().getParent(), outputFiles);
    }

    TreeArtifactTestAction(Runnable effect, Artifact inputFile,
        Collection<ArtifactFile> outputFiles) {
      this(effect, inputFile, ImmutableList.<ArtifactFile>of(),
          outputFiles.iterator().next().getParent(), outputFiles);
    }

    TreeArtifactTestAction(Runnable effect, Collection<ArtifactFile> inputFiles,
        Collection<ArtifactFile> outputFiles) {
      this(effect, inputFiles.iterator().next().getParent(), inputFiles,
          outputFiles.iterator().next().getParent(), outputFiles);
    }

    TreeArtifactTestAction(
        Runnable effect,
        @Nullable Artifact input,
        Collection<ArtifactFile> inputFiles,
        Artifact output,
        Collection<ArtifactFile> outputFiles) {
      super(effect,
          input == null ? ImmutableList.<Artifact>of() : ImmutableList.of(input),
          ImmutableList.of(output));
      Preconditions.checkArgument(
          inputFiles.isEmpty() || (input != null && input.isTreeArtifact()));
      Preconditions.checkArgument(output == null || output.isTreeArtifact());
      this.inputFiles = ImmutableList.copyOf(inputFiles);
      this.outputFiles = ImmutableList.copyOf(outputFiles);
      for (ArtifactFile inputFile : inputFiles) {
        Preconditions.checkState(inputFile.getParent().equals(input));
      }
      for (ArtifactFile outputFile : outputFiles) {
        Preconditions.checkState(outputFile.getParent().equals(output));
      }
    }

    @Override
    public void execute(ActionExecutionContext actionExecutionContext)
        throws ActionExecutionException {
      if (getInputs().iterator().hasNext()) {
        // Sanity check--verify all inputs exist.
        Artifact input = getSoleInput();
        if (!input.getPath().exists()) {
          throw new IllegalStateException("action's input Artifact does not exist: "
              + input.getPath());
        }
        for (ArtifactFile inputFile : inputFiles) {
          if (!inputFile.getPath().exists()) {
            throw new IllegalStateException("action's input does not exist: " + inputFile);
          }
        }
      }

      Artifact output = getSoleOutput();
      assertTrue(output.getPath().exists());
      try {
        effect.call();
        executeTestBehavior(actionExecutionContext);
        for (ArtifactFile outputFile : outputFiles) {
          actionExecutionContext.getMetadataHandler().addExpandedTreeOutput(outputFile);
        }
      } catch (RuntimeException e) {
        throw new RuntimeException(e);
      } catch (Exception e) {
        throw new ActionExecutionException("TestAction failed due to exception",
            e, this, false);
      }
    }

    void executeTestBehavior(ActionExecutionContext c) throws ActionExecutionException {
      // Default: do nothing
    }

    /** Checks there's exactly one input, and returns it. */
    // This prevents us from making testing mistakes, like
    // assuming there's only one input when this isn't actually true.
    Artifact getSoleInput() {
      Iterator<Artifact> it = getInputs().iterator();
      Artifact r = it.next();
      Preconditions.checkNotNull(r);
      Preconditions.checkState(!it.hasNext());
      return r;
    }

    /** Checks there's exactly one output, and returns it. */
    Artifact getSoleOutput() {
      Iterator<Artifact> it = getOutputs().iterator();
      Artifact r = it.next();
      Preconditions.checkNotNull(r);
      Preconditions.checkState(!it.hasNext());
      Preconditions.checkState(r.equals(getPrimaryOutput()));
      return r;
    }

    void registerOutput(ActionExecutionContext context, String outputName) throws IOException {
      context.getMetadataHandler().addExpandedTreeOutput(
          artifactFile(getSoleOutput(), new PathFragment(outputName)));
    }

    static List<ArtifactFile> asArtifactFiles(final Artifact parent, String... files) {
      return Lists.transform(
          Arrays.asList(files),
          new Function<String, ArtifactFile>() {
            @Nullable
            @Override
            public ArtifactFile apply(String s) {
              return ActionInputHelper.artifactFile(parent, s);
            }
          });
    }
  }

  /** An action that touches some output ArtifactFiles. Takes no inputs. */
  private static class TouchingTestAction extends TreeArtifactTestAction {
    TouchingTestAction(ArtifactFile... outputPaths) {
      super(Runnables.doNothing(), outputPaths);
    }

    TouchingTestAction(Runnable effect, Artifact output, String... outputPaths) {
      super(effect, asArtifactFiles(output, outputPaths));
    }

    @Override
    public void executeTestBehavior(ActionExecutionContext actionExecutionContext)
        throws ActionExecutionException {
      try {
        for (ArtifactFile file : outputFiles) {
          touchFile(file);
        }
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }
  }

  /** Takes an input file and populates several copies inside a TreeArtifact. */
  private static class WriteInputToFilesAction extends TreeArtifactTestAction {
    WriteInputToFilesAction(Artifact input, ArtifactFile... outputs) {
      this(Runnables.doNothing(), input, outputs);
    }

    WriteInputToFilesAction(
        Runnable effect,
        Artifact input,
        ArtifactFile... outputs) {
      super(effect, input, Arrays.asList(outputs));
      Preconditions.checkArgument(!input.isTreeArtifact());
    }

    @Override
    public void executeTestBehavior(ActionExecutionContext actionExecutionContext)
        throws ActionExecutionException {
      try {
        for (ArtifactFile file : outputFiles) {
          FileSystemUtils.createDirectoryAndParents(file.getPath().getParentDirectory());
          FileSystemUtils.copyFile(getSoleInput().getPath(), file.getPath());
        }
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }
  }

  /** Copies the given ArtifactFile inputs to the given outputs, in respective order. */
  private static class CopyTreeAction extends TreeArtifactTestAction {

    CopyTreeAction(Runnable effect, Artifact input, Artifact output, String... sourcesAndDests) {
      super(effect, input, asArtifactFiles(input, sourcesAndDests), output,
          asArtifactFiles(output, sourcesAndDests));
    }

    CopyTreeAction(
        Collection<ArtifactFile> inputPaths,
        Collection<ArtifactFile> outputPaths) {
      super(Runnables.doNothing(), inputPaths, outputPaths);
    }

    CopyTreeAction(
        Runnable effect,
        Collection<ArtifactFile> inputPaths,
        Collection<ArtifactFile> outputPaths) {
      super(effect, inputPaths, outputPaths);
    }

    @Override
    public void executeTestBehavior(ActionExecutionContext actionExecutionContext)
        throws ActionExecutionException {
      Iterator<ArtifactFile> inputIterator = inputFiles.iterator();
      Iterator<ArtifactFile> outputIterator = outputFiles.iterator();

      try {
        while (inputIterator.hasNext() || outputIterator.hasNext()) {
          ArtifactFile input = inputIterator.next();
          ArtifactFile output = outputIterator.next();
          FileSystemUtils.createDirectoryAndParents(output.getPath().getParentDirectory());
          FileSystemUtils.copyFile(input.getPath(), output.getPath());
        }
      } catch (IOException e) {
        throw new RuntimeException(e);
      }

      // both iterators must be of the same size
      assertFalse(inputIterator.hasNext());
      assertFalse(inputIterator.hasNext());
    }
  }

  private Artifact createTreeArtifact(String name) {
    FileSystem fs = scratch.getFileSystem();
    Path execRoot = fs.getPath(TestUtils.tmpDir());
    PathFragment execPath = new PathFragment("out").getRelative(name);
    Path path = execRoot.getRelative(execPath);
    return new SpecialArtifact(
        path, Root.asDerivedRoot(execRoot, execRoot.getRelative("out")), execPath, ALL_OWNER,
        SpecialArtifactType.TREE);
  }

  private void buildArtifact(Artifact artifact)
      throws InterruptedException, BuildFailedException, TestExecException, AbruptExitException {
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

  private static void writeFile(ArtifactFile file, String contents) throws IOException {
    writeFile(file.getPath(), contents);
  }

  private static void touchFile(Path path) throws IOException {
    FileSystemUtils.createDirectoryAndParents(path.getParentDirectory());
    path.getParentDirectory().setWritable(true);
    FileSystemUtils.touchFile(path);
  }

  private static void touchFile(ArtifactFile file) throws IOException {
    touchFile(file.getPath());
  }

  private static void deleteFile(ArtifactFile file) throws IOException {
    Path path = file.getPath();
    // sometimes we write read-only files
    if (path.exists()) {
      path.setWritable(true);
      // work around the sticky bit (this might depend on the behavior of the OS?)
      path.getParentDirectory().setWritable(true);
      path.delete();
    }
  }
}
