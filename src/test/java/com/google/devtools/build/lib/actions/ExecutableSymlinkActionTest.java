// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.actions;


import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.NULL_ACTION_OWNER;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.util.DummyExecutor;
import com.google.devtools.build.lib.analysis.actions.ExecutableSymlinkAction;
import com.google.devtools.build.lib.exec.SingleBuildFileCache;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.testutil.TestFileOutErr;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class ExecutableSymlinkActionTest {
  private Scratch scratch = new Scratch();
  private Root inputRoot;
  private Root outputRoot;
  TestFileOutErr outErr;
  private Executor executor;

  @Before
  public final void createExecutor() throws Exception  {
    final Path inputDir = scratch.dir("/in");
    inputRoot = Root.asDerivedRoot(inputDir);
    outputRoot = Root.asDerivedRoot(scratch.dir("/out"));
    outErr = new TestFileOutErr();
    executor = new DummyExecutor(inputDir);
  }

  private ActionExecutionContext createContext() {
    Path execRoot = executor.getExecRoot();
    return new ActionExecutionContext(
        executor,
        new SingleBuildFileCache(execRoot.getPathString(), execRoot.getFileSystem()),
        null, outErr, ImmutableMap.of(), null);
  }

  @Test
  public void testSimple() throws Exception {
    Path inputFile = inputRoot.getPath().getChild("some-file");
    Path outputFile = outputRoot.getPath().getChild("some-output");
    FileSystemUtils.createEmptyFile(inputFile);
    inputFile.setExecutable(/*executable=*/true);
    Artifact input = new Artifact(inputFile, inputRoot);
    Artifact output = new Artifact(outputFile, outputRoot);
    ExecutableSymlinkAction action = new ExecutableSymlinkAction(NULL_ACTION_OWNER, input, output);
    action.execute(createContext());
    assertEquals(inputFile, outputFile.resolveSymbolicLinks());
  }

  @Test
  public void testFailIfInputIsNotAFile() throws Exception {
    Path dir = inputRoot.getPath().getChild("some-dir");
    FileSystemUtils.createDirectoryAndParents(dir);
    Artifact input = new Artifact(dir, inputRoot);
    Artifact output = new Artifact(outputRoot.getPath().getChild("some-output"), outputRoot);
    ExecutableSymlinkAction action = new ExecutableSymlinkAction(NULL_ACTION_OWNER, input, output);
    try {
      action.execute(createContext());
      fail();
    } catch (ActionExecutionException e) {
      assertThat(e.getMessage()).contains("'some-dir' is not a file");
    }
  }

  @Test
  public void testFailIfInputIsNotExecutable() throws Exception {
    Path file = inputRoot.getPath().getChild("some-file");
    FileSystemUtils.createEmptyFile(file);
    file.setExecutable(/*executable=*/false);
    Artifact input = new Artifact(file, inputRoot);
    Artifact output = new Artifact(outputRoot.getPath().getChild("some-output"), outputRoot);
    ExecutableSymlinkAction action = new ExecutableSymlinkAction(NULL_ACTION_OWNER, input, output);
    try {
      action.execute(createContext());
      fail();
    } catch (ActionExecutionException e) {
      String want = "'some-file' is not executable";
      String got = e.getMessage();
      assertTrue(String.format("got %s, want %s", got, want), got.contains(want));
    }
  }
}
