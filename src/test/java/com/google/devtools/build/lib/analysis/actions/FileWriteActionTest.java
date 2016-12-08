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
package com.google.devtools.build.lib.analysis.actions;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.NULL_ACTION_OWNER;

import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.util.LazyString;
import java.util.Random;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class FileWriteActionTest extends FileWriteActionTestCase {

  @Override
  protected FileWriteAction createAction(
      ActionOwner actionOwner, Artifact outputArtifact, String data, boolean makeExecutable) {
    return new FileWriteAction(actionOwner, outputArtifact, data, makeExecutable);
  }

  @Test
  public void testNoInputs() {
    checkNoInputsByDefault();
  }

  @Test
  public void testDestinationArtifactIsOutput() {
    checkDestinationArtifactIsOutput();
  }

  @Test
  public void testCanWriteNonExecutableFile() throws Exception {
    checkCanWriteNonExecutableFile();
  }

  @Test
  public void testCanWriteExecutableFile() throws Exception {
    checkCanWriteExecutableFile();
  }

  @Test
  public void testComputesConsistentKeys() throws Exception {
    checkComputesConsistentKeys();
  }

  @Test
  public void testFileWriteActionWithShortString() throws Exception {
    Artifact outputArtifact = getBinArtifactWithNoOwner("destination.txt");
    String contents = "Hello world";
    FileWriteAction action =
        new FileWriteAction(NULL_ACTION_OWNER, outputArtifact, contents, false);
    assertThat(action.getFileContents()).isEqualTo(contents);
  }

  @Test
  public void testFileWriteActionWithLazyString() throws Exception {
    Artifact outputArtifact = getBinArtifactWithNoOwner("destination.txt");
    final String backingString = "Hello world";
    LazyString contents =
        new LazyString() {
          @Override
          public String toString() {
            return backingString;
          }
        };
    FileWriteAction action =
        new FileWriteAction(NULL_ACTION_OWNER, outputArtifact, contents, false);
    assertThat(action.getFileContents()).isEqualTo(backingString);
  }

  /** Exercises the code path that compresses the string */
  @Test
  public void testFileWriteActionWithLongString() throws Exception {
    Artifact outputArtifact = getBinArtifactWithNoOwner("destination.txt");
    StringBuilder sb = new StringBuilder();

    // Fill buffer with (deterministic) random characters to get a string that won't compress
    // to a tiny size
    Random random = new Random(0);
    for (int i = 0; i < 16 * 1024; ++i) {
      char c = (char) random.nextInt(128);
      sb.append(c);
    }
    String contents = sb.toString();
    FileWriteAction action =
        new FileWriteAction(NULL_ACTION_OWNER, outputArtifact, contents, false);
    assertThat(action.getFileContents()).isEqualTo(contents);
  }
}
