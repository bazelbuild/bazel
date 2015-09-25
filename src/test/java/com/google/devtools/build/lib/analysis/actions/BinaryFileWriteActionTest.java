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

import com.google.common.io.ByteSource;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;

import java.nio.charset.StandardCharsets;

public class BinaryFileWriteActionTest extends FileWriteActionTestCase {
  @Override
  protected Action createAction(
      ActionOwner actionOwner, Artifact outputArtifact, String data, boolean makeExecutable) {
    return new BinaryFileWriteAction(actionOwner, outputArtifact,
        ByteSource.wrap(data.getBytes(StandardCharsets.UTF_8)), makeExecutable);
  }

  public void testNoInputs() {
    checkNoInputsByDefault();
  }

  public void testDestinationArtifactIsOutput() {
    checkDestinationArtifactIsOutput();
  }

  public void testCanWriteNonExecutableFile() throws Exception {
    checkCanWriteNonExecutableFile();
  }

  public void testCanWriteExecutableFile() throws Exception {
    checkCanWriteExecutableFile();
  }

  public void testComputesConsistentKeys() throws Exception {
    checkComputesConsistentKeys();
  }
}
