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
package com.google.devtools.build.lib.analysis;

import static org.junit.Assert.assertEquals;

import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for {@link BlazeDirectories}.
 */
@RunWith(JUnit4.class)
public class BlazeDirectoriesTest extends FoundationTestCase {

  @Test
  public void testCreatingDirectories() {
    FileSystem fs = scratch.getFileSystem();
    Path installBase = fs.getPath("/my/install");
    Path outputBase = fs.getPath("/my/output");
    Path workspace = fs.getPath("/my/ws");
    BlazeDirectories directories = new BlazeDirectories(installBase, outputBase, workspace,
        TestConstants.PRODUCT_NAME);
    assertEquals(directories.getExecRoot(), outputBase.getChild("ws"));

    workspace = null;
    directories = new BlazeDirectories(installBase, outputBase, workspace,
        TestConstants.PRODUCT_NAME);
    assertEquals(directories.getExecRoot(),
        outputBase.getChild(BlazeDirectories.DEFAULT_EXEC_ROOT));

    workspace = fs.getPath("/");
    directories = new BlazeDirectories(installBase, outputBase, workspace,
        TestConstants.PRODUCT_NAME);
    assertEquals(directories.getExecRoot(),
        outputBase.getChild(BlazeDirectories.DEFAULT_EXEC_ROOT));
  }

}
