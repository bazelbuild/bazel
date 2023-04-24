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

package com.google.devtools.build.lib.generatedprojecttest.util;

import com.google.devtools.build.lib.testutil.BuildRuleBuilder;
import com.google.devtools.build.lib.testutil.Scratch;
import java.io.IOException;

/**
 * A builder that generates whole test projects in a scratch file system.
 */

// TODO(blaze-team): (2012) generate valid parameterized BUILD rules.
// TODO(blaze-team): (2012) generate any required src or data or other files.

public final class TestProjectBuilder {

  // Default workspace name.
  private static final String WORKSPACE = "workspace";

  // The directory name to use for the workspace.
  private final String workspace;
  /** Provides functionality to create and manipulate a scratch file system. */
  private final Scratch scratch;

  /**
   * Creates a builder that will use the default workspace name as the directory.
   */
  public TestProjectBuilder() {
    this(WORKSPACE);
  }

  /**
   * Creates a builder that will use the given workspace name as the directory.
   */
  public TestProjectBuilder(String workspace) {
    this.workspace = workspace;
    this.scratch = new Scratch(String.format("/%s", workspace));
  }

  /**
   * Creates a file in the specified directory with the given content.
   *
   * @param dirName directory to create a new file within
   * @param fileName file Name of the new file (must be unique within the directory)
   * @param generator FileContentsGenerator implementation
   * @throws IOException if the input dirName was not valid, or the file already existed
   */
  public void createFileInDir(String dirName, String fileName, FileContentsGenerator generator)
      throws IOException {
    scratch.file(
        String.format("/%s/%s/%s", workspace, dirName, fileName), generator.getContents());
  }

  /**
   * Returns the {@link Scratch} containing the Test Project that has been built.
   */
  public Scratch getScratch() {
    return this.scratch;
  }

  /** Creates a dummy file with dummy content in the given package with the given name. */
  public void createDummyFileInDir(String pkg, String fileName) throws IOException {
    scratch.file(String.format("%s/%s", pkg, fileName), dummyContentFor(fileName));
  }

  /**
   * Generates the files necessary for the rule. 
   */
  public void createFilesToGenerate(BuildRuleBuilder ruleBuilder) throws IOException {
    for (String file : ruleBuilder.getFilesToGenerate()) {
      scratch.file(file, dummyContentFor(file));
    }
  }

  /** Generates dummy content for a file based on its name and extension. */
  private static String dummyContentFor(String filePath) {
    String fileName = filePath.substring(filePath.lastIndexOf('/') + 1);
    String extension = fileName.substring(fileName.lastIndexOf('.') + 1);
    if (extension.equals("bzl")
        || fileName.equals("BUILD")
        || fileName.equals("BUILD.bazel")
        || fileName.equals("WORKSPACE")
        || fileName.equals("WORKSPACE.bazel")) {
      return "# dummy";
    } else {
      return "dummy";
    }
  }
}
