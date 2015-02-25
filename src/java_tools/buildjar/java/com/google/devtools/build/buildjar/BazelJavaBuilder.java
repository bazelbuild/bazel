// Copyright 2007-2014 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.buildjar;

import java.io.IOException;
import java.util.Arrays;

/**
 * The JavaBuilder main called by bazel.
 */
public abstract class BazelJavaBuilder {

  private static final String CMDNAME = "BazelJavaBuilder";

  /**
   * The main method of the BazelJavaBuilder.
   */
  public static void main(String[] args) {
    try {
      JavaLibraryBuildRequest build = JavaLibraryBuildRequest.parse(Arrays.asList(args));
      AbstractJavaBuilder builder = build.getDependencyModule().reduceClasspath()
          ? new ReducedClasspathJavaLibraryBuilder()
          : new SimpleJavaLibraryBuilder();
      builder.run(build, System.err);
    } catch (IOException | InvalidCommandLineException e) {
      System.err.println(CMDNAME + " threw exception : " + e.getMessage());
      System.exit(1);
    }
  }
}
