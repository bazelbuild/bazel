// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.buildjar;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Superclass for all JavaBuilder processor classes
 * involved in compiling and processing java code.
 */
public abstract class CommonJavaLibraryProcessor {

  /**
   * Creates the initial set of arguments to javac from the Build
   * configuration supplied. This set of arguments should be extended
   * by the code invoking it.
   *
   * @param build The build request for the initial set of arguments is needed
   * @return The list of initial arguments
   */
  protected List<String> createInitialJavacArgs(JavaLibraryBuildRequest build,
      String classPath) {
    List<String> args = new ArrayList<>();
    if (!classPath.isEmpty()) {
      args.add("-cp");
      args.add(classPath);
    }
    args.add("-d");
    args.add(build.getClassDir());

    // Add an empty source path to prevent javac from sucking in source files
    // from .jar files on the classpath.
    args.add("-sourcepath");
    args.add(File.pathSeparator);

    if (build.getExtdir() != null) {
      args.add("-extdirs");
      args.add(build.getExtdir());
    }

    return args;
  }
}
