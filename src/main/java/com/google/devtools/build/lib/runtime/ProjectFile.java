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

package com.google.devtools.build.lib.runtime;

import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.List;

/**
 * A file that describes a project - for large source trees that are worked on by multiple
 * independent teams, it is useful to have a larger unit than a package which combines a set of
 * target patterns and a set of corresponding options.
 */
public interface ProjectFile {

  /**
   * A provider for a project file - we generally expect the provider to cache parsed files
   * internally and return a cached version if it can ascertain that that is still correct.
   *
   * <p>Note in particular that packages may be moved between different package path entries, which
   * should lead to cache invalidation.
   */
  public interface Provider {
    /**
     * Returns an (optionally cached) project file instance. If there is no such file, or if the
     * file cannot be parsed, then it throws an exception.
     */
    ProjectFile getProjectFile(Path workingDirectory, List<Path> packagePath, PathFragment path)
        throws AbruptExitException;
  }

  /**
   * A string name of the project file that is reported to the user. It should be in such a format
   * that passing it back in on the command line works.
   */
  String getName();

  /**
   * A list of strings that are parsed into the options for the command.
   *
   * @param command An action from the command line, e.g. "build" or "test".
   * @throws UnsupportedOperationException if an unknown command is passed.
   */
  List<String> getCommandLineFor(String command);
}
