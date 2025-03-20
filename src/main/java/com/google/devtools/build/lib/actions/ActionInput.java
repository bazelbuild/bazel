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

package com.google.devtools.build.lib.actions;

import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * Represents an input file to a build action, with an appropriate relative path.
 *
 * <p>Artifact is the only notable implementer of the interface, but the interface remains
 * because 1) some Google specific rules ship files that could be Artifacts to remote execution
 * by instantiating ad-hoc derived classes of ActionInput.  2) historically, Google C++ rules
 * allow underspecified C++ builds. For that case, we have extra logic to guess the undeclared
 * header inclusions (eg. computed inclusions). The extra logic lives in a file that is not
 * needed for remote execution, but is a dependency, and it is inserted as a non-Artifact
 * ActionInput.
 *
 * <p>ActionInput is used as a cache "key" for ActionInputFileCache: for Artifacts, the
 * digest/size is already stored in Artifact, but for non-artifacts, we use getExecPathString
 * to find this data in a filesystem related cache.
 */
public interface ActionInput {

  /** @return the relative path to the input file. */
  String getExecPathString();

  /**
   * @return the relative path to the input file.
   */
  PathFragment getExecPath();

  /** The input is a directory. */
  boolean isDirectory();

  /** The input is a symlink that is supposed to stay un-dereferenced. */
  boolean isSymlink();

  /**
   * Returns if this input's file system path includes a digest of its content. See {@link
   * com.google.devtools.build.lib.analysis.config.BuildConfigurationValue#useContentBasedOutputPaths}.
   */
  default boolean contentBasedPath() {
    return false;
  }
}
