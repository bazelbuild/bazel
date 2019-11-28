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

package com.google.devtools.build.lib.rules.cpp;

import com.google.devtools.build.lib.actions.Artifact;

/**
 * Something that appears on the command line of the linker. Since we sometimes expand archive files
 * to their constituent object files, we need to keep information whether a certain file contains
 * embedded objects and if so, the list of the object files themselves.
 */
public interface LinkerInput {

  /**
   * Returns the type of the linker input.
   */
  ArtifactCategory getArtifactCategory();

  /** Returns the artifact that is the input of the linker. */
  Artifact getArtifact();

  /**
   * Returns the original library to link. If this library is a solib symlink, returns the
   * artifact the symlink points to, otherwise, the library itself.
   */
  Artifact getOriginalLibraryArtifact();

  /**
   * Whether the input artifact contains object files or is opaque.
   */
  boolean containsObjectFiles();

  /**
   * Returns whether the input artifact is a fake object file or not.
   */
  boolean isFake();

  /**
   * Return the list of object files included in the input artifact, if there are any. It is
   * legal to call this only when {@link #containsObjectFiles()} returns true.
   */
  Iterable<Artifact> getObjectFiles();

  /**
   * Returns whether we must keep debug symbols for this input.
   */
  boolean isMustKeepDebug();

  /** If true, Bazel will not wrap this input in whole-archive block. */
  boolean disableWholeArchive();
}
