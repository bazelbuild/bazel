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

import com.google.common.collect.ImmutableCollection;
import com.google.devtools.build.lib.actions.Artifact;
import javax.annotation.Nullable;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.StarlarkValue;

/**
 * Something that appears on the command line of the linker. Since we sometimes expand archive files
 * to their constituent object files, we need to keep information whether a certain file contains
 * embedded objects and if so, the list of the object files themselves.
 *
 * <p>This is exposed to Starlark via StalarkValue only for the internal C++ code. It should never
 * find its way to public interfaces.
 *
 * @deprecated Will be removed with the Starlarkification of C++ code.
 */
@Deprecated
@StarlarkBuiltin(name = "LegacyLinkerInput", documented = false)
public interface LegacyLinkerInput extends StarlarkValue {

  /** Returns the type of the linker input. */
  ArtifactCategory getArtifactCategory();

  @StarlarkMethod(name = "artifact_category", structField = true, documented = false)
  default String getArtifactCategoryForStarlark() {
    return getArtifactCategory().toString();
  }

  /** Returns the artifact that is the input of the linker. */
  @StarlarkMethod(name = "file", structField = true, documented = false)
  public Artifact getArtifact();

  /**
   * Returns the original library to link. If this library is a solib symlink, returns the artifact
   * the symlink points to, otherwise, the library itself.
   */
  @StarlarkMethod(name = "original_file", structField = true, documented = false)
  public Artifact getOriginalLibraryArtifact();

  /** Whether the input artifact contains object files or is opaque. */
  boolean containsObjectFiles();

  @StarlarkMethod(name = "is_linkstamp", structField = true, documented = false)
  default boolean isLinkstamp() {
    return false;
  }

  /**
   * Return the list of object files included in the input artifact, if there are any. It is legal
   * to call this only when {@link #containsObjectFiles()} returns true.
   */
  @StarlarkMethod(
      name = "object_files",
      structField = true,
      documented = false,
      allowReturnNones = true)
  @Nullable
  ImmutableCollection<Artifact> getObjectFiles();

  /** Returns whether we must keep debug symbols for this input. */
  @StarlarkMethod(name = "must_keep_debug", structField = true, documented = false)
  boolean isMustKeepDebug();

  /** If true, Bazel will not wrap this input in whole-archive block. */
  @StarlarkMethod(name = "disable_whole_archive", structField = true, documented = false)
  boolean disableWholeArchive();

  /**
   * Return the identifier for the library. This is used for de-duplication of linker inputs: two
   * libraries should have the same identifier iff they are in fact the same library but linked in a
   * different way (e.g. static/dynamic, PIC/no-PIC)
   */
  @StarlarkMethod(name = "library_identifier", structField = true, documented = false)
  String getLibraryIdentifier();

  @Override
  default boolean isImmutable() {
    return true;
  }
}
