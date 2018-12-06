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
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Collection;
import java.util.List;
import javax.annotation.Nullable;

/**
 * To be implemented by actions (such as C++ compilation steps) whose inputs can be scanned to
 * discover other implicit inputs (such as C++ header files).
 *
 * <p>This is useful for remote execution strategies to be able to compute the complete set of files
 * that must be distributed in order to execute such an action.
 */
public interface IncludeScannable {
  /**
   * Returns the built-in list of system include paths for the toolchain compiler. All paths in this
   * list should be relative to the exec directory. They may be absolute if they are also installed
   * on the remote build nodes or for local compilation.
   */
  List<PathFragment> getBuiltInIncludeDirectories();

  /**
   * Returns an immutable list of "-iquote" include paths that should be used by
   * the IncludeScanner for this action. GCC searches these paths first, but
   * only for {@code #include "foo"}, not for {@code #include &lt;foo&gt;}.
   */
  List<PathFragment> getQuoteIncludeDirs();

  /**
   * Returns an immutable list of "-I" include paths that should be used by the
   * IncludeScanner for this action. GCC searches these paths ahead of the
   * system include paths, but after "-iquote" include paths.
   */
  List<PathFragment> getIncludeDirs();

  /**
   * Returns an artifact that the compiler may unconditionally include, even if the source file
   * does not mention it.
   */
  @Nullable
  List<Artifact> getBuiltInIncludeFiles();

  /**
   * Returns the artifact relative to which the {@code getCmdlineIncludes()} should be interpreted.
   */
  Artifact getMainIncludeScannerSource();

  /**
   * Returns an immutable list of sources that the IncludeScanner should scan
   * for this action.
   *
   * <p>Must contain {@code getMainIncludeScannerSource()}.
   */
  Collection<Artifact> getIncludeScannerSources();

  /**
   * Returns explicit header files (i.e., header files explicitly listed) of transitive deps.
   */
  NestedSet<Artifact> getDeclaredIncludeSrcs();

  /** Returns all inputs that are represented using {@link NestedSet}s */
  Collection<NestedSet<Artifact>> getAllNestedSetInputs();

  Iterable<Artifact> getInputsForInvalidation();

  Artifact getSourceFile();

  /**
   * Returns an artifact that is the executable for grepping #include lines from a file.
   */
  Artifact getGrepIncludes();

  /** Returns modules necessary for building and using the output of this action. */
  NestedSet<Artifact> getDiscoveredModules();
}
