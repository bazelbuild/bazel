// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.rules.cpp.Link.LinkStaticness;
import com.google.devtools.build.lib.rules.cpp.Link.LinkTargetType;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.config.BuildConfiguration;

import javax.annotation.Nullable;

/**
 * Provides information on how to run a linker step. This is used by Link to
 * determine the proper flags to the linker program.
 */
public interface LinkConfiguration {
  /**
   * Returns the BuildConfiguration used for this link.
   */
  BuildConfiguration getConfiguration();

  /**
   * Returns the ActionOwner (likely a RuleConfiguredTarget) that corresponds
   * to this link step.
   */
  ActionOwner getOwner();

  /**
   * Returns the output artifact produced by the linker. Can be null only if
   * {@link #getLinkTargetType} is {@code EXECUTABLE} or {@code
   * DYNAMIC_LIBRARY} and {@link #getLinkStaticness} is not {@code DYNAMIC}.
   * If null, {@link Link#getArgv} will not include the "zeroth" argument
   * (the linker executable) or the output flag ({@code -o blah}), this allows
   * you to include the linker args in another build system (say, GHC's), which
   * will be responsible for the output.
   */
  Artifact getOutput();

  /**
   * Returns an interface shared object output artifact produced
   * during linking.  Can only be non-null if {@link #getOutput} is
   * non-null.  Currently, this only returns non-null if {@link
   * #getLinkTargetType} is {@code DYNAMIC_LIBRARY} and an interface
   * shared object was requested.
   */
  @Nullable Artifact getInterfaceOutput();

  /**
   * Returns an artifact containing the number of symbols used per object file passed to the
   * linker. This is currently a gold only feature, and is only produced for executables. If another
   * target is being linked, or if symbol counts output is disabled, this will be null.
   */
  @Nullable Artifact getSymbolCountsOutput();

  /**
   * Returns the (ordered, immutable) list of header files that contain build
   * info.
   */
  ImmutableList<Artifact> getBuildInfoHeaderArtifacts();

  /**
   * Returns the (ordered, immutable) list of paths to the linker's input files.
   */
  Iterable<? extends LinkerInput> getLinkerInputs();

  /**
   * Returns the runtime inputs to the linker.
   */
  Iterable<? extends LinkerInput> getRuntimeInputs();

  /**
   * Returns the current type of link target set.
   * @see Link.LinkTargetType
   */
  LinkTargetType getLinkTargetType();

  /**
   * Returns the "staticness" of the link.
   * @see Link.LinkStaticness
   */
  LinkStaticness getLinkStaticness();

  /**
   * Returns the additional linker options for this link.
   */
  ImmutableList<String> getLinkopts();

  /**
   * Returns the default settings affecting this link.
   */
  ImmutableSet<String> getFeatures();

  /**
   * Returns a (possibly empty) mapping of (C++ source file, .o output file)
   * pairs for source files that need to be compiled at link time.
   *
   * <p>This is used to embed various values from the build system into binaries
   * to identify their provenance.
   */
  ImmutableMap<Artifact, Artifact> getLinkstamps();

  /**
   * Returns the location of the C++ runtime solib symlinks. If null, the C++ dynamic runtime
   * libraries either do not exist (because they do not come from the depot) or they are in the
   * regular solib directory.
   */
  @Nullable PathFragment getRuntimeSolibDir();

  /**
   * Returns true for libraries linked as native dependencies for other languages.
   */
  boolean isNativeDeps();

  /**
   * Returns true if this link should use $EXEC_ORIGIN as the root for finding shared
   * libraries, false if it should use $ORIGIN. See bug "Please use $EXEC_ORIGIN instead of
   * $ORIGIN when linking cc_tests" for further context.
   */
  boolean useExecOrigin();

  /**
   * Returns the binary that is used to build interface .so files.
   */
  @Nullable Artifact buildInterfaceSo();
}
