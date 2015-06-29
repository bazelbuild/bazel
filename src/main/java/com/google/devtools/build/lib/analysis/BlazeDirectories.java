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

package com.google.devtools.build.lib.analysis;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.Constants;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.util.StringCanonicalizer;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

import javax.annotation.Nullable;

/**
 * Encapsulation of all of the interesting top-level directories in any Blaze application.
 *
 * <p>The <code>installBase</code> is the directory where the Blaze binary has been installed.The
 * <code>workspace</code> is the top-level directory in the user's client (possibly read-only).The
 * <code>outputBase</code> is the directory below which Blaze puts all its state. The
 * <code>execRoot</code> is the working directory for all spawned tools, which is generally below
 * <code>outputBase</code>.
 *
 * <p>There is a 1:1 correspondence between a running Blaze instance and an output base directory;
 * however, multiple Blaze instances may compile code that's in the same workspace, even on the same
 * machine. If the user does not qualify an output base directory, the startup code will derive it
 * deterministically from the workspace. Note also that while the Blaze server process runs with the
 * workspace directory as its working directory, the client process may have a different working
 * directory, typically a subdirectory.
 *
 * <p>Do not put shortcuts to specific files here!
 */
@Immutable
public final class BlazeDirectories {

  // Output directory name, relative to the execRoot.
  // TODO(bazel-team): (2011) make this private?
  public static final String RELATIVE_OUTPUT_PATH = StringCanonicalizer.intern(
      Constants.PRODUCT_NAME + "-out");

  // Include directory name, relative to execRoot/blaze-out/configuration.
  public static final String RELATIVE_INCLUDE_DIR = StringCanonicalizer.intern("include");

  private final Path installBase;  // Where Blaze gets unpacked
  private final Path workspace;    // Workspace root and server CWD
  private final Path outputBase;   // The root of the temp and output trees
  private final Path execRoot;     // the root of all build actions

  // These two are kept to avoid creating new objects every time they are accessed. This showed up
  // in a profiler.
  private final Path outputPath;
  private final Path localOutputPath;

  public BlazeDirectories(Path installBase, Path outputBase, @Nullable Path workspace) {
    this.installBase = installBase;
    this.workspace = workspace;
    this.outputBase = outputBase;
    if (this.workspace == null) {
      // TODO(bazel-team): this should be null, but at the moment there is a lot of code that
      // depends on it being non-null.
      this.execRoot = outputBase.getChild("default-exec-root");
    } else {
      this.execRoot = outputBase.getChild(workspace.getBaseName());
    }
    this.outputPath = execRoot.getRelative(RELATIVE_OUTPUT_PATH);
    Preconditions.checkState(this.workspace == null || outputPath.asFragment().equals(
        outputPathFromOutputBase(outputBase.asFragment(), workspace.asFragment())));
    this.localOutputPath = outputBase.getRelative(BlazeDirectories.RELATIVE_OUTPUT_PATH);
  }

  /**
   * Returns the Filesystem that all of our directories belong to. Handy for
   * resolving absolute paths.
   */
  public FileSystem getFileSystem() {
    return installBase.getFileSystem();
  }

  /**
   * Returns the installation base directory. Currently used by info command only.
   */
  public Path getInstallBase() {
    return installBase;
  }

  /**
   * Returns the workspace directory, which is also the working dir of the server.
   */
  public Path getWorkspace() {
    return workspace;
  }

  /**
   * Returns if the workspace directory is a valid workspace.
   */
  public boolean inWorkspace() {
    return this.workspace != null;
  }

  /**
   * Returns the base of the output tree, which hosts all build and scratch
   * output for a user and workspace.
   */
  public Path getOutputBase() {
    return outputBase;
  }

  /**
   * Returns the execution root. This is the directory underneath which Blaze builds the source
   * symlink forest, to represent the merged view of different workspaces specified
   * with --package_path.
   */
  public Path getExecRoot() {
    return execRoot;
  }

  /**
   * Returns the output path used by this Blaze instance.
   */
  public Path getOutputPath() {
    return outputPath;
  }

  /**
   * @param outputBase the outputBase as a path fragment.
   * @param workspace the workspace as a path fragment.
   * @return the outputPath as a path fragment, given the outputBase.
   */
  public static PathFragment outputPathFromOutputBase(
      PathFragment outputBase, PathFragment workspace) {
    if (workspace.equals(PathFragment.EMPTY_FRAGMENT)) {
      return outputBase;
    }
    return outputBase.getRelative(workspace.getBaseName() + "/" + RELATIVE_OUTPUT_PATH);
  }

  /**
   * Returns the local output path used by this Blaze instance.
   */
  public Path getLocalOutputPath() {
    return localOutputPath;
  }

  /**
   * Returns the directory where the stdout/stderr for actions can be stored
   * temporarily for a build. If the directory already exists, the directory
   * is cleaned.
   */
  public Path getActionConsoleOutputDirectory() {
    return getOutputBase().getRelative("action_outs");
  }

  /**
   * Returns the installed embedded binaries directory, under the shared
   * installBase location.
   */
  public Path getEmbeddedBinariesRoot() {
    return installBase.getChild("_embedded_binaries");
  }

  /**
   * Returns the configuration-independent root where the build-data should be placed, given the
   * {@link BlazeDirectories} of this server instance. Nothing else should be placed here.
   */
  public Root getBuildDataDirectory() {
    return Root.asDerivedRoot(getExecRoot(), getOutputPath());
  }
}
