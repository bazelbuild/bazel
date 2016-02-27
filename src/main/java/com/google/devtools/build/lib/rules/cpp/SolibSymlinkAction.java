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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Actions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.rules.cpp.LinkerInputs.LibraryToLink;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.io.IOException;

/**
 * Creates mangled symlinks in the solib directory for all shared libraries.
 * Libraries that have a potential to contain SONAME field rely on the mangled
 * symlink to the parent directory instead.
 *
 * Such symlinks are used by the linker to ensure that all rpath entries can be
 * specified relative to the $ORIGIN.
 */
public final class SolibSymlinkAction extends AbstractAction {

  private final Artifact library;
  private final Path target;
  private final Artifact symlink;

  private SolibSymlinkAction(ActionOwner owner, Artifact library, Artifact symlink) {
    super(owner, ImmutableList.of(library), ImmutableList.of(symlink));

    Preconditions.checkArgument(Link.SHARED_LIBRARY_FILETYPES.matches(library.getFilename()));
    this.library = Preconditions.checkNotNull(library);
    this.symlink = Preconditions.checkNotNull(symlink);
    this.target = library.getPath();
  }

  @Override
  protected void deleteOutputs(Path execRoot) throws IOException {
    // Do not delete outputs if action does not intend to do anything.
    if (target != null) {
      super.deleteOutputs(execRoot);
    }
  }

  @Override
  public void execute(
      ActionExecutionContext actionExecutionContext) throws ActionExecutionException {
    Path mangledPath = symlink.getPath();
    try {
      FileSystemUtils.createDirectoryAndParents(mangledPath.getParentDirectory());
      mangledPath.createSymbolicLink(target);
    } catch (IOException e) {
      throw new ActionExecutionException("failed to create _solib symbolic link '"
          + symlink.prettyPrint() + "' to target '" + target + "'", e, this, false);
    }
  }

  @Override
  public Artifact getPrimaryInput() {
    return library;
  }

  @Override
  public Artifact getPrimaryOutput() {
    return symlink;
  }

  @Override
  public ResourceSet estimateResourceConsumption(Executor executor) {
    return ResourceSet.ZERO;
  }

  @Override
  protected String computeKey() {
    Fingerprint f = new Fingerprint();
    f.addPath(symlink.getPath());
    if (target != null) {
      f.addPath(target);
    }
    return f.hexDigestAndReset();
  }

  @Override
  public String getMnemonic() { return "SolibSymlink"; }

  @Override
  protected String getRawProgressMessage() { return null; }

  /**
   * Replaces shared library artifact with mangled symlink and creates related
   * symlink action. For artifacts that should retain filename (e.g. libraries
   * with SONAME tag), link is created to the parent directory instead.
   *
   * This action is performed to minimize number of -rpath entries used during
   * linking process (by essentially "collecting" as many shared libraries as
   * possible in the single directory), since we will be paying quadratic price
   * for each additional entry on the -rpath.
   *
   * @param ruleContext rule context, that requested symlink.
   * @param library Shared library artifact that needs to be mangled.
   * @param preserveName whether to preserve the name of the library
   * @param prefixConsumer whether to prefix the output artifact name with the label of the
   *     consumer
   * @return mangled symlink artifact.
   */
  public static LibraryToLink getDynamicLibrarySymlink(final RuleContext ruleContext,
                                                       final Artifact library,
                                                       boolean preserveName,
                                                       boolean prefixConsumer,
                                                       BuildConfiguration configuration) {
    PathFragment mangledName = getMangledName(
        ruleContext, library.getRootRelativePath(), preserveName, prefixConsumer,
        configuration.getFragment(CppConfiguration.class));
    return getDynamicLibrarySymlinkInternal(ruleContext, library, mangledName, configuration);
  }

   /**
   * Version of {@link #getDynamicLibrarySymlink} for the special case of C++ runtime libraries.
   * These are handled differently than other libraries: neither their names nor directories are
   * mangled, i.e. libstdc++.so.6 is symlinked from _solib_[arch]/libstdc++.so.6
   */
  public static LibraryToLink getCppRuntimeSymlink(RuleContext ruleContext, Artifact library,
      String solibDirOverride, BuildConfiguration configuration) {
    PathFragment solibDir = new PathFragment(solibDirOverride != null
        ? solibDirOverride
        : configuration.getFragment(CppConfiguration.class).getSolibDirectory());
    PathFragment symlinkName = solibDir.getRelative(library.getRootRelativePath().getBaseName());
    return getDynamicLibrarySymlinkInternal(ruleContext, library, symlinkName, configuration);
  }

  /**
   * Internal implementation that takes a pre-determined symlink name; supports both the
   * generic {@link #getDynamicLibrarySymlink} and the specialized {@link #getCppRuntimeSymlink}.
   */
  private static LibraryToLink getDynamicLibrarySymlinkInternal(RuleContext ruleContext,
      Artifact library, PathFragment symlinkName, BuildConfiguration configuration) {
    Preconditions.checkArgument(Link.SHARED_LIBRARY_FILETYPES.matches(library.getFilename()));
    Preconditions.checkArgument(!library.getRootRelativePath().getSegment(0).startsWith("_solib_"));

    // Ignore libraries that are already represented by the symlinks.
    Root root = configuration.getBinDirectory();
    Artifact symlink = ruleContext.getShareableArtifact(symlinkName, root);
    ruleContext.registerAction(
        new SolibSymlinkAction(ruleContext.getActionOwner(), library, symlink));
    return LinkerInputs.solibLibraryToLink(symlink, library);
  }

  /**
   * Returns the name of the symlink that will be created for a library, given
   * its name.
   *
   * @param ruleContext rule context that requests symlink
   * @param libraryPath the root-relative path of the library
   * @param preserveName true if filename should be preserved
   * @param prefixConsumer true if the result should be prefixed with the label of the consumer
   * @returns root relative path name
   */
  public static PathFragment getMangledName(RuleContext ruleContext,
                                            PathFragment libraryPath,
                                            boolean preserveName,
                                            boolean prefixConsumer,
                                            CppConfiguration cppConfiguration) {
    String escapedRulePath = Actions.escapedPath(
        "_" + ruleContext.getLabel());
    String soname = getDynamicLibrarySoname(libraryPath, preserveName);
    PathFragment solibDir = new PathFragment(cppConfiguration.getSolibDirectory());
    if (preserveName) {
      String escapedLibraryPath =
          Actions.escapedPath("_" + libraryPath.getParentDirectory().getPathString());
      PathFragment mangledDir = solibDir.getRelative(prefixConsumer
          ? escapedRulePath + "__" + escapedLibraryPath
          : escapedLibraryPath);
      return mangledDir.getRelative(soname);
    } else {
      return solibDir.getRelative(prefixConsumer
          ? escapedRulePath + "__" + soname
          : soname);
    }
  }

  /**
   * Compute the SONAME to use for a dynamic library. This name is basically the
   * name of the shared library in its final symlinked location.
   *
   * @param libraryPath name of the shared library that needs to be mangled
   * @param preserveName true if filename should be preserved, false - mangled
   * @return soname to embed in the dynamic library
   */
  public static String getDynamicLibrarySoname(PathFragment libraryPath,
                                               boolean preserveName) {
    String mangledName;
    if (preserveName) {
      mangledName = libraryPath.getBaseName();
    } else {
      mangledName = "lib" + Actions.escapedPath(libraryPath.getPathString());
    }
    return mangledName;
  }

  @Override
  public boolean shouldReportPathPrefixConflict(Action action) {
    return false; // Always ignore path prefix conflict for the SolibSymlinkAction.
  }
}
