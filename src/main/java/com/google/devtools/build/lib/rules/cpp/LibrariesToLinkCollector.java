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

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcToolchainVariables.LibraryToLinkValue;
import com.google.devtools.build.lib.rules.cpp.Link.LinkTargetType;
import com.google.devtools.build.lib.rules.cpp.Link.LinkingMode;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.OsPathPolicy;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;

// LINT.IfChange
/** Class that goes over LibrariesToLink and produces {@link LibraryToLinkValue}s */
public class LibrariesToLinkCollector {

  private static final OsPathPolicy OS = OsPathPolicy.getFilePathOs();
  private static final Joiner PATH_JOINER = Joiner.on(PathFragment.SEPARATOR_CHAR);

  private final boolean preferStaticLibs;
  private final boolean isNativeDeps;
  private final PathFragment toolchainLibrariesSolibDir;
  private final CcToolchainProvider ccToolchainProvider;
  private final PathFragment solibDir;
  private final Sequence<LibraryToLink> librariesToLink;
  private final FeatureConfiguration featureConfiguration;
  private final boolean needToolchainLibrariesRpath;
  private final Artifact output;
  private final String workspaceName;
  private final Artifact dynamicLibrarySolibSymlinkOutput;

  public LibrariesToLinkCollector(
      boolean isNativeDeps,
      CcToolchainProvider toolchain,
      PathFragment toolchainLibrariesSolibDir,
      LinkTargetType linkType,
      Link.LinkingMode linkingMode,
      Artifact output,
      PathFragment solibDir,
      FeatureConfiguration featureConfiguration,
      Sequence<LibraryToLink> librariesToLink,
      String workspaceName,
      Artifact dynamicLibrarySolibSymlinkOutput) {
    // When selecting libraries to link, we prefer static or dynamic libraries based on the static
    // or dynamic linking mode.
    // When C++ toolchain doesn't `supports_dynamic_linker`,  toolchain can't produce binaries that
    // load shared libraries at runtime, then we can only link static libraries.

    this.preferStaticLibs =
        linkingMode == LinkingMode.STATIC
            || !featureConfiguration.isEnabled(CppRuleClasses.SUPPORTS_DYNAMIC_LINKER);

    this.isNativeDeps = isNativeDeps;
    this.ccToolchainProvider = toolchain;
    this.toolchainLibrariesSolibDir = toolchainLibrariesSolibDir;
    this.solibDir = solibDir;
    this.featureConfiguration = featureConfiguration;
    this.librariesToLink = librariesToLink;
    this.output = output;
    this.workspaceName = workspaceName;
    this.dynamicLibrarySolibSymlinkOutput = dynamicLibrarySolibSymlinkOutput;

    needToolchainLibrariesRpath =
        toolchainLibrariesSolibDir != null
            && (linkType.isDynamicLibrary()
                || (linkType == LinkTargetType.EXECUTABLE && linkingMode == LinkingMode.DYNAMIC));
  }

  /**
   * Result of {@link LibrariesToLinkCollector#collectLibrariesToLink()}. Provides access to
   * computed sequence of {@link LibraryToLinkValue}s and accompanying library search directories.
   */
  public static class CollectedLibrariesToLink {
    private final NestedSet<String> librarySearchDirectories;
    private final NestedSet<String> runtimeLibrarySearchDirectories;

    private CollectedLibrariesToLink(
        NestedSet<String> librarySearchDirectories,
        NestedSet<String> runtimeLibrarySearchDirectories) {
      this.librarySearchDirectories = librarySearchDirectories;
      this.runtimeLibrarySearchDirectories = runtimeLibrarySearchDirectories;
    }

    public NestedSet<String> getLibrarySearchDirectories() {
      return librarySearchDirectories;
    }

    public NestedSet<String> getRuntimeLibrarySearchDirectories() {
      return runtimeLibrarySearchDirectories;
    }
  }

  private NestedSet<String> collectToolchainRuntimeLibrarySearchDirectories(
      ImmutableList<String> potentialSolibParents) throws EvalException {
    NestedSetBuilder<String> runtimeLibrarySearchDirectories = NestedSetBuilder.linkOrder();
    if (!needToolchainLibrariesRpath) {
      return runtimeLibrarySearchDirectories.build();
    }

    String toolchainLibrariesSolibName = toolchainLibrariesSolibDir.getBaseName();
    if (!(isNativeDeps && ccToolchainProvider.getCppConfiguration().shareNativeDeps())) {
      for (String potentialExecRoot : findToolchainSolibParents(potentialSolibParents)) {
        runtimeLibrarySearchDirectories.add(potentialExecRoot + toolchainLibrariesSolibName + "/");
      }
    }
    if (isNativeDeps) {
      runtimeLibrarySearchDirectories.add("../" + toolchainLibrariesSolibName + "/");
      runtimeLibrarySearchDirectories.add(".");
    }
    runtimeLibrarySearchDirectories.add(toolchainLibrariesSolibName + "/");

    return runtimeLibrarySearchDirectories.build();
  }

  private ImmutableList<String> findPotentialSolibParents() {
    ImmutableList.Builder<String> solibParents = ImmutableList.builder();
    ImmutableList.Builder<Artifact> outputs = ImmutableList.builder();
    outputs.add(output);
    if (dynamicLibrarySolibSymlinkOutput != null) {
      outputs.add(dynamicLibrarySolibSymlinkOutput);
    }
    for (Artifact output : outputs.build()) {
      // The runtime location of the solib directory relative to the binary depends on four factors:
      //
      // * whether the binary is contained in the main repository or an external repository;
      // * whether the binary is executed directly or from a runfiles tree;
      // * whether the binary is staged as a symlink (sandboxed execution; local execution if the
      //   binary is in the runfiles of another target) or a regular file (remote execution) - the
      //   dynamic linker follows sandbox and runfiles symlinks into its location under the
      //   unsandboxed execroot, which thus becomes the effective $ORIGIN;
      // * whether --experimental_sibling_repository_layout is enabled or not.
      //
      // The rpaths emitted into the binary thus have to cover the following cases (assuming that
      // the binary target is located in the pkg `pkg` and has name `file`) for the directory used
      // as $ORIGIN by the dynamic linker and the directory containing the solib directories:
      //
      // 1. main, direct, symlink:
      //    $ORIGIN:    $EXECROOT/pkg
      //    solib root: $EXECROOT
      // 2. main, direct, regular file:
      //    $ORIGIN:    $EXECROOT/pkg
      //    solib root: $EXECROOT/pkg/file.runfiles/main_repo
      // 3. main, runfiles, symlink:
      //    $ORIGIN:    $EXECROOT/pkg
      //    solib root: $EXECROOT
      // 4. main, runfiles, regular file:
      //    $ORIGIN:    other_target.runfiles/main_repo/pkg
      //    solib root: other_target.runfiles/main_repo
      // 5a. external, direct, symlink:
      //    $ORIGIN:    $EXECROOT/external/other_repo/pkg
      //    solib root: $EXECROOT
      // 5b. external, direct, symlink, with --experimental_sibling_repository_layout:
      //    $ORIGIN:    $EXECROOT/../other_repo/pkg
      //    solib root: $EXECROOT/../other_repo
      // 6a. external, direct, regular file:
      //    $ORIGIN:    $EXECROOT/external/other_repo/pkg
      //    solib root: $EXECROOT/external/other_repo/pkg/file.runfiles/main_repo
      // 6b. external, direct, regular file, with --experimental_sibling_repository_layout:
      //    $ORIGIN:    $EXECROOT/../other_repo/pkg
      //    solib root: $EXECROOT/../other_repo/pkg/file.runfiles/other_repo
      // 7a. external, runfiles, symlink:
      //    $ORIGIN:    $EXECROOT/external/other_repo/pkg
      //    solib root: $EXECROOT
      // 7b. external, runfiles, symlink, with --experimental_sibling_repository_layout:
      //    $ORIGIN:    $EXECROOT/../other_repo/pkg
      //    solib root: $EXECROOT/../other_repo
      // 8a. external, runfiles, regular file:
      //    $ORIGIN:    other_target.runfiles/some_repo/pkg
      //    solib root: other_target.runfiles/main_repo
      // 8b. external, runfiles, regular file, with --experimental_sibling_repository_layout:
      //    $ORIGIN:    other_target.runfiles/some_repo/pkg
      //    solib root: other_target.runfiles/some_repo
      //
      // Cases 1, 3, 4, 5, 7, and 8b are covered by an rpath that walks up the root relative path.
      // Cases 2 and 6 covered by walking into file.runfiles/main_repo.
      // Case 8a is covered by walking up some_repo/pkg and then into main_repo.
      boolean isExternal =
          output.getRunfilesPath().startsWith(LabelConstants.EXTERNAL_RUNFILES_PATH_PREFIX);
      boolean usesLegacyRepositoryLayout = output.getRoot().isLegacy();
      // Handles cases 1, 3, 4, 5, and 7.
      solibParents.add("../".repeat(output.getRootRelativePath().segmentCount() - 1));
      // Handle cases 2 and 6.
      String solibRepositoryName;
      if (isExternal && !usesLegacyRepositoryLayout) {
        // Case 6b
        solibRepositoryName = output.getRunfilesPath().getSegment(1);
      } else {
        // Cases 2 and 6a
        solibRepositoryName = workspaceName;
      }
      solibParents.add(output.getFilename() + ".runfiles/" + solibRepositoryName + "/");
      if (isExternal && usesLegacyRepositoryLayout) {
        // Handles case 8a. The runfiles path is of the form ../some_repo/pkg/file and we need to
        // walk up some_repo/pkg and then down into main_repo.
        solibParents.add(
            "../".repeat(output.getRunfilesPath().segmentCount() - 2) + workspaceName + "/");
      }
    }

    return solibParents.build();
  }

  private ImmutableList<String> findToolchainSolibParents(
      ImmutableList<String> potentialSolibParents) throws EvalException {
    boolean usesLegacyRepositoryLayout = output.getRoot().isLegacy();
    // When -experimental_sibling_repository_layout is not enabled, the toolchain solib sits next to
    // the solib_<cpu> directory - so that it shares the same parents.
    if (usesLegacyRepositoryLayout) {
      return potentialSolibParents;
    }

    // When -experimental_sibling_repository_layout is enabled, the toolchain solib is located in
    // these 2 places:
    // 1. The `bin` directory of the repository where the toolchain target is declared (this is the
    // parent directory of `toolchainLibrariesSolibDir`).
    // 2. In `target.runfiles/<toolchain repo>`
    //
    // And the following factors affect what $ORIGIN is resolved to:
    // * whether the binary is contained in the main repository or an external repository;
    // * whether the binary is executed directly or from a runfiles tree;
    // * whether the binary is staged as a symlink (sandboxed execution; local execution if the
    //   binary is in the runfiles of another target) or a regular file (remote execution) - the
    //   dynamic linker follows sandbox and runfiles symlinks into its location under the
    //   unsandboxed execroot, which thus becomes the effective $ORIGIN;
    //
    // The rpaths emitted into the binary thus have to cover the following cases (assuming that
    // the binary target is located in the pkg `pkg` and has name `file`) for the directory used
    // as $ORIGIN by the dynamic linker and the directory containing the solib directories:
    // 1. main, direct, symlink:
    //    $ORIGIN:    $EXECROOT/pkg
    //    solib root: <toolchain repo bin>
    // 2. main, direct, regular file:
    //    $ORIGIN:    $EXECROOT/pkg
    //    solib root: $EXECROOT/pkg/file.runfiles/<toolchain repo>
    // 3. main, runfiles, symlink:
    //    $ORIGIN:    $EXECROOT/pkg
    //    solib root: <toolchain repo bin>
    // 4. main, runfiles, regular file:
    //    $ORIGIN:    other_target.runfiles/main_repo/pkg
    //    solib root: other_target.runfiles/<toolchain repo>
    // 5. external, direct, symlink:
    //    $ORIGIN:    $EXECROOT/../other_repo/pkg
    //    solib root: <toolchain repo bin>
    // 6. external, direct, regular file:
    //    $ORIGIN:    $EXECROOT/../other_repo/pkg
    //    solib root: $EXECROOT/../other_repo/pkg/file.runfiles/<toolchain repo>
    // 7. external, runfiles, symlink:
    //    $ORIGIN:    $EXECROOT/../other_repo/pkg
    //    solib root: <toolchain repo bin>
    // 8. external, runfiles, regular file:
    //    $ORIGIN:    other_target.runfiles/some_repo/pkg
    //    solib root: other_target.runfiles/<toolchain repo>
    //
    // For cases 1, 3, 5, 7, we need to compute the relative path from the output artifact to
    // toolchain repo's bin directory. For 2 and 6, we walk down into `file.runfiles/<toolchain
    // repo>`. For 4 and 8, we need to compute the relative path from the output runfile to
    // <toolchain repo> under runfiles.
    ImmutableList.Builder<String> solibParents = ImmutableList.builder();

    // Cases 1, 3, 5, 7
    PathFragment toolchainBinExecPath = toolchainLibrariesSolibDir.getParentDirectory();
    PathFragment binaryOriginExecPath = output.getExecPath().getParentDirectory();
    solibParents.add(
        getRelative(binaryOriginExecPath, toolchainBinExecPath).getPathString()
            + PathFragment.SEPARATOR_CHAR);

    // Cases 2 and 6
    String toolchainRunfilesRepoName =
        getRunfilesRepoName(ccToolchainProvider.getCcToolchainLabel().getRepository());
    solibParents.add(
        PATH_JOINER.join(output.getFilename() + ".runfiles", toolchainRunfilesRepoName)
            + PathFragment.SEPARATOR_CHAR);

    // Cases 4 and 8
    String binaryRepoName = getRunfilesRepoName(output.getOwnerLabel().getRepository());
    PathFragment toolchainBinRunfilesPath = PathFragment.create(toolchainRunfilesRepoName);
    PathFragment binaryOriginRunfilesPath =
        PathFragment.create(binaryRepoName)
            .getRelative(output.getRepositoryRelativePath())
            .getParentDirectory();
    solibParents.add(
        getRelative(binaryOriginRunfilesPath, toolchainBinRunfilesPath).getPathString()
            + PathFragment.SEPARATOR_CHAR);

    return solibParents.build();
  }

  private String getRunfilesRepoName(RepositoryName repo) {
    if (repo.isMain()) {
      return workspaceName;
    }
    return repo.getName();
  }

  /**
   * Returns the relative {@link PathFragment} from "from" to "to".
   *
   * <p>Example 1: <code>
   * getRelative({@link PathFragment}.create("foo"), {@link PathFragment}.create("foo/bar/wiz"))
   * </code> returns <code>"bar/wiz"</code>.
   *
   * <p>Example 2: <code>
   * getRelative({@link PathFragment}.create("foo/bar/wiz"),
   * {@link PathFragment}.create("foo/wiz"))
   * </code> returns <code>"../../wiz"</code>.
   *
   * <p>The following requirements / assumptions are made: 1) paths must be both relative; 2) they
   * are assumed to be relative to the same location; 3) when the {@code from} path starts with
   * {@code ..} prefixes, the prefix length must not exceed {@code ..} prefixes of the {@code to}
   * path.
   */
  static PathFragment getRelative(PathFragment from, PathFragment to) {
    if (from.isAbsolute() || to.isAbsolute()) {
      throw new IllegalArgumentException("Paths must be both relative.");
    }

    final ImmutableList<String> fromSegments = from.splitToListOfSegments();
    final ImmutableList<String> toSegments = to.splitToListOfSegments();
    final int fromSegCount = fromSegments.size();
    final int toSegCount = toSegments.size();

    int commonSegCount = 0;
    while (commonSegCount < fromSegCount
        && commonSegCount < toSegCount
        && OS.equals(fromSegments.get(commonSegCount), toSegments.get(commonSegCount))) {
      commonSegCount++;
    }

    if (commonSegCount < fromSegCount && fromSegments.get(commonSegCount).equals("..")) {
      throw new IllegalArgumentException(
          "Unable to compute relative path from \""
              + from.getPathString()
              + "\" to \""
              + to.getPathString()
              + "\": too many leading \"..\" segments in from path.");
    }
    PathFragment relativePath =
        PathFragment.create(
            PATH_JOINER.join(Collections.nCopies(fromSegCount - commonSegCount, "..")));
    if (commonSegCount < toSegCount) {
      relativePath = relativePath.getRelative(to.subFragment(commonSegCount, toSegCount));
    }
    return relativePath;
  }

  /**
   * When linking a shared library fully or mostly static then we need to link in *all* dependent
   * files, not just what the shared library needs for its own code. This is done by wrapping all
   * objects/libraries with -Wl,-whole-archive and -Wl,-no-whole-archive. For this case the
   * globalNeedWholeArchive parameter must be set to true. Otherwise only library objects (.lo) need
   * to be wrapped with -Wl,-whole-archive and -Wl,-no-whole-archive.
   *
   * <p>TODO: Factor out of the bazel binary into build variables for crosstool action_configs.
   */
  public CollectedLibrariesToLink collectLibrariesToLink() throws EvalException {
    NestedSetBuilder<String> librarySearchDirectories = NestedSetBuilder.linkOrder();
    ImmutableSet.Builder<String> rpathRootsForExplicitSoDeps = ImmutableSet.builder();

    ImmutableList<String> potentialSolibParents;
    ImmutableList<String> rpathRoots;
    // Calculate the correct relative value for the "-rpath" link option (which sets
    // the search path for finding shared libraries).
    String solibDirPathString = ccToolchainProvider.getSolibDirectory();
    if (isNativeDeps && ccToolchainProvider.getCppConfiguration().shareNativeDeps()) {
      // For shared native libraries, special symlinking is applied to ensure C++
      // toolchain libraries are available under $ORIGIN/_solib_[arch]. So we set the RPATH to find
      // them.
      //
      // Note that we have to do this because $ORIGIN points to different paths for
      // different targets. In other words, blaze-bin/d1/d2/d3/a_shareddeps.so and
      // blaze-bin/d4/b_shareddeps.so have different path depths. The first could
      // reference a standard blaze-bin/_solib_[arch] via $ORIGIN/../../../_solib[arch],
      // and the second could use $ORIGIN/../_solib_[arch]. But since this is a shared
      // artifact, both are symlinks to the same place, so
      // there's no *one* RPATH setting that fits all targets involved in the sharing.
      potentialSolibParents = ImmutableList.of();
      rpathRoots = ImmutableList.of(solibDirPathString + "/");
    } else {
      potentialSolibParents = findPotentialSolibParents();
      rpathRoots =
          potentialSolibParents.stream()
              .map((execRoot) -> execRoot + solibDirPathString + "/")
              .collect(toImmutableList());
    }

    Pair<Boolean, Boolean> includeSolibsPair =
        addLinkerInputs(rpathRoots, librarySearchDirectories, rpathRootsForExplicitSoDeps);
    boolean includeSolibDir = includeSolibsPair.first;
    boolean includeToolchainLibrariesSolibDir = includeSolibsPair.second;

    NestedSetBuilder<String> allRuntimeLibrarySearchDirectories = NestedSetBuilder.linkOrder();
    // rpath ordering matters for performance; first add the one where most libraries are found.
    if (includeSolibDir) {
      for (String rpathRoot : rpathRoots) {
        allRuntimeLibrarySearchDirectories.add(rpathRoot);
      }
    }
    allRuntimeLibrarySearchDirectories.addAll(rpathRootsForExplicitSoDeps.build());
    if (includeToolchainLibrariesSolibDir) {
      allRuntimeLibrarySearchDirectories.addTransitive(
          collectToolchainRuntimeLibrarySearchDirectories(potentialSolibParents));
    }

    return new CollectedLibrariesToLink(
        librarySearchDirectories.build(),
        allRuntimeLibrarySearchDirectories.build());
  }

  private Pair<Boolean, Boolean> addLinkerInputs(
      ImmutableList<String> rpathRoots,
      NestedSetBuilder<String> librarySearchDirectories,
      ImmutableSet.Builder<String> rpathRootsForExplicitSoDeps)
      throws EvalException {
    boolean includeSolibDir = false;
    boolean includeToolchainLibrariesSolibDir = false;
    Map<String, PathFragment> linkedLibrariesPaths = new HashMap<>();

    for (LibraryToLink lib : this.librariesToLink) {
      boolean staticLib =
          (preferStaticLibs
                  && (lib.getStaticLibrary() != null || lib.getPicStaticLibrary() != null))
              || (lib.getInterfaceLibrary() == null && lib.getDynamicLibrary() == null);
      if (!staticLib) {
        final Artifact inputArtifact;
        Artifact originalArtifact;
        if (lib.getInterfaceLibrary() != null) {
          inputArtifact = lib.getInterfaceLibrary();
          originalArtifact = lib.getResolvedSymlinkInterfaceLibrary();
        } else {
          inputArtifact = lib.getDynamicLibrary();
          originalArtifact = lib.getResolvedSymlinkDynamicLibrary();
        }
        if (originalArtifact == null) {
          originalArtifact = inputArtifact;
        }
        PathFragment originalLibDir = originalArtifact.getExecPath().getParentDirectory();
        Preconditions.checkNotNull(originalLibDir);
        String libraryIdentifier = lib.getLibraryIdentifier();
        PathFragment previousLibDir = linkedLibrariesPaths.get(libraryIdentifier);

        if (previousLibDir == null) {
          linkedLibrariesPaths.put(libraryIdentifier, originalLibDir);
        } else if (!previousLibDir.equals(originalLibDir)) {
          throw Starlark.errorf(
              "You are trying to link the same dynamic library %s built in a different"
                  + " configuration. Previously registered instance had path %s, current one"
                  + " has path %s",
              libraryIdentifier, previousLibDir, originalLibDir);
        }

        PathFragment libDir = inputArtifact.getExecPath().getParentDirectory();

        // When COPY_DYNAMIC_LIBRARIES_TO_BINARY is enabled, dynamic libraries are not symlinked
        // under solibDir, so don't check it and don't include solibDir.
        if (!featureConfiguration.isEnabled(CppRuleClasses.COPY_DYNAMIC_LIBRARIES_TO_BINARY)) {
          // The first fragment is bazel-out, and the second may contain a configuration mnemonic.
          // We should always add the default solib dir because that's where libraries will be found
          // e.g., in remote execution, so we ignore the first two fragments.
          if (libDir.subFragment(2).equals(solibDir.subFragment(2))) {
            includeSolibDir = true;
          }
          if (libDir.equals(toolchainLibrariesSolibDir)) {
            includeToolchainLibrariesSolibDir = true;
          }
        }
        addDynamicInputLinkOptions(
            inputArtifact, librarySearchDirectories, rpathRoots, rpathRootsForExplicitSoDeps);
      }
    }
    return Pair.of(includeSolibDir, includeToolchainLibrariesSolibDir);
  }

  /** Adds command-line options for a dynamic library input file into options and libOpts. */
  private void addDynamicInputLinkOptions(
      Artifact inputArtifact,
      NestedSetBuilder<String> librarySearchDirectories,
      ImmutableList<String> rpathRoots,
      ImmutableSet.Builder<String> rpathRootsForExplicitSoDeps)
      throws EvalException {
    if (featureConfiguration.isEnabled(CppRuleClasses.TARGETS_WINDOWS)
        && CcToolchainProvider.supportsInterfaceSharedLibraries(featureConfiguration)) {
      // On Windows, dynamic library (dll) cannot be linked directly when using toolchains that
      // support interface library (eg. MSVC). If the user is doing so, it is only to be referenced
      // in other places (such as copy_dynamic_libraries_to_binary); skip adding it.
      if (CppFileTypes.SHARED_LIBRARY.matches(inputArtifact.getFilename())) {
        return;
      }
    }

    PathFragment libDir = inputArtifact.getExecPath().getParentDirectory();
    if (!libDir.equals(solibDir)
        && (toolchainLibrariesSolibDir == null || !toolchainLibrariesSolibDir.equals(libDir))) {
      String dotdots = "";
      PathFragment commonParent = solibDir;
      while (!libDir.startsWith(commonParent)) {
        dotdots += "../";
        commonParent = commonParent.getParentDirectory();
      }

      // When all dynamic deps are built in transitioned configurations, the default solib dir is
      // not created. While resolving paths, the dynamic linker stops at the first directory that
      // does not exist, even when followed by "../". We thus have to normalize the relative path.
      for (String rpathRoot : rpathRoots) {
        String relativePathToRoot =
            rpathRoot + dotdots + libDir.relativeTo(commonParent).getPathString();
        String normalizedPathToRoot = PathFragment.create(relativePathToRoot).getPathString();
        rpathRootsForExplicitSoDeps.add(normalizedPathToRoot);
      }

      // Unless running locally, libraries will be available under the root relative path, so we
      // should add that to the rpath as well.
      if (inputArtifact.getRootRelativePathString().startsWith("_solib_")) {
        PathFragment artifactPathUnderSolib = inputArtifact.getRootRelativePath().subFragment(1);
        for (String rpathRoot : rpathRoots) {
          rpathRootsForExplicitSoDeps.add(
              rpathRoot + artifactPathUnderSolib.getParentDirectory().getPathString());
        }
      }
    }

    librarySearchDirectories.add(inputArtifact.getExecPath().getParentDirectory().getPathString());


  }
}
// LINT.ThenChange(//src/main/starlark/builtins_bzl/common/cc/link/libraries_to_link_collector.bzl)
