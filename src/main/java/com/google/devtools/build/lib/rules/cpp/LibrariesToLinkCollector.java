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
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleErrorConsumer;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcToolchainVariables.LibraryToLinkValue;
import com.google.devtools.build.lib.rules.cpp.CcToolchainVariables.SequenceBuilder;
import com.google.devtools.build.lib.rules.cpp.Link.LinkTargetType;
import com.google.devtools.build.lib.rules.cpp.Link.LinkingMode;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.OsPathPolicy;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import javax.annotation.Nullable;

/** Class that goes over linker inputs and produces {@link LibraryToLinkValue}s */
public class LibrariesToLinkCollector {

  private static final OsPathPolicy OS = OsPathPolicy.getFilePathOs();
  private static final Joiner PATH_JOINER = Joiner.on(PathFragment.SEPARATOR_CHAR);

  private final boolean isNativeDeps;
  private final PathFragment toolchainLibrariesSolibDir;
  private final CppConfiguration cppConfiguration;
  private final CcToolchainProvider ccToolchainProvider;
  private final boolean isLtoIndexing;
  private final PathFragment solibDir;
  private final Iterable<? extends LinkerInput> linkerInputs;
  private final Iterable<LtoBackendArtifacts> allLtoArtifacts;
  private final boolean allowLtoIndexing;
  private final Artifact thinltoParamFile;
  private final FeatureConfiguration featureConfiguration;
  private final boolean needWholeArchive;
  private final boolean needToolchainLibrariesRpath;
  private final RuleErrorConsumer ruleErrorConsumer;
  private final Artifact output;
  private final String workspaceName;
  private final Artifact dynamicLibrarySolibSymlinkOutput;

  public LibrariesToLinkCollector(
      boolean isNativeDeps,
      CppConfiguration cppConfiguration,
      CcToolchainProvider toolchain,
      PathFragment toolchainLibrariesSolibDir,
      LinkTargetType linkType,
      Link.LinkingMode linkingMode,
      Artifact output,
      PathFragment solibDir,
      boolean isLtoIndexing,
      Iterable<LtoBackendArtifacts> allLtoArtifacts,
      FeatureConfiguration featureConfiguration,
      Artifact thinltoParamFile,
      boolean allowLtoIndexing,
      Iterable<LinkerInput> linkerInputs,
      boolean needWholeArchive,
      RuleErrorConsumer ruleErrorConsumer,
      String workspaceName,
      Artifact dynamicLibrarySolibSymlinkOutput) {
    this.isNativeDeps = isNativeDeps;
    this.cppConfiguration = cppConfiguration;
    this.ccToolchainProvider = toolchain;
    this.toolchainLibrariesSolibDir = toolchainLibrariesSolibDir;
    this.solibDir = solibDir;
    this.isLtoIndexing = isLtoIndexing;
    this.allLtoArtifacts = allLtoArtifacts;
    this.featureConfiguration = featureConfiguration;
    this.thinltoParamFile = thinltoParamFile;
    this.allowLtoIndexing = allowLtoIndexing;
    this.linkerInputs = linkerInputs;
    this.needWholeArchive = needWholeArchive;
    this.ruleErrorConsumer = ruleErrorConsumer;
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
    private final SequenceBuilder librariesToLink;
    private final NestedSet<LinkerInput> expandedLinkerInputs;
    private final NestedSet<String> librarySearchDirectories;
    private final NestedSet<String> runtimeLibrarySearchDirectories;

    private CollectedLibrariesToLink(
        SequenceBuilder librariesToLink,
        NestedSet<LinkerInput> expandedLinkerInputs,
        NestedSet<String> librarySearchDirectories,
        NestedSet<String> runtimeLibrarySearchDirectories) {
      this.librariesToLink = librariesToLink;
      this.expandedLinkerInputs = expandedLinkerInputs;
      this.librarySearchDirectories = librarySearchDirectories;
      this.runtimeLibrarySearchDirectories = runtimeLibrarySearchDirectories;
    }

    public SequenceBuilder getLibrariesToLink() {
      return librariesToLink;
    }

    // TODO(b/78347840): Figure out how to make these Artifacts.
    public NestedSet<LinkerInput> getExpandedLinkerInputs() {
      return expandedLinkerInputs;
    }

    public NestedSet<String> getLibrarySearchDirectories() {
      return librarySearchDirectories;
    }

    public NestedSet<String> getRuntimeLibrarySearchDirectories() {
      return runtimeLibrarySearchDirectories;
    }
  }

  private NestedSet<String> collectToolchainRuntimeLibrarySearchDirectories(
      ImmutableList<String> potentialSolibParents) {
    NestedSetBuilder<String> runtimeLibrarySearchDirectories = NestedSetBuilder.linkOrder();
    if (!needToolchainLibrariesRpath) {
      return runtimeLibrarySearchDirectories.build();
    }

    String toolchainLibrariesSolibName = toolchainLibrariesSolibDir.getBaseName();
    if (!(isNativeDeps && cppConfiguration.shareNativeDeps())) {
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
      ImmutableList<String> potentialSolibParents) {
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
  public CollectedLibrariesToLink collectLibrariesToLink() {
    NestedSetBuilder<String> librarySearchDirectories = NestedSetBuilder.linkOrder();
    ImmutableSet.Builder<String> rpathRootsForExplicitSoDeps = ImmutableSet.builder();
    NestedSetBuilder<LinkerInput> expandedLinkerInputsBuilder = NestedSetBuilder.linkOrder();
    // List of command line parameters that need to be placed *outside* of
    // --whole-archive ... --no-whole-archive.
    SequenceBuilder librariesToLink = new SequenceBuilder();

    ImmutableList<String> potentialSolibParents;
    ImmutableList<String> rpathRoots;
    // Calculate the correct relative value for the "-rpath" link option (which sets
    // the search path for finding shared libraries).
    if (isNativeDeps && cppConfiguration.shareNativeDeps()) {
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
      rpathRoots = ImmutableList.of(ccToolchainProvider.getSolibDirectory() + "/");
    } else {
      potentialSolibParents = findPotentialSolibParents();
      rpathRoots =
          potentialSolibParents.stream()
              .map((execRoot) -> execRoot + ccToolchainProvider.getSolibDirectory() + "/")
              .collect(toImmutableList());
    }

    Map<Artifact, Artifact> ltoMap = generateLtoMap();
    Pair<Boolean, Boolean> includeSolibsPair =
        addLinkerInputs(
            rpathRoots,
            ltoMap,
            librarySearchDirectories,
            rpathRootsForExplicitSoDeps,
            librariesToLink,
            expandedLinkerInputsBuilder);
    boolean includeSolibDir = includeSolibsPair.first;
    boolean includeToolchainLibrariesSolibDir = includeSolibsPair.second;
    Preconditions.checkState(
        ltoMap == null || ltoMap.isEmpty(), "Still have LTO objects left: %s", ltoMap);

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
        librariesToLink,
        expandedLinkerInputsBuilder.build(),
        librarySearchDirectories.build(),
        allRuntimeLibrarySearchDirectories.build());
  }

  private Pair<Boolean, Boolean> addLinkerInputs(
      ImmutableList<String> rpathRoots,
      Map<Artifact, Artifact> ltoMap,
      NestedSetBuilder<String> librarySearchDirectories,
      ImmutableSet.Builder<String> rpathEntries,
      SequenceBuilder librariesToLink,
      NestedSetBuilder<LinkerInput> expandedLinkerInputsBuilder) {
    boolean includeSolibDir = false;
    boolean includeToolchainLibrariesSolibDir = false;
    Map<String, PathFragment> linkedLibrariesPaths = new HashMap<>();

    for (LinkerInput input : linkerInputs) {
      if (input.getArtifactCategory() == ArtifactCategory.DYNAMIC_LIBRARY
          || input.getArtifactCategory() == ArtifactCategory.INTERFACE_LIBRARY) {
        PathFragment originalLibDir =
            input.getOriginalLibraryArtifact().getExecPath().getParentDirectory();
        Preconditions.checkNotNull(originalLibDir);
        String libraryIdentifier = input.getLibraryIdentifier();
        PathFragment previousLibDir = linkedLibrariesPaths.get(libraryIdentifier);

        if (previousLibDir == null) {
          linkedLibrariesPaths.put(libraryIdentifier, originalLibDir);
        } else if (!previousLibDir.equals(originalLibDir)) {
          ruleErrorConsumer.ruleError(
              String.format(
                  "You are trying to link the same dynamic library %s built in a different"
                      + " configuration. Previously registered instance had path %s, current one"
                      + " has path %s",
                  libraryIdentifier, previousLibDir, originalLibDir));
        }

        PathFragment libDir = input.getArtifact().getExecPath().getParentDirectory();

        // When COPY_DYNAMIC_LIBRARIES_TO_BINARY is enabled, dynamic libraries are not symlinked
        // under solibDir, so don't check it and don't include solibDir.
        if (!featureConfiguration.isEnabled(CppRuleClasses.COPY_DYNAMIC_LIBRARIES_TO_BINARY)) {
          // The first fragment is bazel-out, and the second may contain a configuration mnemonic.
          // We should always add the default solib dir because that's where libraries will be found
          // e.g. in remote execution, so we ignore the first two fragments.
          if (libDir.subFragment(2).equals(solibDir.subFragment(2))) {
            includeSolibDir = true;
          }
          if (libDir.equals(toolchainLibrariesSolibDir)) {
            includeToolchainLibrariesSolibDir = true;
          }
        }
        addDynamicInputLinkOptions(
            input,
            librariesToLink,
            expandedLinkerInputsBuilder,
            librarySearchDirectories,
            rpathRoots,
            rpathEntries);
      } else {
        addStaticInputLinkOptions(input, ltoMap, librariesToLink, expandedLinkerInputsBuilder);
      }
    }
    return Pair.of(includeSolibDir, includeToolchainLibrariesSolibDir);
  }

  /**
   * Adds command-line options for a dynamic library input file into options and libOpts.
   *
   * @param librariesToLink - a collection that will be exposed as a build variable.
   */
  private void addDynamicInputLinkOptions(
      LinkerInput input,
      SequenceBuilder librariesToLink,
      NestedSetBuilder<LinkerInput> expandedLinkerInputsBuilder,
      NestedSetBuilder<String> librarySearchDirectories,
      ImmutableList<String> rpathRoots,
      ImmutableSet.Builder<String> rpathRootsForExplicitSoDeps) {
    Preconditions.checkState(
        input.getArtifactCategory() == ArtifactCategory.DYNAMIC_LIBRARY
            || input.getArtifactCategory() == ArtifactCategory.INTERFACE_LIBRARY);
    Preconditions.checkState(
        !Link.useStartEndLib(
            input,
            CppHelper.getArchiveType(cppConfiguration, ccToolchainProvider, featureConfiguration)));

    expandedLinkerInputsBuilder.add(input);
    if (featureConfiguration.isEnabled(CppRuleClasses.TARGETS_WINDOWS)
        && ccToolchainProvider.supportsInterfaceSharedLibraries(featureConfiguration)) {
      // On Windows, dynamic library (dll) cannot be linked directly when using toolchains that
      // support interface library (eg. MSVC). If the user is doing so, it is only to be referenced
      // in other places (such as copy_dynamic_libraries_to_binary); skip adding it.
      if (CppFileTypes.SHARED_LIBRARY.matches(input.getArtifact().getFilename())) {
        return;
      }
    }

    Artifact inputArtifact = input.getArtifact();
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

    String name = inputArtifact.getFilename();

    // Use the normal shared library resolution rules if possible, otherwise treat as a versioned
    // library that must use the exact name. e.g.:
    // -lfoo -> libfoo.so
    // -l:foo -> foo.so
    // -l:libfoo.so.1 -> libfoo.so.1
    boolean hasCompatibleName =
        name.startsWith("lib") || (!name.endsWith(".so") && !name.endsWith(".dylib"));
    if (CppFileTypes.SHARED_LIBRARY.matches(name) && hasCompatibleName) {
      String libName = name.replaceAll("(^lib|\\.(so|dylib)$)", "");
      librariesToLink.addValue(LibraryToLinkValue.forDynamicLibrary(libName));
    } else if (CppFileTypes.SHARED_LIBRARY.matches(name)
        || CppFileTypes.VERSIONED_SHARED_LIBRARY.matches(name)) {
      librariesToLink.addValue(LibraryToLinkValue.forVersionedDynamicLibrary(name));
    } else {
      // Interface shared objects have a non-standard extension
      // that the linker won't be able to find.  So use the
      // filename directly rather than a -l option.  Since the
      // library has an SONAME attribute, this will work fine.
      librariesToLink.addValue(
          LibraryToLinkValue.forInterfaceLibrary(inputArtifact.getExecPathString()));
    }
  }

  /**
   * Adds command-line options for a static library or non-library input into options.
   *
   * @param librariesToLink - a collection that will be exposed as a build variable.
   */
  private void addStaticInputLinkOptions(
      LinkerInput input,
      Map<Artifact, Artifact> ltoMap,
      SequenceBuilder librariesToLink,
      NestedSetBuilder<LinkerInput> expandedLinkerInputsBuilder) {
    ArtifactCategory artifactCategory = input.getArtifactCategory();
    Preconditions.checkArgument(
        artifactCategory.equals(ArtifactCategory.OBJECT_FILE)
            || artifactCategory.equals(ArtifactCategory.STATIC_LIBRARY)
            || artifactCategory.equals(ArtifactCategory.ALWAYSLINK_STATIC_LIBRARY));
    boolean isAlwaysLinkStaticLibrary =
        artifactCategory == ArtifactCategory.ALWAYSLINK_STATIC_LIBRARY;

    // input.disableWholeArchive() should only be true for libstdc++/libc++ etc.
    boolean inputIsWholeArchive =
        !input.disableWholeArchive() && (isAlwaysLinkStaticLibrary || needWholeArchive);

    // If we had any LTO artifacts, ltoMap whould be non-null. In that case,
    // we should have created a thinltoParamFile which the LTO indexing
    // step will populate with the exec paths that correspond to the LTO
    // artifacts that the linker decided to include based on symbol resolution.
    // Those files will be included directly in the link (and not wrapped
    // in --start-lib/--end-lib) to ensure consistency between the two link
    // steps.
    Preconditions.checkState(ltoMap == null || thinltoParamFile != null || !allowLtoIndexing);

    // start-lib/end-lib library: adds its input object files.
    if (Link.useStartEndLib(
        input,
        CppHelper.getArchiveType(cppConfiguration, ccToolchainProvider, featureConfiguration))) {
      Iterable<Artifact> archiveMembers = input.getObjectFiles();
      if (!Iterables.isEmpty(archiveMembers)) {
        ImmutableList.Builder<Artifact> nonLtoArchiveMembersBuilder = ImmutableList.builder();
        for (Artifact member : archiveMembers) {
          Artifact a;
          if (ltoMap != null && (a = ltoMap.remove(member)) != null) {
            // When ltoMap is non-null the backend artifact may be missing due to libraries that
            // list .o files explicitly, or generate .o files from assembler.
            if (handledByLtoIndexing(a, allowLtoIndexing)) {
              // The LTO artifacts that should be included in the final link
              // are listed in the thinltoParamFile, generated by the LTO indexing.

              // Even if this object file is being skipped for exposure as a Build variable, it's
              // still an input to this action.
              expandedLinkerInputsBuilder.add(
                  LinkerInputs.simpleLinkerInput(
                      a,
                      ArtifactCategory.OBJECT_FILE,
                      /* disableWholeArchive= */ false,
                      a.getRootRelativePathString()));
              continue;
            }
            // No LTO indexing step, so use the LTO backend's generated artifact directly
            // instead of the bitcode object.
            member = a;
          }
          nonLtoArchiveMembersBuilder.add(member);
          expandedLinkerInputsBuilder.add(
              LinkerInputs.simpleLinkerInput(
                  member,
                  ArtifactCategory.OBJECT_FILE,
                  /* disableWholeArchive= */ false,
                  member.getRootRelativePathString()));
        }
        ImmutableList<Artifact> nonLtoArchiveMembers = nonLtoArchiveMembersBuilder.build();
        if (!nonLtoArchiveMembers.isEmpty()) {
          if (inputIsWholeArchive) {
            for (Artifact member : nonLtoArchiveMembers) {
              if (member.isTreeArtifact()) {
                // TODO(b/78189629): This object filegroup is expanded at action time but wrapped
                // with --start/--end-lib. There's currently no way to force these objects to be
                // linked in.
                librariesToLink.addValue(
                    LibraryToLinkValue.forObjectFileGroup(
                        ImmutableList.<Artifact>of(member), /* isWholeArchive= */ true));
              } else {
                // TODO(b/78189629): These each need to be their own LibraryToLinkValue so they're
                // not wrapped in --start/--end-lib (which lets the linker leave out objects with
                // unreferenced code).
                librariesToLink.addValue(
                    LibraryToLinkValue.forObjectFile(
                        member.getExecPathString(), /* isWholeArchive= */ true));
              }
            }
          } else {
            librariesToLink.addValue(
                LibraryToLinkValue.forObjectFileGroup(
                    nonLtoArchiveMembers, /* isWholeArchive= */ false));
          }
        }
      }
    } else {
      Artifact inputArtifact = input.getArtifact();
      Artifact a;
      if (ltoMap != null && (a = ltoMap.remove(inputArtifact)) != null) {
        if (handledByLtoIndexing(a, allowLtoIndexing)) {
          // The LTO artifacts that should be included in the final link
          // are listed in the thinltoParamFile, generated by the LTO indexing.

          // Even if this object file is being skipped for exposure as a build variable, it's
          // still an input to this action.
          expandedLinkerInputsBuilder.add(
              LinkerInputs.simpleLinkerInput(
                  a,
                  ArtifactCategory.OBJECT_FILE,
                  /* disableWholeArchive= */ false,
                  a.getRootRelativePathString()));
          return;
        }
        // No LTO indexing step, so use the LTO backend's generated artifact directly
        // instead of the bitcode object.
        inputArtifact = a;
      }

      if (artifactCategory.equals(ArtifactCategory.OBJECT_FILE)) {
        if (inputArtifact.isTreeArtifact()) {
          librariesToLink.addValue(
              LibraryToLinkValue.forObjectFileGroup(
                  ImmutableList.<Artifact>of(inputArtifact), inputIsWholeArchive));
        } else {
          librariesToLink.addValue(
              LibraryToLinkValue.forObjectFile(
                  inputArtifact.getExecPathString(), inputIsWholeArchive));
        }
        if (!input.isLinkstamp()) {
          expandedLinkerInputsBuilder.add(input);
        }
      } else {
        librariesToLink.addValue(
            LibraryToLinkValue.forStaticLibrary(
                inputArtifact.getExecPathString(), inputIsWholeArchive));
        expandedLinkerInputsBuilder.add(input);
      }
    }
  }

  /**
   * Returns true if this artifact is produced from a bitcode file that will be input to the LTO
   * indexing step, in which case that step will add it to the generated thinltoParamFile for
   * inclusion in the final link step if the linker decides to include it.
   *
   * @param a is an artifact produced by an LTO backend.
   * @param allowLtoIndexing
   */
  private static boolean handledByLtoIndexing(Artifact a, boolean allowLtoIndexing) {
    // If no LTO indexing is allowed for this link, then none are handled by LTO indexing.
    // Otherwise, this may be from a linkstatic library that we decided not to include in
    // LTO indexing because we are linking a test, to improve scalability when linking many tests.
    return allowLtoIndexing
        && !a.getRootRelativePath().startsWith(CppHelper.SHARED_NONLTO_BACKEND_ROOT_PREFIX);
  }

  @Nullable
  private Map<Artifact, Artifact> generateLtoMap() {
    if (isLtoIndexing || allLtoArtifacts == null) {
      return null;
    }
    // TODO(bazel-team): The LTO final link can only work if there are individual .o files on
    // the command line. Rather than crashing, this should issue a nice error. We will get
    // this by
    // 1) moving supports_start_end_lib to a toolchain feature
    // 2) having thin_lto require start_end_lib
    // As a bonus, we can rephrase --nostart_end_lib as --features=-start_end_lib and get rid
    // of a command line option.

    Preconditions.checkState(
        CppHelper.useStartEndLib(cppConfiguration, ccToolchainProvider, featureConfiguration));
    Map<Artifact, Artifact> ltoMap = new HashMap<>();
    for (LtoBackendArtifacts l : allLtoArtifacts) {
      ltoMap.put(l.getBitcodeFile(), l.getObjectFile());
    }
    return ltoMap;
  }
}
