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

import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcToolchainVariables.LibraryToLinkValue;
import com.google.devtools.build.lib.rules.cpp.CcToolchainVariables.SequenceBuilder;
import com.google.devtools.build.lib.rules.cpp.Link.LinkTargetType;
import com.google.devtools.build.lib.rules.cpp.Link.LinkingMode;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.HashMap;
import java.util.Map;

/** Class that goes over linker inputs and produces {@link LibraryToLinkValue}s */
public class LibrariesToLinkCollector {

  private final boolean isNativeDeps;
  private final PathFragment toolchainLibrariesSolibDir;
  private final CppConfiguration cppConfiguration;
  private final CcToolchainProvider ccToolchainProvider;
  private final Artifact outputArtifact;
  private final boolean isLtoIndexing;
  private final PathFragment solibDir;
  private final Iterable<? extends LinkerInput> linkerInputs;
  private final Iterable<LtoBackendArtifacts> allLtoArtifacts;
  private final boolean allowLtoIndexing;
  private final Artifact thinltoParamFile;
  private final FeatureConfiguration featureConfiguration;
  private final boolean needWholeArchive;
  private final String rpathRoot;
  private final boolean needToolchainLibrariesRpath;
  private final Map<Artifact, Artifact> ltoMap;

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
      boolean needWholeArchive) {
    this.isNativeDeps = isNativeDeps;
    this.cppConfiguration = cppConfiguration;
    this.ccToolchainProvider = toolchain;
    this.toolchainLibrariesSolibDir = toolchainLibrariesSolibDir;
    this.outputArtifact = output;
    this.solibDir = solibDir;
    this.isLtoIndexing = isLtoIndexing;
    this.allLtoArtifacts = allLtoArtifacts;
    this.featureConfiguration = featureConfiguration;
    this.thinltoParamFile = thinltoParamFile;
    this.allowLtoIndexing = allowLtoIndexing;
    this.linkerInputs = linkerInputs;
    this.needWholeArchive = needWholeArchive;

    needToolchainLibrariesRpath =
        toolchainLibrariesSolibDir != null
            && (linkType.isDynamicLibrary()
                || (linkType == LinkTargetType.EXECUTABLE && linkingMode == LinkingMode.DYNAMIC));

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
      rpathRoot = ccToolchainProvider.getSolibDirectory() + "/";
    } else {
      rpathRoot =
          Strings.repeat("../", outputArtifact.getRootRelativePath().segmentCount() - 1)
              + ccToolchainProvider.getSolibDirectory()
              + "/";
    }

    ltoMap = generateLtoMap();
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
    NestedSetBuilder<String> runtimeLibrarySearchDirectories = NestedSetBuilder.linkOrder();
    ImmutableSet.Builder<String> rpathRootsForExplicitSoDeps = ImmutableSet.builder();
    NestedSetBuilder<LinkerInput> expandedLinkerInputsBuilder = NestedSetBuilder.linkOrder();
    // List of command line parameters that need to be placed *outside* of
    // --whole-archive ... --no-whole-archive.
    SequenceBuilder librariesToLink = new SequenceBuilder();

    String toolchainLibrariesSolibName =
        toolchainLibrariesSolibDir != null ? toolchainLibrariesSolibDir.getBaseName() : null;
    if (isNativeDeps && cppConfiguration.shareNativeDeps()) {
      if (needToolchainLibrariesRpath) {
        runtimeLibrarySearchDirectories.add("../" + toolchainLibrariesSolibName + "/");
      }
    } else {
      // For all other links, calculate the relative path from the output file to _solib_[arch]
      // (the directory where all shared libraries are stored, which resides under the blaze-bin
      // directory. In other words, given blaze-bin/my/package/binary, rpathRoot would be
      // "../../_solib_[arch]".
      if (needToolchainLibrariesRpath) {
        runtimeLibrarySearchDirectories.add(
            Strings.repeat("../", outputArtifact.getRootRelativePath().segmentCount() - 1)
                + toolchainLibrariesSolibName
                + "/");
      }
      if (isNativeDeps) {
        // We also retain the $ORIGIN/ path to solibs that are in _solib_<arch>, as opposed to
        // the package directory)
        if (needToolchainLibrariesRpath) {
          runtimeLibrarySearchDirectories.add("../" + toolchainLibrariesSolibName + "/");
        }
      }
    }

    if (needToolchainLibrariesRpath) {
      if (isNativeDeps) {
        runtimeLibrarySearchDirectories.add(".");
      }
      runtimeLibrarySearchDirectories.add(toolchainLibrariesSolibName + "/");
    }

    Pair<Boolean, Boolean> includeSolibsPair =
        addLinkerInputs(
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
      allRuntimeLibrarySearchDirectories.add(rpathRoot);
    }
    allRuntimeLibrarySearchDirectories.addAll(rpathRootsForExplicitSoDeps.build());
    if (includeToolchainLibrariesSolibDir) {
      allRuntimeLibrarySearchDirectories.addTransitive(runtimeLibrarySearchDirectories.build());
    }

    return new CollectedLibrariesToLink(
        librariesToLink,
        expandedLinkerInputsBuilder.build(),
        librarySearchDirectories.build(),
        allRuntimeLibrarySearchDirectories.build());
  }

  private Pair<Boolean, Boolean> addLinkerInputs(
      NestedSetBuilder<String> librarySearchDirectories,
      ImmutableSet.Builder<String> rpathEntries,
      SequenceBuilder librariesToLink,
      NestedSetBuilder<LinkerInput> expandedLinkerInputsBuilder) {
    boolean includeSolibDir = false;
    boolean includeToolchainLibrariesSolibDir = false;
    for (LinkerInput input : linkerInputs) {
      if (input.getArtifactCategory() == ArtifactCategory.DYNAMIC_LIBRARY
          || input.getArtifactCategory() == ArtifactCategory.INTERFACE_LIBRARY) {
        PathFragment libDir = input.getArtifact().getExecPath().getParentDirectory();
        // When COPY_DYNAMIC_LIBRARIES_TO_BINARY is enabled, dynamic libraries are not symlinked
        // under solibDir, so don't check it and don't include solibDir.
        if (!featureConfiguration.isEnabled(CppRuleClasses.COPY_DYNAMIC_LIBRARIES_TO_BINARY)) {
          Preconditions.checkState(
              libDir.startsWith(solibDir) || libDir.startsWith(toolchainLibrariesSolibDir),
              "Artifact '%s' is not under directory expected '%s',"
                  + " neither it is in directory for toolchain libraries '%s'.",
              input.getArtifact(),
              solibDir,
              toolchainLibrariesSolibDir);
          if (libDir.equals(solibDir)) {
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
            rpathEntries);
      } else {
        addStaticInputLinkOptions(input, librariesToLink, expandedLinkerInputsBuilder);
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

      rpathRootsForExplicitSoDeps.add(
          rpathRoot + dotdots + libDir.relativeTo(commonParent).getPathString());
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
                      a, ArtifactCategory.OBJECT_FILE, /* disableWholeArchive= */ false));
              continue;
            }
            // No LTO indexing step, so use the LTO backend's generated artifact directly
            // instead of the bitcode object.
            member = a;
          }
          nonLtoArchiveMembersBuilder.add(member);
          expandedLinkerInputsBuilder.add(
              LinkerInputs.simpleLinkerInput(
                  member, ArtifactCategory.OBJECT_FILE, /* disableWholeArchive  = */ false));
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
                  a, ArtifactCategory.OBJECT_FILE, /* disableWholeArchive= */ false));
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
        && !a.getRootRelativePath()
        .startsWith(
            PathFragment.create(CppLinkActionBuilder.SHARED_NONLTO_BACKEND_ROOT_PREFIX));
  }

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
