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
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcToolchainVariables.SequenceBuilder;
import com.google.devtools.build.lib.vfs.PathFragment;
import net.starlark.java.eval.EvalException;

/** Enum covering all build variables we create for all various {@link CppLinkAction}. */
public enum LinkBuildVariables {
  /** Entries in the linker runtime search path (usually set by -rpath flag) */
  RUNTIME_LIBRARY_SEARCH_DIRECTORIES("runtime_library_search_directories"),
  /** Entries in the linker search path (usually set by -L flag) */
  LIBRARY_SEARCH_DIRECTORIES("library_search_directories"),
  /** Flags providing files to link as inputs in the linker invocation */
  LIBRARIES_TO_LINK("libraries_to_link"),
  /** Thinlto param file produced by thinlto-indexing action consumed by the final link action. */
  THINLTO_PARAM_FILE("thinlto_param_file"),
  /** Location of def file used on Windows with MSVC */
  DEF_FILE_PATH("def_file_path"),
  /** Location where hinlto should write thinlto_param_file flags when indexing. */
  THINLTO_INDEXING_PARAM_FILE("thinlto_indexing_param_file"),

  THINLTO_PREFIX_REPLACE("thinlto_prefix_replace"),
  /**
   * A build variable to let the LTO indexing step know how to map from the minimized bitcode file
   * to the full bitcode file used by the LTO Backends.
   */
  THINLTO_OBJECT_SUFFIX_REPLACE("thinlto_object_suffix_replace"),
  /**
   * A build variable for the path to the merged object file, which is an object file that is
   * created during the LTO indexing step and needs to be passed to the final link.
   */
  THINLTO_MERGED_OBJECT_FILE("thinlto_merged_object_file"),
  /** Location of linker param file created by bazel to overcome command line length limit */
  LINKER_PARAM_FILE("linker_param_file"),
  /** execpath of the output of the linker. */
  OUTPUT_EXECPATH("output_execpath"),
  /** "yes"|"no" depending on whether interface library should be generated. */
  GENERATE_INTERFACE_LIBRARY("generate_interface_library"),
  /** Path to the interface library builder tool. */
  INTERFACE_LIBRARY_BUILDER("interface_library_builder_path"),
  /** Input for the interface library ifso builder tool. */
  INTERFACE_LIBRARY_INPUT("interface_library_input_path"),
  /** Path where to generate interface library using the ifso builder tool. */
  INTERFACE_LIBRARY_OUTPUT("interface_library_output_path"),
  /** Linker flags coming from the --linkopt or linkopts attribute. */
  USER_LINK_FLAGS("user_link_flags"),
  /** A build variable giving linkstamp paths. */
  LINKSTAMP_PATHS("linkstamp_paths"),
  /** Presence of this variable indicates that PIC code should be generated. */
  FORCE_PIC("force_pic"),
  /** Presence of this variable indicates that the debug symbols should be stripped. */
  STRIP_DEBUG_SYMBOLS("strip_debug_symbols"),
  /** Truthy when current action is a cc_test linking action, falsey otherwise. */
  IS_CC_TEST("is_cc_test"),
  /**
   * Presence of this variable indicates that files were compiled with fission (debug info is in
   * .dwo files instead of .o files and linker needs to know).
   */
  IS_USING_FISSION("is_using_fission"),
  /** Path to the fdo instrument. */
  FDO_INSTRUMENT_PATH("fdo_instrument_path"),
  /** Path to the context sensitive fdo instrument. */
  CS_FDO_INSTRUMENT_PATH("cs_fdo_instrument_path"),
  /** Path to the Propeller Optimize linker profile artifact */
  PROPELLER_OPTIMIZE_LD_PATH("propeller_optimize_ld_path");

  private final String variableName;

  LinkBuildVariables(String variableName) {
    this.variableName = variableName;
  }

  public String getVariableName() {
    return variableName;
  }

  public static CcToolchainVariables setupVariables(
      boolean isUsingLinkerNotArchiver,
      PathFragment binDirectoryPath,
      String outputFile,
      boolean isCreatingSharedLibrary,
      String paramFile,
      String thinltoParamFile,
      String thinltoMergedObjectFile,
      boolean mustKeepDebug,
      CcToolchainProvider ccToolchainProvider,
      CppConfiguration cppConfiguration,
      BuildOptions buildOptions,
      FeatureConfiguration featureConfiguration,
      boolean useTestOnlyFlags,
      boolean isLtoIndexing,
      Iterable<String> userLinkFlags,
      String interfaceLibraryBuilder,
      String interfaceLibraryOutput,
      PathFragment ltoOutputRootPrefix,
      String defFile,
      FdoContext fdoContext,
      NestedSet<String> runtimeLibrarySearchDirectories,
      SequenceBuilder librariesToLink,
      NestedSet<String> librarySearchDirectories,
      boolean addIfsoRelatedVariables)
      throws EvalException {
    CcToolchainVariables.Builder buildVariables =
        CcToolchainVariables.builder(
            ccToolchainProvider.getBuildVariables(buildOptions, cppConfiguration));

    // pic
    if (cppConfiguration.forcePic()) {
      buildVariables.addStringVariable(FORCE_PIC.getVariableName(), "");
    }

    if (!mustKeepDebug && cppConfiguration.shouldStripBinaries()) {
      buildVariables.addStringVariable(STRIP_DEBUG_SYMBOLS.getVariableName(), "");
    }

    if (isUsingLinkerNotArchiver
        && ccToolchainProvider.shouldCreatePerObjectDebugInfo(
            featureConfiguration, cppConfiguration)) {
      buildVariables.addStringVariable(IS_USING_FISSION.getVariableName(), "");
    }

    if (useTestOnlyFlags) {
      buildVariables.addIntegerVariable(IS_CC_TEST.getVariableName(), 1);
    } else {
      buildVariables.addIntegerVariable(IS_CC_TEST.getVariableName(), 0);
    }

    if (runtimeLibrarySearchDirectories != null) {
      buildVariables.addStringSequenceVariable(
          RUNTIME_LIBRARY_SEARCH_DIRECTORIES.getVariableName(), runtimeLibrarySearchDirectories);
    }

    if (librariesToLink != null) {
      buildVariables.addCustomBuiltVariable(LIBRARIES_TO_LINK.getVariableName(), librariesToLink);
    }

    buildVariables.addStringSequenceVariable(
        LIBRARY_SEARCH_DIRECTORIES.getVariableName(), librarySearchDirectories);

    if (paramFile != null) {
      buildVariables.addStringVariable(LINKER_PARAM_FILE.getVariableName(), paramFile);
    }

    // output exec path
    if (outputFile != null && !isLtoIndexing) {
      buildVariables.addStringVariable(OUTPUT_EXECPATH.getVariableName(), outputFile);
    }

    if (isLtoIndexing) {
      if (thinltoParamFile != null) {
        // This is a lto-indexing action and we want it to populate param file.
        buildVariables.addStringVariable(
            THINLTO_INDEXING_PARAM_FILE.getVariableName(), thinltoParamFile);
        // TODO(b/33846234): Remove once all the relevant crosstools don't depend on the variable.
        buildVariables.addStringVariable("thinlto_optional_params_file", "=" + thinltoParamFile);
      } else {
        buildVariables.addStringVariable(THINLTO_INDEXING_PARAM_FILE.getVariableName(), "");
        // TODO(b/33846234): Remove once all the relevant crosstools don't depend on the variable.
        buildVariables.addStringVariable("thinlto_optional_params_file", "");
      }
      buildVariables.addStringVariable(
          THINLTO_PREFIX_REPLACE.getVariableName(),
          binDirectoryPath.getSafePathString()
              + ";"
              + binDirectoryPath.getRelative(ltoOutputRootPrefix));
      String objectFileExtension =
          ccToolchainProvider
              .getFeatures()
              .getArtifactNameExtensionForCategory(ArtifactCategory.OBJECT_FILE);
      if (!featureConfiguration.isEnabled(CppRuleClasses.NO_USE_LTO_INDEXING_BITCODE_FILE)) {
        buildVariables.addStringVariable(
            THINLTO_OBJECT_SUFFIX_REPLACE.getVariableName(),
            Iterables.getOnlyElement(CppFileTypes.LTO_INDEXING_OBJECT_FILE.getExtensions())
                + ";"
                + objectFileExtension);
      }
      if (thinltoMergedObjectFile != null) {
        buildVariables.addStringVariable(
            THINLTO_MERGED_OBJECT_FILE.getVariableName(), thinltoMergedObjectFile);
      }
    } else {
      if (thinltoParamFile != null) {
        // This is a normal link action and we need to use param file created by lto-indexing.
        buildVariables.addStringVariable(THINLTO_PARAM_FILE.getVariableName(), thinltoParamFile);
      }
    }

    if (addIfsoRelatedVariables) {
      boolean shouldGenerateInterfaceLibrary =
          outputFile != null
              && interfaceLibraryBuilder != null
              && interfaceLibraryOutput != null
              && !isLtoIndexing;
      buildVariables.addStringVariable(
          GENERATE_INTERFACE_LIBRARY.getVariableName(),
          shouldGenerateInterfaceLibrary ? "yes" : "no");
      buildVariables.addStringVariable(
          INTERFACE_LIBRARY_BUILDER.getVariableName(),
          shouldGenerateInterfaceLibrary ? interfaceLibraryBuilder : "ignored");
      buildVariables.addStringVariable(
          INTERFACE_LIBRARY_INPUT.getVariableName(),
          shouldGenerateInterfaceLibrary ? outputFile : "ignored");
      buildVariables.addStringVariable(
          INTERFACE_LIBRARY_OUTPUT.getVariableName(),
          shouldGenerateInterfaceLibrary ? interfaceLibraryOutput : "ignored");
    }

    if (defFile != null) {
      buildVariables.addStringVariable(DEF_FILE_PATH.getVariableName(), defFile);
    }

    if (featureConfiguration.isEnabled(CppRuleClasses.FDO_INSTRUMENT)) {
      Preconditions.checkArgument(fdoContext.getBranchFdoProfile() == null);
      String fdoInstrument = cppConfiguration.getFdoInstrument();
      Preconditions.checkNotNull(fdoInstrument);
      buildVariables.addStringVariable(FDO_INSTRUMENT_PATH.getVariableName(), fdoInstrument);
    } else if (featureConfiguration.isEnabled(CppRuleClasses.CS_FDO_INSTRUMENT)) {
      String csFdoInstrument = ccToolchainProvider.getCSFdoInstrument();
      Preconditions.checkNotNull(csFdoInstrument);
      buildVariables.addStringVariable(CS_FDO_INSTRUMENT_PATH.getVariableName(), csFdoInstrument);
    }

    if (featureConfiguration.isEnabled(CppRuleClasses.PROPELLER_OPTIMIZE)
        && fdoContext.getPropellerOptimizeInputFile().getLdArtifact() != null) {
      buildVariables.addStringVariable(
          PROPELLER_OPTIMIZE_LD_PATH.getVariableName(),
          fdoContext.getPropellerOptimizeInputFile().getLdArtifact().getExecPathString());
    }
    Iterable<String> userLinkFlagsWithLtoIndexingIfNeeded;
    if (!isLtoIndexing || cppConfiguration.useStandaloneLtoIndexingCommandLines()) {
      userLinkFlagsWithLtoIndexingIfNeeded = userLinkFlags;
    } else {
      ImmutableList.Builder<String> opts = ImmutableList.builder();
      opts.addAll(userLinkFlags);
      opts.addAll(
          featureConfiguration.getCommandLine(
              CppActionNames.LTO_INDEXING, buildVariables.build(), /* expander= */ null));
      opts.addAll(cppConfiguration.getLtoIndexOptions());
      userLinkFlagsWithLtoIndexingIfNeeded = opts.build();
    }

    // For now, silently ignore linkopts if this is a static library
    userLinkFlagsWithLtoIndexingIfNeeded =
        isUsingLinkerNotArchiver ? userLinkFlagsWithLtoIndexingIfNeeded : ImmutableList.of();

    buildVariables.addStringSequenceVariable(
        LinkBuildVariables.USER_LINK_FLAGS.getVariableName(),
        removePieIfCreatingSharedLibrary(
            isCreatingSharedLibrary, userLinkFlagsWithLtoIndexingIfNeeded));
    return buildVariables.build();
  }

  private static Iterable<String> removePieIfCreatingSharedLibrary(
      boolean isCreatingSharedLibrary, Iterable<String> flags) {
    if (isCreatingSharedLibrary) {
      return Iterables.filter(
          flags,
          Predicates.not(
              Predicates.or(Predicates.equalTo("-pie"), Predicates.equalTo("-Wl,-pie"))));
    } else {
      return flags;
    }
  }
}
