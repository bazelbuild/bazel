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

import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Variables;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Variables.SequenceBuilder;
import com.google.devtools.build.lib.vfs.PathFragment;

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
  /** Linker flags coming from the legacy crosstool fields. */
  LEGACY_LINK_FLAGS("legacy_link_flags"),
  /** Path to which to write symbol counts. */
  SYMBOL_COUNTS_OUTPUT("symbol_counts_output"),
  /** A build variable giving linkstamp paths. */
  LINKSTAMP_PATHS("linkstamp_paths"),
  /** Presence of this variable indicates that PIC code should be generated. */
  FORCE_PIC("force_pic"),
  /** Presence of this variable indicates that the debug symbols should be stripped. */
  STRIP_DEBUG_SYMBOLS("strip_debug_symbols"),
  @Deprecated
  IS_CC_TEST_LINK_ACTION("is_cc_test_link_action"),
  @Deprecated
  IS_NOT_CC_TEST_LINK_ACTION("is_not_cc_test_link_action"),
  /** Truthy when current action is a cc_test linking action, falsey otherwise. */
  IS_CC_TEST("is_cc_test"),
  /**
   * Presence of this variable indicates that files were compiled with fission (debug info is in
   * .dwo files instead of .o files and linker needs to know).
   */
  IS_USING_FISSION("is_using_fission");

  private final String variableName;

  LinkBuildVariables(String variableName) {
    this.variableName = variableName;
  }

  public String getVariableName() {
    return variableName;
  }

  public static Variables setupVariables(
      boolean isUsingLinkerNotArchiver,
      BuildConfiguration configuration,
      Artifact outputArtifact,
      Artifact paramFile,
      Artifact thinltoParamFile,
      Artifact thinltoMergedObjectFile,
      boolean mustKeepDebug,
      Artifact symbolCounts,
      CcToolchainProvider ccToolchainProvider,
      FeatureConfiguration featureConfiguration,
      boolean useTestOnlyFlags,
      boolean isLtoIndexing,
      Artifact interfaceLibraryBuilder,
      Artifact interfaceLibraryOutput,
      PathFragment ltoOutputRootPrefix,
      ActionInput defFile,
      FdoSupportProvider fdoSupport,
      Iterable<String> runtimeLibrarySearchDirectories,
      SequenceBuilder librariesToLink,
      Iterable<String> librarySearchDirectories) {
    Variables.Builder buildVariables = new Variables.Builder();

    // symbol counting
    if (symbolCounts != null) {
      buildVariables.addStringVariable(
          SYMBOL_COUNTS_OUTPUT.getVariableName(), symbolCounts.getExecPathString());
    }

    // pic
    if (ccToolchainProvider.getForcePic()) {
      buildVariables.addStringVariable(FORCE_PIC.getVariableName(), "");
    }

    if (!mustKeepDebug && ccToolchainProvider.getShouldStripBinaries()) {
      buildVariables.addStringVariable(STRIP_DEBUG_SYMBOLS.getVariableName(), "");
    }

    if (isUsingLinkerNotArchiver
        && ccToolchainProvider.shouldCreatePerObjectDebugInfo(featureConfiguration)) {
      buildVariables.addStringVariable(IS_USING_FISSION.getVariableName(), "");
    }

    if (useTestOnlyFlags) {
      buildVariables.addIntegerVariable(IS_CC_TEST.getVariableName(), 1);
      buildVariables.addStringVariable(IS_CC_TEST_LINK_ACTION.getVariableName(), "");
    } else {
      buildVariables.addIntegerVariable(IS_CC_TEST.getVariableName(), 0);
      buildVariables.addStringVariable(IS_NOT_CC_TEST_LINK_ACTION.getVariableName(), "");
    }

    if (runtimeLibrarySearchDirectories != null) {
      buildVariables.addStringSequenceVariable(
          RUNTIME_LIBRARY_SEARCH_DIRECTORIES.getVariableName(), runtimeLibrarySearchDirectories);
    }

    buildVariables.addCustomBuiltVariable(LIBRARIES_TO_LINK.getVariableName(), librariesToLink);
    // TODO(b/72803478): Remove once existing crosstools have been migrated
    buildVariables.addStringVariable("libs_to_link_dont_emit_objects_for_archiver", "");

    buildVariables.addStringSequenceVariable(
        LIBRARY_SEARCH_DIRECTORIES.getVariableName(), librarySearchDirectories);

    if (paramFile != null) {
      buildVariables.addStringVariable(
          LINKER_PARAM_FILE.getVariableName(), paramFile.getExecPathString());
    }

    // output exec path
    if (outputArtifact != null && !isLtoIndexing) {
      buildVariables.addStringVariable(
          OUTPUT_EXECPATH.getVariableName(), outputArtifact.getExecPathString());
    }

    if (isLtoIndexing) {
      if (thinltoParamFile != null) {
        // This is a lto-indexing action and we want it to populate param file.
        buildVariables.addStringVariable(
            THINLTO_INDEXING_PARAM_FILE.getVariableName(), thinltoParamFile.getExecPathString());
        // TODO(b/33846234): Remove once all the relevant crosstools don't depend on the variable.
        buildVariables.addStringVariable(
            "thinlto_optional_params_file", "=" + thinltoParamFile.getExecPathString());
      } else {
        buildVariables.addStringVariable(THINLTO_INDEXING_PARAM_FILE.getVariableName(), "");
        // TODO(b/33846234): Remove once all the relevant crosstools don't depend on the variable.
        buildVariables.addStringVariable("thinlto_optional_params_file", "");
      }
      buildVariables.addStringVariable(
          THINLTO_PREFIX_REPLACE.getVariableName(),
          configuration.getBinDirectory().getExecPathString()
              + ";"
              + configuration.getBinDirectory().getExecPath().getRelative(ltoOutputRootPrefix));
      buildVariables.addStringVariable(
          THINLTO_OBJECT_SUFFIX_REPLACE.getVariableName(),
          Iterables.getOnlyElement(CppFileTypes.LTO_INDEXING_OBJECT_FILE.getExtensions())
              + ";"
              + Iterables.getOnlyElement(CppFileTypes.OBJECT_FILE.getExtensions()));
      if (thinltoMergedObjectFile != null) {
        buildVariables.addStringVariable(
            THINLTO_MERGED_OBJECT_FILE.getVariableName(),
            thinltoMergedObjectFile.getExecPathString());
      }
    } else {
      if (thinltoParamFile != null) {
        // This is a normal link action and we need to use param file created by lto-indexing.
        buildVariables.addStringVariable(
            THINLTO_PARAM_FILE.getVariableName(), thinltoParamFile.getExecPathString());
      }
    }
    boolean shouldGenerateInterfaceLibrary =
        outputArtifact != null
            && interfaceLibraryBuilder != null
            && interfaceLibraryOutput != null
            && !isLtoIndexing;
    buildVariables.addStringVariable(
        GENERATE_INTERFACE_LIBRARY.getVariableName(),
        shouldGenerateInterfaceLibrary ? "yes" : "no");
    buildVariables.addStringVariable(
        INTERFACE_LIBRARY_BUILDER.getVariableName(),
        shouldGenerateInterfaceLibrary ? interfaceLibraryBuilder.getExecPathString() : "ignored");
    buildVariables.addStringVariable(
        INTERFACE_LIBRARY_INPUT.getVariableName(),
        shouldGenerateInterfaceLibrary ? outputArtifact.getExecPathString() : "ignored");
    buildVariables.addStringVariable(
        INTERFACE_LIBRARY_OUTPUT.getVariableName(),
        shouldGenerateInterfaceLibrary ? interfaceLibraryOutput.getExecPathString() : "ignored");

    if (defFile != null) {
      buildVariables.addStringVariable(
          DEF_FILE_PATH.getVariableName(), defFile.getExecPathString());
    }

    fdoSupport.getFdoSupport().getLinkOptions(featureConfiguration, buildVariables);
    return buildVariables.build();
  }
}