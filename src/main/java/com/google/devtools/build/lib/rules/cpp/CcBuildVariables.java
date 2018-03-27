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

import com.google.common.base.Supplier;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Variables;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Variables.StringSequenceBuilder;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Variables.VariablesExtension;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Collection;

/**
 * Utility class for collecting knowledge about what build variables we create and when we create
 * them. Having this central location is useful so we can reason about what variables expose to
 * Skylark, what variables are redundant etc.
 */
public class CcBuildVariables {

  /** Enum covering all build variables we create for 'strip' actions. */
  public enum StripBuildVariables {
    /** Variable for the path to the compilation output file. */
    OUTPUT_FILE("output_file"),
    /** Variable for stripopts. */
    STRIPOPTS("stripopts");

    private final String variableName;

    StripBuildVariables(String variableName) {
      this.variableName = variableName;
    }

    public String getVariableName() {
      return variableName;
    }
  }

  /** Enum covering all build variables we create for all various {@link CppCompileAction}. */
  public enum CompileBuildVariables {
    /** Variable for the path to the source file being compiled. */
    SOURCE_FILE("source_file"),
    /**
     * Variable for all flags coming from copt rule attribute, and from --copt, --cxxopt, or
     * --conlyopt options.
     */
    USER_COMPILE_FLAGS("user_compile_flags"),
    /**
     * Variable for all flags coming from legacy crosstool fields, such as compiler_flag,
     * optional_compiler_flag, cxx_flag.
     */
    LEGACY_COMPILE_FLAGS("legacy_compile_flags"),
    /** Variable for flags coming from unfiltered_cxx_flag CROSSTOOL fields. */
    UNFILTERED_COMPILE_FLAGS("unfiltered_compile_flags"),
    /** Variable for the path to the output file when output is an object file. */
    OUTPUT_OBJECT_FILE("output_object_file"),
    /** Variable for the path to the compilation output file. */
    OUTPUT_FILE("output_file"),
    /** Variable for the dependency file path */
    DEPENDENCY_FILE("dependency_file"),
    /** Variable for the module file name. */
    MODULE_NAME("module_name"),
    /**
     * Variable for the collection of include paths.
     *
     * @see CcCompilationInfo#getIncludeDirs().
     */
    INCLUDE_PATHS("include_paths"),
    /**
     * Variable for the collection of quote include paths.
     *
     * @see CcCompilationInfo#getIncludeDirs().
     */
    QUOTE_INCLUDE_PATHS("quote_include_paths"),
    /**
     * Variable for the collection of system include paths.
     *
     * @see CcCompilationInfo#getIncludeDirs().
     */
    SYSTEM_INCLUDE_PATHS("system_include_paths"),
    /** Variable for the module map file name. */
    MODULE_MAP_FILE("module_map_file"),
    /** Variable for the dependent module map file name. */
    DEPENDENT_MODULE_MAP_FILES("dependent_module_map_files"),
    /** Variable for the collection of module files. */
    MODULE_FILES("module_files"),
    /** Variable for the collection of macros defined for preprocessor. */
    PREPROCESSOR_DEFINES("preprocessor_defines"),
    /** Variable for the gcov coverage file path. */
    GCOV_GCNO_FILE("gcov_gcno_file"),
    /** Variable for the LTO indexing bitcode file. */
    LTO_INDEXING_BITCODE_FILE("lto_indexing_bitcode_file"),
    /** Variable for the per object debug info file. */
    PER_OBJECT_DEBUG_INFO_FILE("per_object_debug_info_file"),
    /** Variable present when the output is compiled as position independent. */
    PIC("pic"),
    /** Variable marking that we are generating preprocessed sources (from --save_temps). */
    OUTPUT_PREPROCESS_FILE("output_preprocess_file"),
    /** Variable marking that we are generating assembly source (from --save_temps). */
    OUTPUT_ASSEMBLY_FILE("output_assembly_file"),
    /** Path to the fdo instrument artifact */
    FDO_INSTRUMENT_PATH("fdo_instrument_path"),
    /** Path to the fdo profile artifact */
    FDO_PROFILE_PATH("fdo_profile_path"),
    /** Variable for includes that compiler needs to include into sources. */
    INCLUDES("includes");

    private final String variableName;

    CompileBuildVariables(String variableName) {
      this.variableName = variableName;
    }

    public String getVariableName() {
      return variableName;
    }
  }

  public static Variables setupCompileBuildVariables(
      RuleContext ruleContext,
      FeatureConfiguration featureConfiguration,
      CcToolchainProvider ccToolchainProvider,
      Artifact sourceFile,
      Artifact outputFile,
      Artifact gcnoFile,
      Artifact dwoFile,
      Artifact ltoIndexingFile,
      CcCompilationInfo ccCompilationInfo,
      ImmutableList<String> includes,
      ImmutableList<String> userCompileFlags,
      CppModuleMap cppModuleMap,
      boolean usePic,
      PathFragment realOutputFilePath,
      String fdoStamp,
      String dotdFileExecPath,
      ImmutableList<VariablesExtension> variablesExtensions,
      ImmutableMap<String, String> additionalBuildVariables,
      // TODO(b/76449614): Remove use of optional_*_flag from CROSSTOOL and get rid of this param
      ImmutableSet<String> features) {
    Variables.Builder buildVariables =
        new Variables.Builder(ccToolchainProvider.getBuildVariables());

    buildVariables.addStringSequenceVariable(
        CompileBuildVariables.USER_COMPILE_FLAGS.getVariableName(), userCompileFlags);

    buildVariables.addStringVariable(
        CompileBuildVariables.SOURCE_FILE.getVariableName(), sourceFile.getExecPathString());

    String sourceFilename = sourceFile.getExecPathString();
    buildVariables.addLazyStringSequenceVariable(
        CompileBuildVariables.LEGACY_COMPILE_FLAGS.getVariableName(),
        getLegacyCompileFlagsSupplier(
            ruleContext.getFragment(CppConfiguration.class),
            ccToolchainProvider,
            sourceFilename,
            features));

    if (!CppFileTypes.OBJC_SOURCE.matches(sourceFilename)
        && !CppFileTypes.OBJCPP_SOURCE.matches(sourceFilename)) {
      buildVariables.addLazyStringSequenceVariable(
          CompileBuildVariables.UNFILTERED_COMPILE_FLAGS.getVariableName(),
          getUnfilteredCompileFlagsSupplier(ccToolchainProvider, features));
    }

    // TODO(b/76195763): Remove once blaze with cl/189769259 is released and crosstools are updated.
    if (!FileType.contains(
        outputFile,
        CppFileTypes.ASSEMBLER,
        CppFileTypes.PIC_ASSEMBLER,
        CppFileTypes.PREPROCESSED_C,
        CppFileTypes.PREPROCESSED_CPP,
        CppFileTypes.PIC_PREPROCESSED_C,
        CppFileTypes.PIC_PREPROCESSED_CPP)) {
      buildVariables.addStringVariable(
          CompileBuildVariables.OUTPUT_OBJECT_FILE.getVariableName(),
          realOutputFilePath.getSafePathString());
    }
    buildVariables.addStringVariable(
        CompileBuildVariables.OUTPUT_FILE.getVariableName(),
        realOutputFilePath.getSafePathString());

    // Set dependency_file to enable <object>.d file generation.
    if (dotdFileExecPath != null) {
      buildVariables.addStringVariable(
          CompileBuildVariables.DEPENDENCY_FILE.getVariableName(), dotdFileExecPath);
    }

    if (featureConfiguration.isEnabled(CppRuleClasses.MODULE_MAPS) && cppModuleMap != null) {
      // If the feature is enabled and cppModuleMap is null, we are about to fail during analysis
      // in any case, but don't crash.
      buildVariables.addStringVariable(
          CompileBuildVariables.MODULE_NAME.getVariableName(), cppModuleMap.getName());
      buildVariables.addStringVariable(
          CompileBuildVariables.MODULE_MAP_FILE.getVariableName(),
          cppModuleMap.getArtifact().getExecPathString());
      StringSequenceBuilder sequence = new StringSequenceBuilder();
      for (Artifact artifact : ccCompilationInfo.getDirectModuleMaps()) {
        sequence.addValue(artifact.getExecPathString());
      }
      buildVariables.addCustomBuiltVariable(
          CompileBuildVariables.DEPENDENT_MODULE_MAP_FILES.getVariableName(), sequence);
    }
    if (featureConfiguration.isEnabled(CppRuleClasses.USE_HEADER_MODULES)) {
      // Module inputs will be set later when the action is executed.
      buildVariables.addStringSequenceVariable(
          CompileBuildVariables.MODULE_FILES.getVariableName(), ImmutableSet.of());
    }
    if (featureConfiguration.isEnabled(CppRuleClasses.INCLUDE_PATHS)) {
      buildVariables.addStringSequenceVariable(
          CompileBuildVariables.INCLUDE_PATHS.getVariableName(),
          getSafePathStrings(ccCompilationInfo.getIncludeDirs()));
      buildVariables.addStringSequenceVariable(
          CompileBuildVariables.QUOTE_INCLUDE_PATHS.getVariableName(),
          getSafePathStrings(ccCompilationInfo.getQuoteIncludeDirs()));
      buildVariables.addStringSequenceVariable(
          CompileBuildVariables.SYSTEM_INCLUDE_PATHS.getVariableName(),
          getSafePathStrings(ccCompilationInfo.getSystemIncludeDirs()));
    }

    if (!includes.isEmpty()) {
      buildVariables.addStringSequenceVariable(
          CompileBuildVariables.INCLUDES.getVariableName(), includes);
    }

    if (featureConfiguration.isEnabled(CppRuleClasses.PREPROCESSOR_DEFINES)) {
      ImmutableList<String> defines;
      if (fdoStamp != null) {
        // Stamp FDO builds with FDO subtype string
        defines =
            ImmutableList.<String>builder()
                .addAll(ccCompilationInfo.getDefines())
                .add(CppConfiguration.FDO_STAMP_MACRO + "=\"" + fdoStamp + "\"")
                .build();
      } else {
        defines = ccCompilationInfo.getDefines();
      }

      buildVariables.addStringSequenceVariable(
          CompileBuildVariables.PREPROCESSOR_DEFINES.getVariableName(), defines);
    }

    if (usePic) {
      if (!featureConfiguration.isEnabled(CppRuleClasses.PIC)) {
        ruleContext.ruleError(CcCommon.PIC_CONFIGURATION_ERROR);
      }
      buildVariables.addStringVariable(CompileBuildVariables.PIC.getVariableName(), "");
    }

    if (gcnoFile != null) {
      buildVariables.addStringVariable(
          CompileBuildVariables.GCOV_GCNO_FILE.getVariableName(), gcnoFile.getExecPathString());
    }

    if (dwoFile != null) {
      buildVariables.addStringVariable(
          CompileBuildVariables.PER_OBJECT_DEBUG_INFO_FILE.getVariableName(),
          dwoFile.getExecPathString());
    }

    if (ltoIndexingFile != null) {
      buildVariables.addStringVariable(
          CompileBuildVariables.LTO_INDEXING_BITCODE_FILE.getVariableName(),
          ltoIndexingFile.getExecPathString());
    }

    buildVariables.addAllStringVariables(additionalBuildVariables);
    for (VariablesExtension extension : variablesExtensions) {
      extension.addVariables(buildVariables);
    }

    return buildVariables.build();
  }

  /** Get the safe path strings for a list of paths to use in the build variables. */
  private static ImmutableSet<String> getSafePathStrings(Collection<PathFragment> paths) {
    ImmutableSet.Builder<String> result = ImmutableSet.builder();
    for (PathFragment path : paths) {
      result.add(path.getSafePathString());
    }
    return result.build();
  }

  /**
   * Supplier that computes legacy_compile_flags lazily at the execution phase.
   *
   * <p>Dear friends of the lambda, this method exists to limit the scope of captured variables only
   * to arguments (to prevent accidental capture of enclosing instance which could regress memory).
   */
  private static Supplier<ImmutableList<String>> getLegacyCompileFlagsSupplier(
      CppConfiguration cppConfiguration,
      CcToolchainProvider toolchain,
      String sourceFilename,
      ImmutableSet<String> features) {
    return () -> {
      ImmutableList.Builder<String> legacyCompileFlags = ImmutableList.builder();
      legacyCompileFlags.addAll(
          CppHelper.getCrosstoolCompilerOptions(cppConfiguration, toolchain, features));
      if (CppFileTypes.CPP_SOURCE.matches(sourceFilename)
          || CppFileTypes.CPP_HEADER.matches(sourceFilename)
          || CppFileTypes.CPP_MODULE_MAP.matches(sourceFilename)
          || CppFileTypes.CLIF_INPUT_PROTO.matches(sourceFilename)) {
        legacyCompileFlags.addAll(
            CppHelper.getCrosstoolCxxOptions(cppConfiguration, toolchain, features));
      }
      return legacyCompileFlags.build();
    };
  }

  /**
   * Supplier that computes unfiltered_compile_flags lazily at the execution phase.
   *
   * <p>Dear friends of the lambda, this method exists to limit the scope of captured variables only
   * to arguments (to prevent accidental capture of enclosing instance which could regress memory).
   */
  private static Supplier<ImmutableList<String>> getUnfilteredCompileFlagsSupplier(
      CcToolchainProvider ccToolchain, ImmutableSet<String> features) {
    return () -> ccToolchain.getUnfilteredCompilerOptions(features);
  }
}
