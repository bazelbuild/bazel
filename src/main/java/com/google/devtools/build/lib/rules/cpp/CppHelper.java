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

import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.packages.BuildType.NODEP_LABEL;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.FailAction;
import com.google.devtools.build.lib.analysis.AliasProvider;
import com.google.devtools.build.lib.analysis.AnalysisUtils;
import com.google.devtools.build.lib.analysis.Expander;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RuleErrorConsumer;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.actions.SymlinkAction;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.CompilationMode;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.cpp.CcLinkingContext.Linkstamp;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.ExpansionException;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.Link.LinkTargetType;
import com.google.devtools.build.lib.server.FailureDetails.FailAction.Code;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;

/**
 * Helper class for functionality shared by cpp related rules.
 *
 * <p>This class can be used only after the loading phase.
 */
public class CppHelper {

  static final PathFragment OBJS = PathFragment.create("_objs");
  static final PathFragment PIC_OBJS = PathFragment.create("_pic_objs");
  static final PathFragment DOTD_FILES = PathFragment.create("_dotd");
  static final PathFragment PIC_DOTD_FILES = PathFragment.create("_pic_dotd");
  static final PathFragment DIA_FILES = PathFragment.create("_dia");
  static final PathFragment PIC_DIA_FILES = PathFragment.create("_pic_dia");

  public static final PathFragment SHARED_NONLTO_BACKEND_ROOT_PREFIX =
      PathFragment.create("shared.nonlto");

  // TODO(bazel-team): should this use Link.SHARED_LIBRARY_FILETYPES?
  public static final FileTypeSet SHARED_LIBRARY_FILETYPES =
      FileTypeSet.of(CppFileTypes.SHARED_LIBRARY, CppFileTypes.VERSIONED_SHARED_LIBRARY);

  /** Base label of the c++ toolchain category. */
  public static final String TOOLCHAIN_TYPE_LABEL = "//tools/cpp:toolchain_type";

  private CppHelper() {
    // prevents construction
  }

  /** Returns the malloc implementation for the given target. */
  public static TransitiveInfoCollection mallocForTarget(
      RuleContext ruleContext, String mallocAttrName) {
    if (ruleContext.getFragment(CppConfiguration.class).customMalloc() != null) {
      return ruleContext.getPrerequisite(":default_malloc");
    } else {
      return ruleContext.getPrerequisite(mallocAttrName);
    }
  }

  public static TransitiveInfoCollection mallocForTarget(RuleContext ruleContext) {
    return mallocForTarget(ruleContext, "malloc");
  }

  /** Tokenizes and expands make variables. */
  public static List<String> expandLinkopts(
      RuleContext ruleContext, String attrName, Iterable<String> values)
      throws InterruptedException {
    List<String> result = new ArrayList<>();
    ImmutableMap.Builder<Label, ImmutableCollection<Artifact>> builder = ImmutableMap.builder();

    if (ruleContext.attributes().has("additional_linker_inputs", LABEL_LIST)) {
      for (TransitiveInfoCollection current :
          ruleContext.getPrerequisites("additional_linker_inputs")) {
        builder.put(
            AliasProvider.getDependencyLabel(current),
            current.getProvider(FileProvider.class).getFilesToBuild().toList());
      }
    }

    Expander expander = ruleContext.getExpander(builder.buildOrThrow()).withDataExecLocations();
    for (String value : values) {
      expander.tokenizeAndExpandMakeVars(result, attrName, value);
    }
    return result;
  }

  /** Returns the linkopts for the rule context. */
  public static ImmutableList<String> getLinkopts(RuleContext ruleContext)
      throws InterruptedException {
    if (ruleContext.attributes().has("linkopts", Type.STRING_LIST)) {
      Iterable<String> linkopts = ruleContext.attributes().get("linkopts", Type.STRING_LIST);
      if (linkopts != null) {
        return ImmutableList.copyOf(expandLinkopts(ruleContext, "linkopts", linkopts));
      }
    }
    return ImmutableList.of();
  }

  /**
   * This almost trivial method looks up the default cc toolchain attribute on the rule context,
   * makes sure that it refers to a rule that has a {@link CcToolchainProvider} (gives an error
   * otherwise), and returns a reference to that {@link CcToolchainProvider}. The method only
   * returns {@code null} if there is no such attribute (this is currently not an error).
   *
   * <p>Be careful to provide explicit attribute name if the rule doesn't store cc_toolchain under
   * the default name.
   */
  @Nullable
  public static CcToolchainProvider getToolchainUsingDefaultCcToolchainAttribute(
      RuleContext ruleContext) throws RuleErrorException {
    if (ruleContext.attributes().has(CcToolchainRule.CC_TOOLCHAIN_DEFAULT_ATTRIBUTE_NAME)) {
      return getToolchain(ruleContext, CcToolchainRule.CC_TOOLCHAIN_DEFAULT_ATTRIBUTE_NAME);
    } else if (ruleContext
        .attributes()
        .has(CcToolchainRule.CC_TOOLCHAIN_DEFAULT_ATTRIBUTE_NAME_FOR_STARLARK)) {
      return getToolchain(
          ruleContext, CcToolchainRule.CC_TOOLCHAIN_DEFAULT_ATTRIBUTE_NAME_FOR_STARLARK);
    }
    return null;
  }

  /**
   * Makes sure that the given info collection has a {@link CcToolchainProvider} (gives an error
   * otherwise), and returns a reference to that {@link CcToolchainProvider}.
   */
  public static CcToolchainProvider getToolchain(RuleContext ruleContext, String toolchainAttribute)
      throws RuleErrorException {
    if (!ruleContext.isAttrDefined(toolchainAttribute, LABEL)) {
      throw ruleContext.throwWithRuleError(
          String.format(
              "INTERNAL BLAZE ERROR: Tried to locate a cc_toolchain via the attribute %s, but it"
                  + " is not defined",
              toolchainAttribute));
    }
    TransitiveInfoCollection dep = ruleContext.getPrerequisite(toolchainAttribute);
    return getToolchain(ruleContext, dep);
  }

  /**
   * Makes sure that the given info collection has a {@link CcToolchainProvider} (gives an error
   * otherwise), and returns a reference to that {@link CcToolchainProvider}. The method never
   * returns {@code null}, even if there is no toolchain.
   */
  public static CcToolchainProvider getToolchain(
      RuleContext ruleContext, TransitiveInfoCollection dep) throws RuleErrorException {
    Label toolchainType = getToolchainTypeFromRuleClass(ruleContext);
    return getToolchain(ruleContext, dep, toolchainType);
  }

  public static CcToolchainProvider getToolchain(
      RuleContext ruleContext, TransitiveInfoCollection dep, Label toolchainType)
      throws RuleErrorException {
    if (toolchainType != null && useToolchainResolution(ruleContext)) {
      return getToolchainFromPlatformConstraints(ruleContext, toolchainType);
    }
    return getToolchainFromLegacyToolchain(ruleContext, dep);
  }

  /** Returns the c++ toolchain type, or null if it is not specified on the rule class. */
  public static Label getToolchainTypeFromRuleClass(RuleContext ruleContext) {
    Label toolchainType;
    // TODO(b/65835260): Remove this conditional once j2objc can learn the toolchain type.
    if (ruleContext
        .attributes()
        .has(CcToolchainRule.CC_TOOLCHAIN_TYPE_ATTRIBUTE_NAME, NODEP_LABEL)) {
      toolchainType =
          ruleContext
              .attributes()
              .get(CcToolchainRule.CC_TOOLCHAIN_TYPE_ATTRIBUTE_NAME, NODEP_LABEL);
    } else if (ruleContext
        .attributes()
        .has(CcToolchainRule.CC_TOOLCHAIN_TYPE_ATTRIBUTE_NAME, LABEL)) {
      toolchainType =
          ruleContext.getPrerequisite(CcToolchainRule.CC_TOOLCHAIN_TYPE_ATTRIBUTE_NAME).getLabel();
    } else {
      toolchainType = null;
    }
    return toolchainType;
  }

  private static CcToolchainProvider getToolchainFromPlatformConstraints(
      RuleContext ruleContext, Label toolchainType) throws RuleErrorException {
    ToolchainInfo toolchainInfo = ruleContext.getToolchainInfo(toolchainType);
    if (toolchainInfo == null) {
      throw ruleContext.throwWithRuleError(
          "Unable to find a CC toolchain using toolchain resolution. Did you properly set"
              + " --platforms?");
    }
    try {
      return (CcToolchainProvider) toolchainInfo.getValue("cc");
    } catch (EvalException e) {
      // There is not actually any reason for toolchainInfo.getValue to throw an exception.
      throw ruleContext.throwWithRuleError(
          "Unexpected eval exception from toolchainInfo.getValue('cc')");
    }
  }

  private static CcToolchainProvider getToolchainFromLegacyToolchain(
      RuleContext ruleContext, TransitiveInfoCollection dep) throws RuleErrorException {
    // TODO(bazel-team): Consider checking this generally at the attribute level.
    if ((dep == null) || (dep.get(CcToolchainProvider.PROVIDER) == null)) {
      throw ruleContext.throwWithRuleError("The selected C++ toolchain is not a cc_toolchain rule");
    }
    return dep.get(CcToolchainProvider.PROVIDER);
  }

  /** Returns the directory where object files are created. */
  public static PathFragment getObjDirectory(Label ruleLabel, boolean siblingRepositoryLayout) {
    return getObjDirectory(ruleLabel, false, siblingRepositoryLayout);
  }

  /** Returns the directory where object files are created. */
  public static PathFragment getObjDirectory(
      Label ruleLabel, boolean usePic, boolean siblingRepositoryLayout) {
    if (usePic) {
      return AnalysisUtils.getUniqueDirectory(ruleLabel, PIC_OBJS, siblingRepositoryLayout);
    } else {
      return AnalysisUtils.getUniqueDirectory(ruleLabel, OBJS, siblingRepositoryLayout);
    }
  }

  /** Returns the directory where dotd files are created. */
  private static PathFragment getDotdDirectory(
      Label ruleLabel, boolean usePic, boolean siblingRepositoryLayout) {
    return AnalysisUtils.getUniqueDirectory(
        ruleLabel, usePic ? PIC_DOTD_FILES : DOTD_FILES, siblingRepositoryLayout);
  }

  /** Returns the directory where serialized diagnostics files are created. */
  private static PathFragment getDiagnosticsDirectory(
      Label ruleLabel, boolean usePic, boolean siblingRepositoryLayout) {
    return AnalysisUtils.getUniqueDirectory(
        ruleLabel, usePic ? PIC_DIA_FILES : DIA_FILES, siblingRepositoryLayout);
  }

  /**
   * Given the output file path, returns the directory where the results of thinlto indexing will be
   * created: output_file.lto/
   */
  public static PathFragment getLtoOutputRootPrefix(PathFragment outputRootRelativePath) {
    return FileSystemUtils.appendExtension(outputRootRelativePath, ".lto");
  }

  /**
   * Given the lto output root directory path, returns the directory where thinlto native object
   * files are created: output_file.lto-obj/
   */
  public static PathFragment getThinLtoNativeObjectDirectoryFromLtoOutputRoot(
      PathFragment ltoOutputRootRelativePath) {
    return FileSystemUtils.appendExtension(ltoOutputRootRelativePath, "-obj");
  }

  public static Artifact getLinkedArtifact(
      Label label,
      ActionConstructionContext actionConstructionContext,
      ArtifactRoot artifactRoot,
      BuildConfigurationValue config,
      LinkTargetType linkType,
      String linkedArtifactNameSuffix,
      PathFragment name) {
    Artifact result = actionConstructionContext.getPackageRelativeArtifact(name, artifactRoot);

    // If the linked artifact is not the linux default, then a FailAction is generated for said
    // linux default to satisfy the requirements of any implicit outputs.
    // TODO(b/30132703): Remove the implicit outputs of cc_library.
    Artifact linuxDefault =
        getLinuxLinkedArtifact(
            label, actionConstructionContext, config, linkType, linkedArtifactNameSuffix);
    if (!result.equals(linuxDefault)) {
      actionConstructionContext.registerAction(
          new FailAction(
              actionConstructionContext.getActionOwner(),
              ImmutableList.of(linuxDefault),
              String.format(
                  "the given toolchain supports creation of %s instead of %s",
                  result.getExecPathString(), linuxDefault.getExecPathString()),
              Code.INCORRECT_TOOLCHAIN));
    }

    return result;
  }

  private static Artifact getLinuxLinkedArtifact(
      Label label,
      ActionConstructionContext actionConstructionContext,
      BuildConfigurationValue config,
      LinkTargetType linkType,
      String linkedArtifactNameSuffix) {
    PathFragment name = PathFragment.create(label.getName());
    if (linkType != LinkTargetType.EXECUTABLE) {
      name =
          name.replaceName(
              "lib"
                  + name.getBaseName()
                  + linkedArtifactNameSuffix
                  + linkType.getPicExtensionWhenApplicable()
                  + linkType.getDefaultExtension());
    }

    return actionConstructionContext.getPackageRelativeArtifact(
        name, config.getBinDirectory(label.getRepository()));
  }

  /**
   * Emits a warning on the rule if there are identical linkstamp artifacts with different {@code
   * CcCompilationContext}s.
   */
  public static void checkLinkstampsUnique(
      RuleErrorConsumer listener, Iterable<Linkstamp> linkstamps) {
    Map<Artifact, NestedSet<Artifact>> result = new LinkedHashMap<>();
    for (Linkstamp pair : linkstamps) {
      Artifact artifact = pair.getArtifact();
      if (result.containsKey(artifact)) {
        listener.ruleWarning(
            "rule inherits the '"
                + artifact.toDetailString()
                + "' linkstamp file from more than one cc_library rule");
      }
      result.put(artifact, pair.getDeclaredIncludeSrcs());
    }
  }

  // TODO(bazel-team): figure out a way to merge these 2 methods. See the Todo in
  // CcCommonConfiguredTarget.noCoptsMatches().

  /** Returns whether binaries must be compiled with position independent code. */
  public static boolean usePicForBinaries(
      CcToolchainProvider toolchain,
      CppConfiguration cppConfiguration,
      FeatureConfiguration featureConfiguration) {
    return cppConfiguration.forcePic()
        || (toolchain.usePicForDynamicLibraries(cppConfiguration, featureConfiguration)
            && (cppConfiguration.getCompilationMode() != CompilationMode.OPT
                || featureConfiguration.isEnabled(CppRuleClasses.PREFER_PIC_FOR_OPT_BINARIES)));
  }

  /**
   * Creates a CppModuleMap object for pure c++ builds. The module map artifact becomes a candidate
   * input to a CppCompileAction.
   */
  public static CppModuleMap createDefaultCppModuleMap(
      ActionConstructionContext actionConstructionContext,
      BuildConfigurationValue configuration,
      Label label) {
    // Create the module map artifact as a genfile.
    Artifact mapFile =
        actionConstructionContext.getPackageRelativeArtifact(
            PathFragment.create(
                label.getName()
                    + Iterables.getOnlyElement(CppFileTypes.CPP_MODULE_MAP.getExtensions())),
            configuration.getGenfilesDirectory(label.getRepository()));
    return new CppModuleMap(mapFile, label.toString());
  }

  /** Returns the FDO build subtype. */
  @Nullable
  public static String getFdoBuildStamp(
      CppConfiguration cppConfiguration,
      FdoContext fdoContext,
      FeatureConfiguration featureConfiguration) {
    FdoContext.BranchFdoProfile branchFdoProfile = fdoContext.getBranchFdoProfile();
    if (branchFdoProfile != null) {
      // If the profile has a .afdo extension and was supplied to the build via the xbinary_fdo
      // flag, then this is a safdo profile.
      if (branchFdoProfile.isAutoFdo() && cppConfiguration.getXFdoProfileLabel() != null) {
        return featureConfiguration.isEnabled(CppRuleClasses.AUTOFDO) ? "SAFDO" : null;
      }
      if (branchFdoProfile.isAutoFdo()) {
        return featureConfiguration.isEnabled(CppRuleClasses.AUTOFDO) ? "AFDO" : null;
      }
      if (branchFdoProfile.isAutoXBinaryFdo()) {
        return featureConfiguration.isEnabled(CppRuleClasses.XBINARYFDO) ? "XFDO" : null;
      }
    }
    if (cppConfiguration.isCSFdo()) {
      return "CSFDO";
    }
    if (cppConfiguration.isFdo()) {
      return "FDO";
    }
    return null;
  }

  /** Creates an action to strip an executable. */
  public static void createStripAction(
      RuleContext ruleContext,
      CcToolchainProvider toolchain,
      CppConfiguration cppConfiguration,
      Artifact input,
      Artifact output,
      FeatureConfiguration featureConfiguration)
      throws RuleErrorException, InterruptedException {
    if (featureConfiguration.isEnabled(CppRuleClasses.NO_STRIPPING)) {
      ruleContext.registerAction(
          SymlinkAction.toArtifact(
              ruleContext.getActionOwner(),
              input,
              output,
              "Symlinking original binary as stripped binary"));
      return;
    }

    if (!featureConfiguration.actionIsConfigured(CppActionNames.STRIP)) {
      ruleContext.ruleError("Expected action_config for 'strip' to be configured.");
      return;
    }

    CcToolchainVariables baseVars;
    try {
      baseVars =
          toolchain.getBuildVariables(
              ruleContext.getStarlarkThread(),
              ruleContext.getConfiguration().getOptions(),
              cppConfiguration);
    } catch (EvalException e) {
      throw new RuleErrorException(e.getMessage());
    }

    CcToolchainVariables variables =
        CcToolchainVariables.builder(baseVars)
            .addStringVariable(
                StripBuildVariables.OUTPUT_FILE.getVariableName(), output.getExecPathString())
            .addStringSequenceVariable(
                StripBuildVariables.STRIPOPTS.getVariableName(), cppConfiguration.getStripOpts())
            .addStringVariable(CcCommon.INPUT_FILE_VARIABLE_NAME, input.getExecPathString())
            .build();
    ImmutableList<String> commandLine =
        getCommandLine(ruleContext, featureConfiguration, variables, CppActionNames.STRIP);
    ImmutableMap.Builder<String, String> executionInfoBuilder = ImmutableMap.builder();
    for (String executionRequirement :
        featureConfiguration.getToolRequirementsForAction(CppActionNames.STRIP)) {
      executionInfoBuilder.put(executionRequirement, "");
    }
    SpawnAction stripAction =
        new SpawnAction.Builder()
            .addInput(input)
            .addTransitiveInputs(toolchain.getStripFiles())
            .addOutput(output)
            .useDefaultShellEnvironment()
            .setExecutable(
                PathFragment.create(
                    featureConfiguration.getToolPathForAction(CppActionNames.STRIP)))
            .setExecutionInfo(executionInfoBuilder.buildOrThrow())
            .setProgressMessage("Stripping %s for %s", output.prettyPrint(), ruleContext.getLabel())
            .setMnemonic("CcStrip")
            .addCommandLine(CustomCommandLine.builder().addAll(commandLine).build())
            .build(ruleContext);
    ruleContext.registerAction(stripAction);
  }

  public static ImmutableList<String> getCommandLine(
      RuleErrorConsumer ruleErrorConsumer,
      FeatureConfiguration featureConfiguration,
      CcToolchainVariables variables,
      String actionName)
      throws RuleErrorException {
    try {
      return ImmutableList.copyOf(featureConfiguration.getCommandLine(actionName, variables));
    } catch (ExpansionException e) {
      throw ruleErrorConsumer.throwWithRuleError(e);
    }
  }

  public static ImmutableMap<String, String> getEnvironmentVariables(
      RuleErrorConsumer ruleErrorConsumer,
      FeatureConfiguration featureConfiguration,
      CcToolchainVariables variables,
      String actionName)
      throws RuleErrorException {
    try {
      return featureConfiguration.getEnvironmentVariables(actionName, variables);
    } catch (ExpansionException e) {
      throw ruleErrorConsumer.throwWithRuleError(e);
    }
  }

  static Artifact getCompileOutputArtifact(
      ActionConstructionContext actionConstructionContext,
      Label label,
      String outputName,
      BuildConfigurationValue config) {
    PathFragment objectDir = getObjDirectory(label, config.isSiblingRepositoryLayout());
    return actionConstructionContext.getDerivedArtifact(
        objectDir.getRelative(outputName), config.getBinDirectory(label.getRepository()));
  }

  /** Returns the corresponding compiled TreeArtifact given the source TreeArtifact. */
  public static SpecialArtifact getCompileOutputTreeArtifact(
      ActionConstructionContext actionConstructionContext,
      Label label,
      Artifact sourceTreeArtifact,
      String outputName,
      boolean usePic) {
    return actionConstructionContext.getTreeArtifact(
        getObjDirectory(
                label,
                usePic,
                actionConstructionContext.getConfiguration().isSiblingRepositoryLayout())
            .getRelative(outputName),
        sourceTreeArtifact.getRoot());
  }

  /** Returns the corresponding dotd files TreeArtifact given the source TreeArtifact. */
  public static SpecialArtifact getDotdOutputTreeArtifact(
      ActionConstructionContext actionConstructionContext,
      Label label,
      Artifact sourceTreeArtifact,
      String outputName,
      boolean usePic) {
    return actionConstructionContext.getTreeArtifact(
        getDotdDirectory(
                label,
                usePic,
                actionConstructionContext.getConfiguration().isSiblingRepositoryLayout())
            .getRelative(outputName),
        sourceTreeArtifact.getRoot());
  }

  /**
   * Returns the corresponding serialized diagnostics files TreeArtifact given the source
   * TreeArtifact.
   */
  public static SpecialArtifact getDiagnosticsOutputTreeArtifact(
      ActionConstructionContext actionConstructionContext,
      Label label,
      Artifact sourceTreeArtifact,
      String outputName,
      boolean usePic) {
    return actionConstructionContext.getTreeArtifact(
        getDiagnosticsDirectory(
                label,
                usePic,
                actionConstructionContext.getConfiguration().isSiblingRepositoryLayout())
            .getRelative(outputName),
        sourceTreeArtifact.getRoot());
  }

  public static String getArtifactNameForCategory(
      CcToolchainProvider toolchain,
      ArtifactCategory category,
      String outputName)
      throws RuleErrorException {
    return toolchain.getFeatures().getArtifactNameForCategory(category, outputName);
  }

  static String getDotdFileName(
      CcToolchainProvider toolchain,
      ArtifactCategory outputCategory,
      String outputName)
      throws RuleErrorException {
    String baseName =
        outputCategory == ArtifactCategory.OBJECT_FILE
                || outputCategory == ArtifactCategory.PROCESSED_HEADER
            ? outputName
            : getArtifactNameForCategory(toolchain, outputCategory, outputName);

    return getArtifactNameForCategory(toolchain, ArtifactCategory.INCLUDED_FILE_LIST, baseName);
  }

  static String getDiagnosticsFileName(
      CcToolchainProvider toolchain, ArtifactCategory outputCategory, String outputName)
      throws RuleErrorException {
    String baseName =
        outputCategory == ArtifactCategory.OBJECT_FILE
                || outputCategory == ArtifactCategory.PROCESSED_HEADER
            ? outputName
            : getArtifactNameForCategory(toolchain, outputCategory, outputName);

    return getArtifactNameForCategory(
        toolchain, ArtifactCategory.SERIALIZED_DIAGNOSTICS_FILE, baseName);
  }

  /**
   * Returns true if the build implied by the given config and toolchain uses --start-lib/--end-lib
   * ld options.
   */
  public static boolean useStartEndLib(
      CppConfiguration config,
      CcToolchainProvider toolchain,
      FeatureConfiguration featureConfiguration) {
    return config.startEndLibIsRequested() && toolchain.supportsStartEndLib(featureConfiguration);
  }

  /**
   * Returns the type of archives being used by the build implied by the given config and toolchain.
   */
  public static Link.ArchiveType getArchiveType(
      CppConfiguration config,
      CcToolchainProvider toolchain,
      FeatureConfiguration featureConfiguration) {
    return useStartEndLib(config, toolchain, featureConfiguration)
        ? Link.ArchiveType.START_END_LIB
        : Link.ArchiveType.REGULAR;
  }

  /**
   * Returns true if interface shared objects should be used in the build implied by the given
   * cppConfiguration and toolchain.
   */
  public static boolean useInterfaceSharedLibraries(
      CppConfiguration cppConfiguration,
      CcToolchainProvider toolchain,
      FeatureConfiguration featureConfiguration) {
    return toolchain.supportsInterfaceSharedLibraries(featureConfiguration)
        && cppConfiguration.getUseInterfaceSharedLibraries();
  }

  static boolean useToolchainResolution(RuleContext ruleContext) {
    CppOptions cppOptions =
        Preconditions.checkNotNull(
            ruleContext.getConfiguration().getOptions().get(CppOptions.class));

    return cppOptions.enableCcToolchainResolution;
  }
}
