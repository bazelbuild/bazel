// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.objc;

import static com.google.devtools.build.lib.rules.objc.ObjcProvider.DEFINE;
import static com.google.devtools.build.lib.rules.objc.XcodeProductType.LIBRARY_STATIC;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.Platform;
import com.google.devtools.build.lib.rules.cpp.CcLibraryHelper;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Variables.ValueSequence;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Variables.VariablesExtension;
import com.google.devtools.build.lib.rules.cpp.CcToolchainProvider;
import com.google.devtools.build.lib.rules.cpp.CppRuleClasses;
import com.google.devtools.build.lib.rules.cpp.Link.LinkTargetType;
import com.google.devtools.build.lib.rules.cpp.PrecompiledFiles;
import com.google.devtools.build.lib.rules.objc.ObjcCommon.ResourceAttributes;
import java.util.Collection;

/** Implementation for experimental_objc_library. */
public class ExperimentalObjcLibrary implements RuleConfiguredTargetFactory {

  private static final String PCH_FILE_VARIABLE_NAME = "pch_file";
  private static final String FRAMEWORKS_VARIABLE_NAME = "framework_paths";
  private static final String VERSION_MIN_VARIABLE_NAME = "version_min";
  private static final String MODULES_MAPS_DIR_NAME = "module_maps_dir";
  private static final String OBJC_MODULE_CACHE_DIR_NAME = "_objc_module_cache";
  private static final String OBJC_MODULE_CACHE_KEY = "modules_cache_path";
  private static final String OBJC_MODULE_FEATURE_NAME = "use_objc_modules";
  private static final String OBJ_LIST_PATH_VARIABLE_NAME = "obj_list_path";
  private static final String ARCHIVE_PATH_VARIABLE_NAME = "archive_path";
  private static final Iterable<String> ACTIVATED_ACTIONS =
      ImmutableList.of("objc-compile", "objc++-compile", "objc-archive");

  /** Build variable extensions for templating a toolchain for objc builds. */
  static class ObjcVariablesExtension implements VariablesExtension {

    private final RuleContext ruleContext;
    private final ObjcProvider objcProvider;
    private final CompilationArtifacts compilationArtifacts;

    private final AppleConfiguration appleConfiguration;
    private final ObjcConfiguration objcConfiguration;

    public ObjcVariablesExtension(
        RuleContext ruleContext,
        ObjcProvider objcProvider,
        CompilationArtifacts compilationArtifacts) {
      this.ruleContext = ruleContext;
      this.objcProvider = objcProvider;
      this.compilationArtifacts = compilationArtifacts;

      this.appleConfiguration = ruleContext.getFragment(AppleConfiguration.class);
      this.objcConfiguration = ruleContext.getFragment(ObjcConfiguration.class);
    }
  
    @Override
    public void addVariables(CcToolchainFeatures.Variables.Builder builder) {
      addPchVariables(builder);
      addFrameworkVariables(builder);
      addArchVariables(builder);
      if (ObjcCommon.shouldUseObjcModules(ruleContext)) {
        addModuleMapVariables(builder);
      }
      if (isStaticArchive(compilationArtifacts)) {
        addArchiveVariables(builder);
      }
    }

    private void addPchVariables(CcToolchainFeatures.Variables.Builder builder) {
      if (ruleContext.getPrerequisiteArtifact("pch", Mode.TARGET) != null) {
        builder.addVariable(
            PCH_FILE_VARIABLE_NAME,
            ruleContext.getPrerequisiteArtifact("pch", Mode.TARGET).getExecPathString());
      }
    }
  
    private void addFrameworkVariables(CcToolchainFeatures.Variables.Builder builder) {
      ValueSequence.Builder frameworkSequence = new ValueSequence.Builder();
      AppleConfiguration appleConfig = ruleContext.getFragment(AppleConfiguration.class);
      for (String framework : CompilationSupport.commonFrameworkNames(objcProvider, appleConfig)) {
        frameworkSequence.addValue(framework);
      }
      builder.addSequence(FRAMEWORKS_VARIABLE_NAME, frameworkSequence.build());
    }
  
    private void addModuleMapVariables(CcToolchainFeatures.Variables.Builder builder) {
      builder.addVariable(
          MODULES_MAPS_DIR_NAME,
          ObjcRuleClasses.intermediateArtifacts(ruleContext)
              .moduleMap()
              .getArtifact()
              .getExecPath()
              .getParentDirectory()
              .toString());
      builder.addVariable(
          OBJC_MODULE_CACHE_KEY,
          ruleContext.getConfiguration().getGenfilesFragment() + "/" + OBJC_MODULE_CACHE_DIR_NAME);
    }
    
    private void addArchVariables(CcToolchainFeatures.Variables.Builder builder) {
      Platform platform = appleConfiguration.getSingleArchPlatform();
      switch (platform.getType()) {
        case IOS:
          builder.addVariable(
              VERSION_MIN_VARIABLE_NAME, objcConfiguration.getMinimumOs().toString());
          break;
        case WATCHOS:
          builder.addVariable(
              VERSION_MIN_VARIABLE_NAME,
              appleConfiguration.getSdkVersionForPlatform(platform).toString());
          break;
        default: // don't handle MACOS and TVOS
          throw new IllegalArgumentException("Unhandled platform: " + platform);
      }
    }
    
    private void addArchiveVariables(CcToolchainFeatures.Variables.Builder builder) {
      builder.addVariable(
          OBJ_LIST_PATH_VARIABLE_NAME,
          ObjcRuleClasses.intermediateArtifacts(ruleContext).archiveObjList().getExecPathString());
      builder.addVariable(
          ARCHIVE_PATH_VARIABLE_NAME, compilationArtifacts.getArchive().get().getExecPathString());
    }
  }

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {
    validateAttributes(ruleContext);

    CompilationArtifacts compilationArtifacts =
        CompilationSupport.compilationArtifacts(ruleContext);
    CompilationAttributes compilationAttributes =
        CompilationAttributes.Builder.fromRuleContext(ruleContext).build();
    PrecompiledFiles precompiledFiles = new PrecompiledFiles(ruleContext);
    CompilationSupport compilationSupport = new CompilationSupport(ruleContext);
    IntermediateArtifacts intermediateArtifacts =
        ObjcRuleClasses.intermediateArtifacts(ruleContext);

    ObjcCommon common = common(ruleContext, compilationAttributes, compilationArtifacts);

    Collection<Artifact> arcSources = Sets.newHashSet(compilationArtifacts.getSrcs());
    Collection<Artifact> nonArcSources = Sets.newHashSet(compilationArtifacts.getNonArcSrcs());
    Collection<Artifact> privateHdrs = Sets.newHashSet(compilationArtifacts.getPrivateHdrs());
    Collection<Artifact> publicHdrs = Sets.newHashSet(compilationAttributes.hdrs());

    CcLibraryHelper helper =
        new CcLibraryHelper(
                ruleContext,
                new ObjcCppSemantics(common.getObjcProvider()),
                getFeatureConfiguration(ruleContext),
                CcLibraryHelper.SourceCategory.CC_AND_OBJC)
            .addSources(arcSources, ImmutableMap.of("objc_arc", ""))
            .addSources(nonArcSources, ImmutableMap.of("no_objc_arc", ""))
            .addSources(privateHdrs)
            .addDefines(common.getObjcProvider().get(DEFINE))
            .enableCompileProviders()
            .addPublicHeaders(publicHdrs)
            .addPrecompiledFiles(precompiledFiles)
            .addDeps(ruleContext.getPrerequisites("deps", Mode.TARGET))
            .addCopts(compilationSupport.getCompileRuleCopts())
            .addVariableExtension(
                new ObjcVariablesExtension(
                    ruleContext, common.getObjcProvider(), compilationArtifacts));

    if (isStaticArchive(compilationArtifacts)) {
      Artifact objList = intermediateArtifacts.archiveObjList();

      // TODO(b/30783125): Signal the need for this action in the CROSSTOOL.
      compilationSupport.registerObjFilelistAction(
          getObjFiles(compilationArtifacts, intermediateArtifacts), objList);

      helper
          .setLinkType(LinkTargetType.OBJC_ARCHIVE)
          .addLinkActionInput(objList);
    }

    if (ObjcCommon.shouldUseObjcModules(ruleContext)) {
      helper.setCppModuleMap(ObjcRuleClasses.intermediateArtifacts(ruleContext).moduleMap());
    }

    CcLibraryHelper.Info info = helper.build();

    NestedSetBuilder<Artifact> filesToBuild =
        NestedSetBuilder.<Artifact>stableOrder().addAll(common.getCompiledArchive().asSet());

    XcodeProvider.Builder xcodeProviderBuilder = new XcodeProvider.Builder();
    compilationSupport.addXcodeSettings(xcodeProviderBuilder, common);
    new XcodeSupport(ruleContext)
        .addFilesToBuild(filesToBuild)
        .addXcodeSettings(xcodeProviderBuilder, common.getObjcProvider(), LIBRARY_STATIC)
        .addDependencies(xcodeProviderBuilder, new Attribute("bundles", Mode.TARGET))
        .addDependencies(xcodeProviderBuilder, new Attribute("deps", Mode.TARGET))
        .addNonPropagatedDependencies(
            xcodeProviderBuilder, new Attribute("non_propagated_deps", Mode.TARGET))
        .registerActions(xcodeProviderBuilder.build());

    return ObjcRuleClasses.ruleConfiguredTarget(ruleContext, filesToBuild.build())
        .addProvider(ObjcProvider.class, common.getObjcProvider())
        .addProviders(info.getProviders())
        .addProvider(XcodeProvider.class, xcodeProviderBuilder.build())
        .build();
  }

  private FeatureConfiguration getFeatureConfiguration(RuleContext ruleContext) {
    CcToolchainProvider toolchain =
        ruleContext
            .getPrerequisite(":cc_toolchain", Mode.TARGET)
            .getProvider(CcToolchainProvider.class);

    ImmutableList.Builder<String> activatedCrosstoolSelectables =
        ImmutableList.<String>builder().addAll(ACTIVATED_ACTIONS);

    if (ruleContext.getPrerequisiteArtifact("pch", Mode.TARGET) != null) {
      activatedCrosstoolSelectables.add("pch");
    }

    if (ObjcCommon.shouldUseObjcModules(ruleContext)) {
      activatedCrosstoolSelectables.add(OBJC_MODULE_FEATURE_NAME);
    }

    activatedCrosstoolSelectables.addAll(
        ruleContext.getFragment(AppleConfiguration.class).getBitcodeMode().getFeatureNames());

    // We create a module map by default to allow for swift interop.
    activatedCrosstoolSelectables.add(CppRuleClasses.MODULE_MAPS);
    activatedCrosstoolSelectables.add(CppRuleClasses.COMPILE_ACTION_FLAGS_IN_FLAG_SET);
    activatedCrosstoolSelectables.add(CppRuleClasses.DEPENDENCY_FILE);

    return toolchain.getFeatures().getFeatureConfiguration(activatedCrosstoolSelectables.build());
  }

  /**
   * Throws errors or warnings for bad attribute state.
   */
  private void validateAttributes(RuleContext ruleContext) {
    for (String copt : ObjcCommon.getNonCrosstoolCopts(ruleContext)) {
      if (copt.contains("-fmodules-cache-path")) {
        ruleContext.ruleWarning(CompilationSupport.MODULES_CACHE_PATH_WARNING);
      }
    }
  }

  /**
   * Constructs an {@link ObjcCommon} instance based on the attributes of the given rule context.
   */
  private ObjcCommon common(
      RuleContext ruleContext,
      CompilationAttributes compilationAttributes,
      CompilationArtifacts compilationArtifacts) {
    return new ObjcCommon.Builder(ruleContext)
        .setCompilationAttributes(compilationAttributes)
        .setResourceAttributes(new ResourceAttributes(ruleContext))
        .addDefines(ruleContext.getTokenizedStringListAttr("defines"))
        .setCompilationArtifacts(compilationArtifacts)
        .addDeps(ruleContext.getPrerequisites("deps", Mode.TARGET))
        .addRuntimeDeps(ruleContext.getPrerequisites("runtime_deps", Mode.TARGET))
        .setIntermediateArtifacts(ObjcRuleClasses.intermediateArtifacts(ruleContext))
        .build();
  }
  
  private static boolean isStaticArchive(CompilationArtifacts compilationArtifacts) {
    return compilationArtifacts.getArchive().isPresent();
  }

  private ImmutableList<Artifact> getObjFiles(
      CompilationArtifacts compilationArtifacts, IntermediateArtifacts intermediateArtifacts) {
    ImmutableList.Builder<Artifact> result = new ImmutableList.Builder<>();
    for (Artifact sourceFile : compilationArtifacts.getSrcs()) {
      Artifact objFile = intermediateArtifacts.objFile(sourceFile);
      result.add(objFile);
    }
    for (Artifact nonArcSourceFile : compilationArtifacts.getNonArcSrcs()) {
      Artifact objFile = intermediateArtifacts.objFile(nonArcSourceFile);
      result.add(objFile);
    }
    return result.build();
  }
}
