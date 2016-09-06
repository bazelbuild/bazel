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
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.IMPORTED_LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.INCLUDE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.INCLUDE_SYSTEM;
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
import com.google.devtools.build.lib.rules.cpp.CcLibraryHelper;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Variables.VariablesExtension;
import com.google.devtools.build.lib.rules.cpp.CcToolchainProvider;
import com.google.devtools.build.lib.rules.cpp.CppLinkAction;
import com.google.devtools.build.lib.rules.cpp.CppLinkActionBuilder;
import com.google.devtools.build.lib.rules.cpp.CppRuleClasses;
import com.google.devtools.build.lib.rules.cpp.Link.LinkStaticness;
import com.google.devtools.build.lib.rules.cpp.Link.LinkTargetType;
import com.google.devtools.build.lib.rules.cpp.PrecompiledFiles;
import com.google.devtools.build.lib.rules.objc.ObjcCommon.ResourceAttributes;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Collection;

/** Implementation for experimental_objc_library. */
public class ExperimentalObjcLibrary implements RuleConfiguredTargetFactory {

  private static final String OBJC_MODULE_FEATURE_NAME = "use_objc_modules";
  private static final Iterable<String> ACTIVATED_ACTIONS =
      ImmutableList.of("objc-compile", "objc++-compile", "objc-archive", "objc-fully-link",
          "assemble", "preprocess-assemble", "c-compile", "c++-compile");

  @Override
  public ConfiguredTarget create(RuleContext ruleContext) 
    throws InterruptedException, RuleErrorException {
    return configureExperimentalObjcLibrary(ruleContext);
  }
  
  /**
   * Returns a configured target using the given context as an experimental_objc_library.
   * 
   * <p>Implemented outside of {@link RuleClass.ConfiguredTargetFactory#create} so as to allow
   * experimental analysis of objc_library targets as experimental_objc_library.
   */
  public static ConfiguredTarget configureExperimentalObjcLibrary(RuleContext ruleContext)
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
    ObjcVariablesExtension variablesExtension =
        new ObjcVariablesExtension(
            ruleContext,
            common.getObjcProvider(),
            compilationArtifacts,
            ruleContext.getImplicitOutputArtifact(CompilationSupport.FULLY_LINKED_LIB),
            intermediateArtifacts,
            ruleContext.getConfiguration());

    FeatureConfiguration featureConfiguration = getFeatureConfiguration(ruleContext);

    Collection<Artifact> arcSources = Sets.newHashSet(compilationArtifacts.getSrcs());
    Collection<Artifact> nonArcSources = Sets.newHashSet(compilationArtifacts.getNonArcSrcs());
    Collection<Artifact> privateHdrs = Sets.newHashSet(compilationArtifacts.getPrivateHdrs());
    Collection<Artifact> publicHdrs = Sets.newHashSet(compilationAttributes.hdrs());

    CcLibraryHelper helper =
        new CcLibraryHelper(
                ruleContext,
                new ObjcCppSemantics(common.getObjcProvider()),
                featureConfiguration,
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
            .addIncludeDirs(common.getObjcProvider().get(INCLUDE))
            .addSystemIncludeDirs(common.getObjcProvider().get(INCLUDE_SYSTEM))
            .addVariableExtension(variablesExtension);

    if (compilationArtifacts.getArchive().isPresent()) {
      registerArchiveAction(
          intermediateArtifacts, compilationSupport, compilationArtifacts, helper);
    }
    registerFullyLinkAction(ruleContext, common, variablesExtension, featureConfiguration);

    if (ObjcCommon.shouldUseObjcModules(ruleContext)) {
      helper.setCppModuleMap(ObjcRuleClasses.intermediateArtifacts(ruleContext).moduleMap());
    }

    CcLibraryHelper.Info info = helper.build();

    NestedSetBuilder<Artifact> filesToBuild =
        NestedSetBuilder.<Artifact>stableOrder().addAll(common.getCompiledArchive().asSet());

    XcodeProvider.Builder xcodeProviderBuilder = new XcodeProvider.Builder();
    compilationSupport.addXcodeSettings(xcodeProviderBuilder, common);
    
    new ResourceSupport(ruleContext)
        .validateAttributes()
        .addXcodeSettings(xcodeProviderBuilder);
    
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
        .addProvider(ObjcProvider.class, common.getObjcProvider())
        .addProvider(XcodeProvider.class, xcodeProviderBuilder.build())
        .build();
  }

  private static FeatureConfiguration getFeatureConfiguration(RuleContext ruleContext) {
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
    activatedCrosstoolSelectables.add(CppRuleClasses.INCLUDE_PATHS);

    return toolchain.getFeatures().getFeatureConfiguration(activatedCrosstoolSelectables.build());
  }

  private static void registerArchiveAction(
      IntermediateArtifacts intermediateArtifacts,
      CompilationSupport compilationSupport,
      CompilationArtifacts compilationArtifacts,
      CcLibraryHelper helper) {
    Artifact objList = intermediateArtifacts.archiveObjList();

    // TODO(b/30783125): Signal the need for this action in the CROSSTOOL.
    compilationSupport.registerObjFilelistAction(
        getObjFiles(compilationArtifacts, intermediateArtifacts), objList);

    helper.setLinkType(LinkTargetType.OBJC_ARCHIVE).addLinkActionInput(objList);
  }

  private static void registerFullyLinkAction(
      RuleContext ruleContext,
      ObjcCommon common,
      VariablesExtension variablesExtension,
      FeatureConfiguration featureConfiguration)
      throws InterruptedException {
    Artifact fullyLinkedArchive =
        ruleContext.getImplicitOutputArtifact(CompilationSupport.FULLY_LINKED_LIB);
    PathFragment labelName = new PathFragment(ruleContext.getLabel().getName());
    String libraryIdentifier =
        ruleContext
            .getPackageDirectory()
            .getRelative(labelName.replaceName("lib" + labelName.getBaseName()))
            .getPathString();
    CppLinkAction fullyLinkAction =
        new CppLinkActionBuilder(ruleContext, fullyLinkedArchive)
            .addActionInputs(common.getObjcProvider().getObjcLibraries())
            .addActionInputs(common.getObjcProvider().getCcLibraries())
            .addActionInputs(common.getObjcProvider().get(IMPORTED_LIBRARY).toSet())
            .setLinkType(LinkTargetType.OBJC_FULLY_LINKED_ARCHIVE)
            .setLinkStaticness(LinkStaticness.FULLY_STATIC)
            .setLibraryIdentifier(libraryIdentifier)
            .addVariablesExtension(variablesExtension)
            .setFeatureConfiguration(featureConfiguration)
            .build();
    ruleContext.registerAction(fullyLinkAction);
  }

  /** Throws errors or warnings for bad attribute state. */
  private static void validateAttributes(RuleContext ruleContext) {
    for (String copt : ObjcCommon.getNonCrosstoolCopts(ruleContext)) {
      if (copt.contains("-fmodules-cache-path")) {
        ruleContext.ruleWarning(CompilationSupport.MODULES_CACHE_PATH_WARNING);
      }
    }
  }

  /**
   * Constructs an {@link ObjcCommon} instance based on the attributes of the given rule context.
   */
  private static ObjcCommon common(
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
 
  private static ImmutableList<Artifact> getObjFiles(
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
