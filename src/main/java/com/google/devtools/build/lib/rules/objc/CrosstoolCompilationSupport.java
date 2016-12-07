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
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.DYNAMIC_FRAMEWORK_FILE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.IMPORTED_LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.INCLUDE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.INCLUDE_SYSTEM;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.STATIC_FRAMEWORK_FILE;

import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcLibraryHelper;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Variables.VariablesExtension;
import com.google.devtools.build.lib.rules.cpp.CcToolchainProvider;
import com.google.devtools.build.lib.rules.cpp.CppCompileAction;
import com.google.devtools.build.lib.rules.cpp.CppHelper;
import com.google.devtools.build.lib.rules.cpp.CppLinkAction;
import com.google.devtools.build.lib.rules.cpp.CppLinkActionBuilder;
import com.google.devtools.build.lib.rules.cpp.CppRuleClasses;
import com.google.devtools.build.lib.rules.cpp.Link.LinkStaticness;
import com.google.devtools.build.lib.rules.cpp.Link.LinkTargetType;
import com.google.devtools.build.lib.rules.cpp.PrecompiledFiles;
import com.google.devtools.build.lib.rules.objc.ObjcProvider.Flag;
import com.google.devtools.build.lib.rules.objc.ObjcVariablesExtension.VariableCategory;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Collection;

/**
 * Constructs command lines for objc compilation, archiving, and linking. Uses the crosstool
 * infrastructure to register {@link CppCompileAction} and {@link CppLinkAction} instances, making
 * use of a provided toolchain.
 *
 * <p>TODO(b/28403953): Deprecate LegacyCompilationSupport in favor of this implementation for all
 * objc rules.
 */
public class CrosstoolCompilationSupport extends CompilationSupport {

  private static final String OBJC_MODULE_FEATURE_NAME = "use_objc_modules";
  private static final String NO_ENABLE_MODULES_FEATURE_NAME = "no_enable_modules";
  private static final ImmutableList<String> ACTIVATED_ACTIONS =
      ImmutableList.of(
          "objc-compile",
          "objc++-compile",
          "objc-archive",
          "objc-fully-link",
          "objc-executable",
          "objc++-executable",
          "assemble",
          "preprocess-assemble",
          "c-compile",
          "c++-compile");

  private CompilationArtifacts compilationArtifacts;

  /**
   * Creates a new CompilationSupport instance that uses the c++ rule backend
   *
   * @param ruleContext the RuleContext for the calling target
   */
  public CrosstoolCompilationSupport(RuleContext ruleContext) {
    super(
        ruleContext,
        ruleContext.getConfiguration(),
        ObjcRuleClasses.intermediateArtifacts(ruleContext),
        CompilationAttributes.Builder.fromRuleContext(ruleContext).build());
    this.compilationArtifacts = compilationArtifacts(ruleContext);
  }
 
  @Override
  CompilationSupport registerCompileAndArchiveActions(
      ObjcCommon common, ExtraCompileArgs extraCompileArgs, Iterable<PathFragment> priorityHeaders)
      throws RuleErrorException, InterruptedException {

    ObjcVariablesExtension.Builder extension = new ObjcVariablesExtension.Builder()
        .setRuleContext(ruleContext)
        .setObjcProvider(common.getObjcProvider())
        .setCompilationArtifacts(compilationArtifacts)
        .setIntermediateArtifacts(intermediateArtifacts)
        .setConfiguration(ruleContext.getConfiguration());
    CcLibraryHelper helper;
    
    if (compilationArtifacts.getArchive().isPresent()) {
      Artifact objList = intermediateArtifacts.archiveObjList();

      // TODO(b/30783125): Signal the need for this action in the CROSSTOOL.
      registerObjFilelistAction(getObjFiles(compilationArtifacts, intermediateArtifacts), objList);

      extension.addVariableCategory(VariableCategory.ARCHIVE_VARIABLES);
      
      helper = createCcLibraryHelper(common, extension.build())
          .setLinkType(LinkTargetType.OBJC_ARCHIVE)
          .addLinkActionInput(objList);
    } else {
      helper = createCcLibraryHelper(common, extension.build());
    }
    
    helper.build();

    return this;
  }

  @Override
  protected CompilationSupport registerFullyLinkAction(
      ObjcProvider objcProvider, Iterable<Artifact> inputArtifacts, Artifact outputArchive)
      throws InterruptedException {
    Artifact fullyLinkedArchive =
        ruleContext.getImplicitOutputArtifact(CompilationSupport.FULLY_LINKED_LIB);
    PathFragment labelName = new PathFragment(ruleContext.getLabel().getName());
    String libraryIdentifier =
        ruleContext
            .getPackageDirectory()
            .getRelative(labelName.replaceName("lib" + labelName.getBaseName()))
            .getPathString();
    ObjcVariablesExtension extension = new ObjcVariablesExtension.Builder()
        .setRuleContext(ruleContext)
        .setObjcProvider(objcProvider)
        .setConfiguration(ruleContext.getConfiguration())
        .setIntermediateArtifacts(intermediateArtifacts)
        .setFullyLinkArchive(
            ruleContext.getImplicitOutputArtifact(CompilationSupport.FULLY_LINKED_LIB))
        .addVariableCategory(VariableCategory.FULLY_LINK_VARIABLES)
        .build();
    
    CppLinkAction fullyLinkAction =
        new CppLinkActionBuilder(ruleContext, fullyLinkedArchive)
            .addActionInputs(objcProvider.getObjcLibraries())
            .addActionInputs(objcProvider.getCcLibraries())
            .addActionInputs(objcProvider.get(IMPORTED_LIBRARY).toSet())
            .setCrosstoolInputs(CppHelper.getToolchain(ruleContext).getLink())
            .setLinkType(LinkTargetType.OBJC_FULLY_LINKED_ARCHIVE)
            .setLinkStaticness(LinkStaticness.FULLY_STATIC)
            .setLibraryIdentifier(libraryIdentifier)
            .addVariablesExtension(extension)
            .setFeatureConfiguration(getFeatureConfiguration(ruleContext))
            .build();
    ruleContext.registerAction(fullyLinkAction);

    return this;
  }

  @Override
  CompilationSupport registerLinkActions(
      ObjcProvider objcProvider,
      J2ObjcMappingFileProvider j2ObjcMappingFileProvider,
      J2ObjcEntryClassProvider j2ObjcEntryClassProvider,
      ExtraLinkArgs extraLinkArgs,
      Iterable<Artifact> extraLinkInputs,
      DsymOutputType dsymOutputType) throws InterruptedException {
    Iterable<Artifact> prunedJ2ObjcArchives =
        computeAndStripPrunedJ2ObjcArchives(
            j2ObjcEntryClassProvider, j2ObjcMappingFileProvider, objcProvider);
    ImmutableList<Artifact> bazelBuiltLibraries =
        Iterables.isEmpty(prunedJ2ObjcArchives)
            ? objcProvider.getObjcLibraries()
            : substituteJ2ObjcPrunedLibraries(objcProvider);

    Artifact inputFileList = intermediateArtifacts.linkerObjList();
    ImmutableSet<Artifact> forceLinkArtifacts = getForceLoadArtifacts(objcProvider);

    Iterable<Artifact> objFiles =
        Iterables.concat(
            bazelBuiltLibraries, objcProvider.get(IMPORTED_LIBRARY), objcProvider.getCcLibraries());
    // Clang loads archives specified in filelists and also specified as -force_load twice,
    // resulting in duplicate symbol errors unless they are deduped.
    objFiles = Iterables.filter(objFiles, Predicates.not(Predicates.in(forceLinkArtifacts)));

    registerObjFilelistAction(objFiles, inputFileList);

    LinkTargetType linkType = (objcProvider.is(Flag.USES_CPP))
        ? LinkTargetType.OBJCPP_EXECUTABLE
        : LinkTargetType.OBJC_EXECUTABLE;
    
    ObjcVariablesExtension extension = new ObjcVariablesExtension.Builder()
        .setRuleContext(ruleContext)
        .setObjcProvider(objcProvider)
        .setConfiguration(ruleContext.getConfiguration())
        .setIntermediateArtifacts(intermediateArtifacts)
        .setFrameworkNames(frameworkNames(objcProvider))
        .setLibraryNames(libraryNames(objcProvider))
        .setForceLoadArtifacts(getForceLoadArtifacts(objcProvider))
        .setAttributeLinkopts(attributes.linkopts())
        .addVariableCategory(VariableCategory.EXECUTABLE_LINKING_VARIABLES)
        .build();
   
    Artifact binaryToLink = getBinaryToLink();
    CppLinkAction executableLinkAction =
        new CppLinkActionBuilder(ruleContext, binaryToLink)
            .setMnemonic("ObjcLink")
            .addActionInputs(bazelBuiltLibraries)
            .addActionInputs(objcProvider.getCcLibraries())
            .addTransitiveActionInputs(objcProvider.get(IMPORTED_LIBRARY))
            .addTransitiveActionInputs(objcProvider.get(STATIC_FRAMEWORK_FILE))
            .addTransitiveActionInputs(objcProvider.get(DYNAMIC_FRAMEWORK_FILE))
            .setCrosstoolInputs(CppHelper.getToolchain(ruleContext).getLink())
            .addActionInputs(prunedJ2ObjcArchives)
            .addActionInput(inputFileList)
            .setLinkType(linkType)
            .setLinkStaticness(LinkStaticness.FULLY_STATIC)
            .addVariablesExtension(extension)
            .setFeatureConfiguration(getFeatureConfiguration(ruleContext))
            .build();
    ruleContext.registerAction(executableLinkAction);    
    
    return this;
  }

  @Override
  CompilationSupport validateAttributes() throws RuleErrorException {
    super.validateAttributes();

    FeatureConfiguration featureConfiguration = ruleContext
        .getPrerequisite(":cc_toolchain", Mode.TARGET)
        .getProvider(CcToolchainProvider.class)
        .getFeatures()
        .getFeatureConfiguration(ACTIVATED_ACTIONS);
    for (String action : ACTIVATED_ACTIONS) {
      if (!featureConfiguration.actionIsConfigured(action)) {
        ruleContext.ruleError(
            String.format("Toolchain derived from CROSSTOOL for this build does not support objc."
                + " Missing action %s.", action));
      }
    }
    return this;
  }

  private CcLibraryHelper createCcLibraryHelper(ObjcCommon common, VariablesExtension extension) {
    PrecompiledFiles precompiledFiles = new PrecompiledFiles(ruleContext);
    Collection<Artifact> arcSources = Sets.newHashSet(compilationArtifacts.getSrcs());
    Collection<Artifact> nonArcSources = Sets.newHashSet(compilationArtifacts.getNonArcSrcs());
    Collection<Artifact> privateHdrs = Sets.newHashSet(compilationArtifacts.getPrivateHdrs());
    Collection<Artifact> publicHdrs = Sets.newHashSet(attributes.hdrs());
    Artifact pchHdr = ruleContext.getPrerequisiteArtifact("pch", Mode.TARGET);
    ImmutableList<Artifact> pchHdrList =
        (pchHdr != null) ? ImmutableList.<Artifact>of(pchHdr) : ImmutableList.<Artifact>of();
    return new CcLibraryHelper(
            ruleContext,
            new ObjcCppSemantics(
                common.getObjcProvider(), ruleContext.getFragment(ObjcConfiguration.class)),
            getFeatureConfiguration(ruleContext),
            CcLibraryHelper.SourceCategory.CC_AND_OBJC)
        .addSources(arcSources, ImmutableMap.of("objc_arc", ""))
        .addSources(nonArcSources, ImmutableMap.of("no_objc_arc", ""))
        .addSources(privateHdrs)
        .addDefines(common.getObjcProvider().get(DEFINE))
        .enableCompileProviders()
        .addPublicHeaders(publicHdrs)
        .addPublicHeaders(pchHdrList)
        .addPrecompiledFiles(precompiledFiles)
        .addDeps(ruleContext.getPrerequisites("deps", Mode.TARGET))
        .addCopts(getCompileRuleCopts())
        .addIncludeDirs(common.getObjcProvider().get(INCLUDE))
        .addCopts(ruleContext.getFragment(ObjcConfiguration.class).getCoptsForCompilationMode())
        .addSystemIncludeDirs(common.getObjcProvider().get(INCLUDE_SYSTEM))
        .setCppModuleMap(intermediateArtifacts.moduleMap())
        .addVariableExtension(extension);
  }

  private static FeatureConfiguration getFeatureConfiguration(RuleContext ruleContext) {
    CcToolchainProvider toolchain =
        ruleContext
            .getPrerequisite(":cc_toolchain", Mode.TARGET)
            .getProvider(CcToolchainProvider.class);

    ImmutableList.Builder<String> activatedCrosstoolSelectables =
        ImmutableList.<String>builder()
            .addAll(ACTIVATED_ACTIONS)
            .addAll(
                ruleContext
                    .getFragment(AppleConfiguration.class)
                    .getBitcodeMode()
                    .getFeatureNames())
            // We create a module map by default to allow for Swift interop.
            .add(CppRuleClasses.MODULE_MAPS)
            .add(CppRuleClasses.COMPILE_ACTION_FLAGS_IN_FLAG_SET)
            .add(CppRuleClasses.DEPENDENCY_FILE)
            .add(CppRuleClasses.INCLUDE_PATHS);

    if (ruleContext.getConfiguration().getFragment(ObjcConfiguration.class).moduleMapsEnabled()) {
      activatedCrosstoolSelectables.add(OBJC_MODULE_FEATURE_NAME);
    }
    if (!CompilationAttributes.Builder.fromRuleContext(ruleContext).build().enableModules()) {
      activatedCrosstoolSelectables.add(NO_ENABLE_MODULES_FEATURE_NAME);
    } 

    if (ruleContext.getPrerequisiteArtifact("pch", Mode.TARGET) != null) {
      activatedCrosstoolSelectables.add("pch");
    }

    return toolchain.getFeatures().getFeatureConfiguration(activatedCrosstoolSelectables.build());
  }

  private static ImmutableList<Artifact> getObjFiles(
      CompilationArtifacts compilationArtifacts, IntermediateArtifacts intermediateArtifacts) {
    ImmutableList.Builder<Artifact> result = new ImmutableList.Builder<>();
    for (Artifact sourceFile : compilationArtifacts.getSrcs()) {
      result.add(intermediateArtifacts.objFile(sourceFile));
    }
    for (Artifact nonArcSourceFile : compilationArtifacts.getNonArcSrcs()) {
      result.add(intermediateArtifacts.objFile(nonArcSourceFile));
    }
    return result.build();
  }
}
