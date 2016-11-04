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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
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
import com.google.devtools.build.lib.rules.cpp.CppLinkAction;
import com.google.devtools.build.lib.rules.cpp.CppLinkActionBuilder;
import com.google.devtools.build.lib.rules.cpp.CppRuleClasses;
import com.google.devtools.build.lib.rules.cpp.Link.LinkStaticness;
import com.google.devtools.build.lib.rules.cpp.Link.LinkTargetType;
import com.google.devtools.build.lib.rules.cpp.PrecompiledFiles;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Collection;

/**
 * Constructs command lines for objc compilation, archiving, and linking.  Uses the crosstool
 * infrastructure to register {@link CppCompileAction} and {@link CppLinkAction} instances,
 * making use of a provided toolchain.
 * 
 * TODO(b/28403953): Deprecate LegacyCompilationSupport in favor of this implementation for all
 * objc rules.
 */
public class CrosstoolCompilationSupport extends CompilationSupport {

  private static final String OBJC_MODULE_FEATURE_NAME = "use_objc_modules";
  private static final String NO_ENABLE_MODULES_FEATURE_NAME = "no_enable_modules";
  private static final Iterable<String> ACTIVATED_ACTIONS =
      ImmutableList.of(
          "objc-compile",
          "objc++-compile",
          "objc-archive",
          "objc-fully-link",
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

    CcLibraryHelper helper = createCcLibraryHelper(common);

    if (compilationArtifacts.getArchive().isPresent()) {
      Artifact objList = intermediateArtifacts.archiveObjList();

      // TODO(b/30783125): Signal the need for this action in the CROSSTOOL.
      registerObjFilelistAction(getObjFiles(compilationArtifacts, intermediateArtifacts), objList);

      helper.setLinkType(LinkTargetType.OBJC_ARCHIVE).addLinkActionInput(objList);
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
    CppLinkAction fullyLinkAction =
        new CppLinkActionBuilder(ruleContext, fullyLinkedArchive)
            .addActionInputs(objcProvider.getObjcLibraries())
            .addActionInputs(objcProvider.getCcLibraries())
            .addActionInputs(objcProvider.get(IMPORTED_LIBRARY).toSet())
            .setLinkType(LinkTargetType.OBJC_FULLY_LINKED_ARCHIVE)
            .setLinkStaticness(LinkStaticness.FULLY_STATIC)
            .setLibraryIdentifier(libraryIdentifier)
            .addVariablesExtension(getVariablesExtension(objcProvider))
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
      DsymOutputType dsymOutputType) {
    // TODO(b/29582284): Implement the link action in the objc crosstool.
    throw new UnsupportedOperationException();
  }

  private CcLibraryHelper createCcLibraryHelper(ObjcCommon common) throws InterruptedException {
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
        .addVariableExtension(getVariablesExtension(common.getObjcProvider()));
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
            // We create a module map by default to allow for swift interop.
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

  private VariablesExtension getVariablesExtension(ObjcProvider objcProvider)
      throws InterruptedException {
    return new ObjcVariablesExtension(
        ruleContext,
        objcProvider,
        compilationArtifacts,
        ruleContext.getImplicitOutputArtifact(CompilationSupport.FULLY_LINKED_LIB),
        intermediateArtifacts,
        ruleContext.getConfiguration());
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
