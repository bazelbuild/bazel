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
import com.google.devtools.build.lib.analysis.TransitiveInfoProviderMap;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcLibraryHelper;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcToolchainProvider;
import com.google.devtools.build.lib.rules.cpp.CppLinkAction;
import com.google.devtools.build.lib.rules.cpp.CppLinkActionBuilder;
import com.google.devtools.build.lib.rules.cpp.CppRuleClasses;
import com.google.devtools.build.lib.rules.cpp.Link.LinkStaticness;
import com.google.devtools.build.lib.rules.cpp.Link.LinkTargetType;
import com.google.devtools.build.lib.rules.cpp.PrecompiledFiles;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Collection;

/**
 * Support for rules that compile, archive, and link sources using the c++ rules backend, including
 * the crosstool.
 */
public class CrosstoolSupport {

  private static final String OBJC_MODULE_FEATURE_NAME = "use_objc_modules";
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

  private final RuleContext ruleContext;
  private final ObjcVariablesExtension variablesExtension;
  private final IntermediateArtifacts intermediateArtifacts;
  private final CompilationSupport compilationSupport;
  private final CompilationArtifacts compilationArtifacts;
  private final FeatureConfiguration featureConfiguration;

  /**
   * Creates a new instance of CrosstoolSupport.
   *
   * @param ruleContext the RuleContext for the target in question
   * @param objcProvider the provider to query in populating build variables for templating the
   *     crosstool
   * @return a CrosstoolSupport instance that can register compile and archive actions
   */
  public CrosstoolSupport(RuleContext ruleContext, ObjcProvider objcProvider)
      throws InterruptedException {
    this.ruleContext = ruleContext;
    this.compilationArtifacts = CompilationSupport.compilationArtifacts(ruleContext);
    this.intermediateArtifacts = ObjcRuleClasses.intermediateArtifacts(ruleContext);
    this.compilationSupport = new CompilationSupport(ruleContext);
    this.featureConfiguration = getFeatureConfiguration(ruleContext);
    this.variablesExtension =
        new ObjcVariablesExtension(
            ruleContext,
            objcProvider,
            compilationArtifacts,
            ruleContext.getImplicitOutputArtifact(CompilationSupport.FULLY_LINKED_LIB),
            intermediateArtifacts,
            ruleContext.getConfiguration());
  }

  /**
   * Registers a {@link CppCompileAction} to build objc or objc++ source.
   *
   * @param common the common instance that should be queried in the construction of a compile
   *     action
   * @return providers that should be exported by the calling rule implementation
   */
  public TransitiveInfoProviderMap registerCompileActions(ObjcCommon common)
      throws RuleErrorException, InterruptedException {
    return createCcLibraryHelper(common).build().getProviders();
  }

  /**
   * Registers a {@link CppCompileAction} to build objc or objc++ source as well as a {@link
   * CppLinkAction} to archive the resulting object files.
   *
   * @param common the common instance that should be queried in the construction of a compile and
   *     archive action
   * @return providers that should be exported by the calling rule implementation
   */
  public TransitiveInfoProviderMap registerCompileAndArchiveActions(ObjcCommon common)
      throws RuleErrorException, InterruptedException {
    Artifact objList = intermediateArtifacts.archiveObjList();

    // TODO(b/30783125): Signal the need for this action in the CROSSTOOL.
    compilationSupport.registerObjFilelistAction(
        getObjFiles(compilationArtifacts, intermediateArtifacts), objList);

    return createCcLibraryHelper(common)
        .setLinkType(LinkTargetType.OBJC_ARCHIVE)
        .addLinkActionInput(objList)
        .build()
        .getProviders();
  }

  /**
   * Registers a {@link CppLinkAction} to produce a fully linked archive.
   *
   * @param common the common instance that hsould be queried in the construction of a fully link
   *     action
   */
  public void registerFullyLinkAction(ObjcCommon common) throws InterruptedException {
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
            .add(OBJC_MODULE_FEATURE_NAME)
            .add(CppRuleClasses.MODULE_MAPS)
            .add(CppRuleClasses.COMPILE_ACTION_FLAGS_IN_FLAG_SET)
            .add(CppRuleClasses.DEPENDENCY_FILE)
            .add(CppRuleClasses.INCLUDE_PATHS);

    if (ruleContext.getPrerequisiteArtifact("pch", Mode.TARGET) != null) {
      activatedCrosstoolSelectables.add("pch");
    }
 
    return toolchain.getFeatures().getFeatureConfiguration(activatedCrosstoolSelectables.build());
  }

  private CcLibraryHelper createCcLibraryHelper(ObjcCommon common) {
    CompilationAttributes compilationAttributes =
        CompilationAttributes.Builder.fromRuleContext(ruleContext).build();
    PrecompiledFiles precompiledFiles = new PrecompiledFiles(ruleContext);
    Collection<Artifact> arcSources = Sets.newHashSet(compilationArtifacts.getSrcs());
    Collection<Artifact> nonArcSources = Sets.newHashSet(compilationArtifacts.getNonArcSrcs());
    Collection<Artifact> privateHdrs = Sets.newHashSet(compilationArtifacts.getPrivateHdrs());
    Collection<Artifact> publicHdrs = Sets.newHashSet(compilationAttributes.hdrs());
    Artifact pchHdr = ruleContext.getPrerequisiteArtifact("pch", Mode.TARGET);
    ImmutableList<Artifact> pchHdrList =
        (pchHdr != null) ? ImmutableList.<Artifact>of(pchHdr) : ImmutableList.<Artifact>of();
    return new CcLibraryHelper(
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
        .addPublicHeaders(pchHdrList)
        .addPrecompiledFiles(precompiledFiles)
        .addDeps(ruleContext.getPrerequisites("deps", Mode.TARGET))
        .addCopts(compilationSupport.getCompileRuleCopts())
        .addIncludeDirs(common.getObjcProvider().get(INCLUDE))
        .addCopts(ruleContext.getFragment(ObjcConfiguration.class).getCoptsForCompilationMode())
        .addSystemIncludeDirs(common.getObjcProvider().get(INCLUDE_SYSTEM))
        .setCppModuleMap(intermediateArtifacts.moduleMap())
        .addVariableExtension(variablesExtension);
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
