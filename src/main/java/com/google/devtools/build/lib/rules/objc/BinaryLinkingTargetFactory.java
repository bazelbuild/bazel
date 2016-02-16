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

package com.google.devtools.build.lib.rules.objc;

import static com.google.devtools.build.lib.rules.objc.ObjcProvider.IMPORTED_LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.LIBRARY;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RunfilesSupport;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.Platform;
import com.google.devtools.build.lib.rules.cpp.CppCompilationContext;
import com.google.devtools.build.lib.rules.objc.CompilationSupport.ExtraLinkArgs;
import com.google.devtools.build.lib.rules.objc.ObjcCommon.CompilationAttributes;
import com.google.devtools.build.lib.rules.objc.ObjcCommon.ResourceAttributes;
import com.google.devtools.build.lib.rules.objc.ReleaseBundlingSupport.LinkedBinary;
import com.google.devtools.build.lib.rules.test.InstrumentedFilesProvider;

/**
 * Implementation for rules that link binaries.
 */
abstract class BinaryLinkingTargetFactory implements RuleConfiguredTargetFactory {
  /**
   * Indicates whether this binary generates an application bundle. If so, it causes the
   * {@code infoplist} attribute to be read and a bundle to be added to the files-to-build.
   */
  enum HasReleaseBundlingSupport {
    YES, NO;
  }

  private final HasReleaseBundlingSupport hasReleaseBundlingSupport;
  private final XcodeProductType productType;

  protected BinaryLinkingTargetFactory(HasReleaseBundlingSupport hasReleaseBundlingSupport,
      XcodeProductType productType) {
    this.hasReleaseBundlingSupport = hasReleaseBundlingSupport;
    this.productType = productType;
  }

  /**
   * Returns extra linker arguments. Default implementation returns empty list.
   * Subclasses can override and customize.
   */
  protected ExtraLinkArgs getExtraLinkArgs(RuleContext ruleContext) {
    return new ExtraLinkArgs();
  }

  @VisibleForTesting
  static final String REQUIRES_AT_LEAST_ONE_LIBRARY_OR_SOURCE_FILE =
      "At least one library dependency or source file is required.";

  @Override
  public final ConfiguredTarget create(RuleContext ruleContext) throws InterruptedException {
    ObjcCommon common = common(ruleContext);

    ObjcProvider objcProvider = common.getObjcProvider();
    if (!hasLibraryOrSources(objcProvider)) {
      ruleContext.ruleError(REQUIRES_AT_LEAST_ONE_LIBRARY_OR_SOURCE_FILE);
      return null;
    }

    IntermediateArtifacts intermediateArtifacts =
        ObjcRuleClasses.intermediateArtifacts(ruleContext);

    XcodeProvider.Builder xcodeProviderBuilder = new XcodeProvider.Builder();
    NestedSetBuilder<Artifact> filesToBuild =
        NestedSetBuilder.<Artifact>stableOrder()
            .add(intermediateArtifacts.strippedSingleArchitectureBinary());

    new ResourceSupport(ruleContext)
        .validateAttributes()
        .addXcodeSettings(xcodeProviderBuilder);

    if (ruleContext.hasErrors()) {
      return null;
    }

    CompilationSupport compilationSupport =
        new CompilationSupport(ruleContext)
            .registerCompileAndArchiveActions(common)
            .addXcodeSettings(xcodeProviderBuilder, common)
            .registerLinkActions(
                objcProvider, getExtraLinkArgs(ruleContext), ImmutableList.<Artifact>of())
            .validateAttributes();

    if (ruleContext.hasErrors()) {
      return null;
    }

    Optional<XcTestAppProvider> xcTestAppProvider;
    Optional<RunfilesSupport> maybeRunfilesSupport = Optional.absent();
    switch (hasReleaseBundlingSupport) {
      case YES:
        ObjcConfiguration objcConfiguration = ObjcRuleClasses.objcConfiguration(ruleContext);
        AppleConfiguration appleConfiguration = ruleContext.getFragment(AppleConfiguration.class);
        // TODO(bazel-team): Remove once all bundle users are migrated to ios_application.
        ReleaseBundlingSupport releaseBundlingSupport = new ReleaseBundlingSupport(
            ruleContext, objcProvider, LinkedBinary.LOCAL_AND_DEPENDENCIES,
            ReleaseBundlingSupport.APP_BUNDLE_DIR_FORMAT, objcConfiguration.getMinimumOs());
        releaseBundlingSupport
            .registerActions()
            .addXcodeSettings(xcodeProviderBuilder)
            .addFilesToBuild(filesToBuild)
            .validateResources()
            .validateAttributes();

        xcTestAppProvider = Optional.of(releaseBundlingSupport.xcTestAppProvider());
        if (appleConfiguration.getBundlingPlatform() == Platform.IOS_SIMULATOR) {
          Artifact runnerScript = intermediateArtifacts.runnerScript();
          Artifact ipaFile = ruleContext.getImplicitOutputArtifact(ReleaseBundlingSupport.IPA);
          releaseBundlingSupport.registerGenerateRunnerScriptAction(runnerScript, ipaFile);
          maybeRunfilesSupport = Optional.of(releaseBundlingSupport.runfilesSupport(runnerScript));
        }
        break;
      case NO:
        xcTestAppProvider = Optional.absent();
        break;
      default:
        throw new AssertionError();
    }

    XcodeSupport xcodeSupport = new XcodeSupport(ruleContext)
        // TODO(bazel-team): Use LIBRARY_STATIC as parameter instead of APPLICATION once objc_binary
        // no longer creates an application bundle
        .addXcodeSettings(xcodeProviderBuilder, objcProvider, productType)
        .addDependencies(xcodeProviderBuilder, new Attribute("bundles", Mode.TARGET))
        .addDependencies(xcodeProviderBuilder, new Attribute("deps", Mode.TARGET))
        .addNonPropagatedDependencies(
            xcodeProviderBuilder, new Attribute("non_propagated_deps", Mode.TARGET))
        .addFilesToBuild(filesToBuild);

    if (productType != XcodeProductType.LIBRARY_STATIC) {
        xcodeSupport.generateCompanionLibXcodeTarget(xcodeProviderBuilder);
    }
    XcodeProvider xcodeProvider = xcodeProviderBuilder.build();
    xcodeSupport.registerActions(xcodeProvider);

    RuleConfiguredTargetBuilder targetBuilder =
        ObjcRuleClasses.ruleConfiguredTarget(ruleContext, filesToBuild.build())
            .addProvider(XcodeProvider.class, xcodeProvider)
            .addProvider(ObjcProvider.class, objcProvider)
            .addProvider(
                InstrumentedFilesProvider.class,
                compilationSupport.getInstrumentedFilesProvider(common));
    if (xcTestAppProvider.isPresent()) {
      // TODO(bazel-team): Stop exporting an XcTestAppProvider once objc_binary no longer creates an
      // application bundle.
      targetBuilder.addProvider(XcTestAppProvider.class, xcTestAppProvider.get());
    }
    if (maybeRunfilesSupport.isPresent()) {
      RunfilesSupport runfilesSupport = maybeRunfilesSupport.get();
      targetBuilder.setRunfilesSupport(runfilesSupport, runfilesSupport.getExecutable());
    }
    configureTarget(targetBuilder, ruleContext);
    return targetBuilder.build();
  }

  private boolean hasLibraryOrSources(ObjcProvider objcProvider) {
    return !Iterables.isEmpty(objcProvider.get(LIBRARY)) // Includes sources from this target.
        || !Iterables.isEmpty(objcProvider.get(IMPORTED_LIBRARY));
  }

  private ObjcCommon common(RuleContext ruleContext) {
    IntermediateArtifacts intermediateArtifacts =
        ObjcRuleClasses.intermediateArtifacts(ruleContext);
    CompilationArtifacts compilationArtifacts =
        CompilationSupport.compilationArtifacts(ruleContext);

    ObjcCommon.Builder builder =
        new ObjcCommon.Builder(ruleContext)
            .setCompilationAttributes(new CompilationAttributes(ruleContext))
            .setResourceAttributes(new ResourceAttributes(ruleContext))
            .setCompilationArtifacts(compilationArtifacts)
            .addDefines(ruleContext.getTokenizedStringListAttr("defines"))
            .addDepObjcProviders(
                ruleContext.getPrerequisites("deps", Mode.TARGET, ObjcProvider.class))
            .addDepCcHeaderProviders(
                ruleContext.getPrerequisites("deps", Mode.TARGET, CppCompilationContext.class))
            .addDepCcLinkProviders(ruleContext)
            .addDepObjcProviders(
                ruleContext.getPrerequisites("bundles", Mode.TARGET, ObjcProvider.class))
            .addNonPropagatedDepObjcProviders(
                ruleContext.getPrerequisites(
                    "non_propagated_deps", Mode.TARGET, ObjcProvider.class))
            .setIntermediateArtifacts(intermediateArtifacts)
            .setAlwayslink(false)
            .setHasModuleMap()
            .setLinkedBinary(intermediateArtifacts.strippedSingleArchitectureBinary());

    if (ObjcRuleClasses.objcConfiguration(ruleContext).generateDebugSymbols()) {
      builder.setBreakpadFile(intermediateArtifacts.breakpadSym());
    }

    return builder.build();
  }

  /**
   * Performs additional configuration of the target. The default implementation does nothing, but
   * subclasses may override it to add logic.
   */
  protected void configureTarget(RuleConfiguredTargetBuilder target, RuleContext ruleContext) {};
}
