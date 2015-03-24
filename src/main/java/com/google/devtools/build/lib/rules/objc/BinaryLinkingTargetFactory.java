// Copyright 2014 Google Inc. All rights reserved.
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
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RunfilesSupport;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.rules.objc.ObjcActionsBuilder.ExtraLinkArgs;
import com.google.devtools.build.lib.rules.objc.ObjcActionsBuilder.ExtraLinkInputs;
import com.google.devtools.build.lib.rules.objc.ObjcCommon.CompilationAttributes;
import com.google.devtools.build.lib.rules.objc.ObjcCommon.ResourceAttributes;
import com.google.devtools.build.lib.rules.objc.ReleaseBundlingSupport.LinkedBinary;

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
  private final ExtraLinkArgs extraLinkArgs;
  private final XcodeProductType productType;

  protected BinaryLinkingTargetFactory(HasReleaseBundlingSupport hasReleaseBundlingSupport,
      ExtraLinkArgs extraLinkArgs, XcodeProductType productType) {
    this.hasReleaseBundlingSupport = hasReleaseBundlingSupport;
    this.extraLinkArgs = extraLinkArgs;
    this.productType = productType;
  }

  @VisibleForTesting
  static final String REQUIRES_AT_LEAST_ONE_LIBRARY_OR_SOURCE_FILE =
      "At least one library dependency or source file is required.";

  @Override
  public final ConfiguredTarget create(RuleContext ruleContext) throws InterruptedException {
    ObjcCommon common = common(ruleContext);
    OptionsProvider optionsProvider = optionsProvider(ruleContext);

    ObjcProvider objcProvider = common.getObjcProvider();
    if (!hasLibraryOrSources(objcProvider)) {
      ruleContext.ruleError(REQUIRES_AT_LEAST_ONE_LIBRARY_OR_SOURCE_FILE);
      return null;
    }

    IntermediateArtifacts intermediateArtifacts =
        ObjcRuleClasses.intermediateArtifacts(ruleContext);

    XcodeProvider.Builder xcodeProviderBuilder = new XcodeProvider.Builder();
    NestedSetBuilder<Artifact> filesToBuild = NestedSetBuilder.<Artifact>stableOrder()
        .add(intermediateArtifacts.singleArchitectureBinary());

    new CompilationSupport(ruleContext)
        .registerJ2ObjcCompileAndArchiveActions(optionsProvider, objcProvider)
        .registerCompileAndArchiveActions(common, optionsProvider)
        .addXcodeSettings(xcodeProviderBuilder, common, optionsProvider)
        .registerLinkActions(objcProvider, extraLinkArgs, new ExtraLinkInputs())
        .validateAttributes();

    Optional<XcTestAppProvider> xcTestAppProvider;
    Optional<RunfilesSupport> maybeRunfilesSupport = Optional.<RunfilesSupport>absent();
    switch (hasReleaseBundlingSupport) {
      case YES:
        // TODO(bazel-team): Remove once all bundle users are migrated to ios_application.
        ReleaseBundlingSupport releaseBundlingSupport = new ReleaseBundlingSupport(
            ruleContext, objcProvider, optionsProvider,
            LinkedBinary.LOCAL_AND_DEPENDENCIES, ReleaseBundlingSupport.APP_BUNDLE_DIR_FORMAT);
        releaseBundlingSupport
            .registerActions()
            .addXcodeSettings(xcodeProviderBuilder)
            .addFilesToBuild(filesToBuild)
            .validateResources()
            .validateAttributes();
        xcTestAppProvider = Optional.of(releaseBundlingSupport.xcTestAppProvider());
        if (ObjcRuleClasses.objcConfiguration(ruleContext).getPlatform() == Platform.SIMULATOR) {
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

    new ResourceSupport(ruleContext)
        .validateAttributes()
        .addXcodeSettings(xcodeProviderBuilder);

    XcodeSupport xcodeSupport = new XcodeSupport(ruleContext)
        // TODO(bazel-team): Use LIBRARY_STATIC as parameter instead of APPLICATION once objc_binary
        // no longer creates an application bundle
        .addXcodeSettings(xcodeProviderBuilder, objcProvider, productType)
        .addDependencies(xcodeProviderBuilder, new Attribute("bundles", Mode.TARGET))
        .addDependencies(xcodeProviderBuilder, new Attribute("deps", Mode.TARGET))
        .addDependencies(xcodeProviderBuilder, new Attribute("non_propagated_deps", Mode.TARGET))
        .addFilesToBuild(filesToBuild);
    XcodeProvider xcodeProvider = xcodeProviderBuilder.build();
    xcodeSupport.registerActions(xcodeProvider);

    // TODO(bazel-team): Stop exporting an XcTestAppProvider once objc_binary no longer creates an
    // application bundle.
    RuleConfiguredTargetBuilder target = common.configuredTargetBuilder(
        filesToBuild.build(),
        Optional.of(xcodeProvider),
        Optional.of(objcProvider),
        xcTestAppProvider,
        Optional.<J2ObjcSrcsProvider>absent());
    for (RunfilesSupport runfilesSupport : maybeRunfilesSupport.asSet()) {
      target.setRunfilesSupport(runfilesSupport, runfilesSupport.getExecutable());
    }
    return target.build();
  }

  private OptionsProvider optionsProvider(RuleContext ruleContext) {
    OptionsProvider.Builder provider = new OptionsProvider.Builder()
        .addCopts(ruleContext.getTokenizedStringListAttr("copts"))
        .addTransitive(Optional.fromNullable(
            ruleContext.getPrerequisite("options", Mode.TARGET, OptionsProvider.class)));
    if (hasReleaseBundlingSupport == HasReleaseBundlingSupport.YES) {
        provider
            .addInfoplists(ruleContext.getPrerequisiteArtifacts("infoplist", Mode.TARGET).list());
    }
    return provider.build();
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

    return new ObjcCommon.Builder(ruleContext)
        .setCompilationAttributes(new CompilationAttributes(ruleContext))
        .setResourceAttributes(new ResourceAttributes(ruleContext))
        .setCompilationArtifacts(compilationArtifacts)
        .addDefines(ruleContext.getTokenizedStringListAttr("defines"))
        .addDepObjcProviders(ruleContext.getPrerequisites("deps", Mode.TARGET, ObjcProvider.class))
        .addDepObjcProviders(
            ruleContext.getPrerequisites("bundles", Mode.TARGET, ObjcProvider.class))
        .addNonPropagatedDepObjcProviders(
            ruleContext.getPrerequisites("non_propagated_deps", Mode.TARGET, ObjcProvider.class))
        .setIntermediateArtifacts(intermediateArtifacts)
        .setAlwayslink(false)
        .addExtraImportLibraries(ObjcRuleClasses.j2ObjcLibraries(ruleContext))
        .setLinkedBinary(intermediateArtifacts.singleArchitectureBinary())
        .build();
  }
}
