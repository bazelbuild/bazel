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
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RunfilesSupport;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.ApplePlatform;
import com.google.devtools.build.lib.rules.apple.ApplePlatform.PlatformType;
import com.google.devtools.build.lib.rules.apple.XcodeConfig;
import com.google.devtools.build.lib.rules.objc.CompilationSupport.ExtraLinkArgs;
import com.google.devtools.build.lib.rules.objc.ObjcCommon.ResourceAttributes;
import com.google.devtools.build.lib.rules.objc.ReleaseBundlingSupport.LinkedBinary;
import com.google.devtools.build.lib.rules.proto.ProtoSourcesProvider;
import java.util.Map;
import java.util.TreeMap;

/**
 * Implementation for rules that link binaries.
 *
 * @deprecated The native bundling rules have been deprecated. This class will be removed in the
 *     future.
 */
@Deprecated
abstract class BinaryLinkingTargetFactory implements RuleConfiguredTargetFactory {
  /**
   * Indicates whether this binary generates an application bundle. If so, it causes the {@code
   * infoplist} attribute to be read and a bundle to be added to the files-to-build.
   */
  enum HasReleaseBundlingSupport {
    YES,
    NO;
  }

  private final HasReleaseBundlingSupport hasReleaseBundlingSupport;

  protected BinaryLinkingTargetFactory(HasReleaseBundlingSupport hasReleaseBundlingSupport) {
    this.hasReleaseBundlingSupport = hasReleaseBundlingSupport;
  }

  /**
   * Returns extra linker arguments. Default implementation returns empty list. Subclasses can
   * override and customize.
   */
  protected ExtraLinkArgs getExtraLinkArgs(RuleContext ruleContext) {
    return new ExtraLinkArgs();
  }

  @VisibleForTesting
  static final String REQUIRES_AT_LEAST_ONE_LIBRARY_OR_SOURCE_FILE =
      "At least one library dependency or source file is required.";

  @Override
  public final ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {
    ruleContext.ruleWarning(
        "This rule is deprecated. Please use the new Apple build rules "
            + "(https://github.com/bazelbuild/rules_apple) to build Apple targets.");

    Iterable<ObjcProtoProvider> objcProtoProviders =
        ruleContext.getPrerequisites("deps", Mode.TARGET, ObjcProtoProvider.class);

    ProtobufSupport protoSupport =
        new ProtobufSupport(
                ruleContext,
                ruleContext.getConfiguration(),
                ImmutableList.<ProtoSourcesProvider>of(),
                objcProtoProviders,
                ProtobufSupport.getTransitivePortableProtoFilters(objcProtoProviders))
            .registerGenerationActions()
            .registerCompilationActions();

    Optional<ObjcProvider> protosObjcProvider = protoSupport.getObjcProvider();

    ObjcCommon common = common(ruleContext, protosObjcProvider);

    ObjcProvider objcProvider = common.getObjcProvider();
    assertLibraryOrSources(objcProvider, ruleContext);

    IntermediateArtifacts intermediateArtifacts =
        ObjcRuleClasses.intermediateArtifacts(ruleContext);

    NestedSetBuilder<Artifact> filesToBuild =
        NestedSetBuilder.<Artifact>stableOrder()
            .add(intermediateArtifacts.strippedSingleArchitectureBinary());

    new ResourceSupport(ruleContext).validateAttributes();

    ruleContext.assertNoErrors();

    J2ObjcMappingFileProvider j2ObjcMappingFileProvider =
        J2ObjcMappingFileProvider.union(
            ruleContext.getPrerequisites("deps", Mode.TARGET, J2ObjcMappingFileProvider.class));
    J2ObjcEntryClassProvider j2ObjcEntryClassProvider =
        new J2ObjcEntryClassProvider.Builder()
            .addTransitive(
                ruleContext.getPrerequisites("deps", Mode.TARGET, J2ObjcEntryClassProvider.class))
            .build();

    Map<String, NestedSet<Artifact>> outputGroupCollector = new TreeMap<>();
    CompilationSupport compilationSupport =
        new CompilationSupport.Builder()
            .setRuleContext(ruleContext)
            .setOutputGroupCollector(outputGroupCollector)
            .build();

    compilationSupport
        .validateAttributes()
        .registerCompileAndArchiveActions(common)
        .registerFullyLinkAction(
            common.getObjcProvider(),
            ruleContext.getImplicitOutputArtifact(CompilationSupport.FULLY_LINKED_LIB))
        .registerLinkActions(
            objcProvider,
            j2ObjcMappingFileProvider,
            j2ObjcEntryClassProvider,
            getExtraLinkArgs(ruleContext),
            ImmutableList.<Artifact>of(),
            DsymOutputType.APP);

    Optional<XcTestAppProvider> xcTestAppProvider;
    Optional<RunfilesSupport> maybeRunfilesSupport = Optional.absent();
    switch (hasReleaseBundlingSupport) {
      case YES:
        AppleConfiguration appleConfiguration = ruleContext.getFragment(AppleConfiguration.class);
        // TODO(bazel-team): Remove once all bundle users are migrated to ios_application.
        ReleaseBundlingSupport releaseBundlingSupport =
            new ReleaseBundlingSupport(
                ruleContext,
                objcProvider,
                LinkedBinary.LOCAL_AND_DEPENDENCIES,
                ReleaseBundlingSupport.APP_BUNDLE_DIR_FORMAT,
                XcodeConfig.getMinimumOsForPlatformType(ruleContext, PlatformType.IOS),
                appleConfiguration.getSingleArchPlatform());
        releaseBundlingSupport
            .registerActions(DsymOutputType.APP)
            .addFilesToBuild(filesToBuild, Optional.of(DsymOutputType.APP))
            .validateResources()
            .validateAttributes();

        xcTestAppProvider = Optional.of(releaseBundlingSupport.xcTestAppProvider());
        if (appleConfiguration.getMultiArchPlatform(PlatformType.IOS)
            == ApplePlatform.IOS_SIMULATOR) {
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

    RuleConfiguredTargetBuilder targetBuilder =
        ObjcRuleClasses.ruleConfiguredTarget(ruleContext, filesToBuild.build())
            .addNativeDeclaredProvider(objcProvider)
            .addNativeDeclaredProvider(objcProvider)
            .addProvider(
                InstrumentedFilesProvider.class,
                compilationSupport.getInstrumentedFilesProvider(common))
            .addOutputGroups(outputGroupCollector);

    if (xcTestAppProvider.isPresent()) {
      // TODO(bazel-team): Stop exporting an XcTestAppProvider once objc_binary no longer creates an
      // application bundle.
      targetBuilder.addNativeDeclaredProvider(xcTestAppProvider.get());
    }
    if (maybeRunfilesSupport.isPresent()) {
      RunfilesSupport runfilesSupport = maybeRunfilesSupport.get();
      targetBuilder.setRunfilesSupport(runfilesSupport, runfilesSupport.getExecutable());
    }
    configureTarget(targetBuilder, ruleContext);
    return targetBuilder.build();
  }

  private void assertLibraryOrSources(ObjcProvider objcProvider, RuleContext ruleContext)
      throws RuleErrorException {
    if (Iterables.isEmpty(objcProvider.get(LIBRARY)) // Includes sources from this target.
        && Iterables.isEmpty(objcProvider.get(IMPORTED_LIBRARY))) {
      ruleContext.throwWithRuleError(REQUIRES_AT_LEAST_ONE_LIBRARY_OR_SOURCE_FILE);
    }
  }

  private ObjcCommon common(RuleContext ruleContext, Optional<ObjcProvider> protosObjcProvider) {
    IntermediateArtifacts intermediateArtifacts =
        ObjcRuleClasses.intermediateArtifacts(ruleContext);
    CompilationArtifacts compilationArtifacts =
        CompilationSupport.compilationArtifacts(ruleContext);

    ObjcCommon.Builder builder =
        new ObjcCommon.Builder(ruleContext)
            .setCompilationAttributes(
                CompilationAttributes.Builder.fromRuleContext(ruleContext).build())
            .setCompilationArtifacts(compilationArtifacts)
            .setResourceAttributes(new ResourceAttributes(ruleContext))
            .addDefines(ruleContext.getTokenizedStringListAttr("defines"))
            .addDeps(ruleContext.getPrerequisites("deps", Mode.TARGET))
            .addRuntimeDeps(ruleContext.getPrerequisites("runtime_deps", Mode.TARGET))
            .addDeps(ruleContext.getPrerequisites("bundles", Mode.TARGET))
            .addDepObjcProviders(protosObjcProvider.asSet())
            .addNonPropagatedDepObjcProviders(
                ruleContext.getPrerequisites(
                    "non_propagated_deps", Mode.TARGET, ObjcProvider.SKYLARK_CONSTRUCTOR))
            .setIntermediateArtifacts(intermediateArtifacts)
            .setAlwayslink(false)
            .setHasModuleMap()
            .setLinkedBinary(intermediateArtifacts.strippedSingleArchitectureBinary());

    if (ObjcRuleClasses.objcConfiguration(ruleContext).generateDsym()) {
      builder.addDebugArtifacts(DsymOutputType.APP);
    }

    if (ObjcRuleClasses.objcConfiguration(ruleContext).generateLinkmap()) {
      builder.setLinkmapFile(intermediateArtifacts.linkmap());
    }

    return builder.build();
  }

  /**
   * Performs additional configuration of the target. The default implementation does nothing, but
   * subclasses may override it to add logic.
   */
  protected void configureTarget(RuleConfiguredTargetBuilder target, RuleContext ruleContext) {};
}
