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

import static com.google.devtools.build.lib.rules.objc.XcodeProductType.LIBRARY_STATIC;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.rules.cpp.CcLinkParamsProvider;
import com.google.devtools.build.lib.rules.cpp.CppCompilationContext;
import com.google.devtools.build.lib.rules.objc.ObjcCommon.CompilationAttributes;
import com.google.devtools.build.lib.rules.objc.ObjcCommon.ResourceAttributes;
import com.google.devtools.build.lib.rules.test.InstrumentedFilesProvider;
import com.google.devtools.build.lib.syntax.Type;

/**
 * Implementation for {@code objc_library}.
 */
public class ObjcLibrary implements RuleConfiguredTargetFactory {

  /**
   * An {@link IterableWrapper} containing extra library {@link Artifact}s to be linked into the
   * final ObjC application bundle.
   */
  static final class ExtraImportLibraries extends IterableWrapper<Artifact> {

    ExtraImportLibraries(Artifact... extraImportLibraries) {
      super(extraImportLibraries);
    }

    ExtraImportLibraries(Iterable<Artifact> extraImportLibraries) {
      super(extraImportLibraries);
    }
  }

  /**
   * Constructs an {@link ObjcCommon} instance based on the attributes of the given rule. The rule
   * should inherit from {@link ObjcLibraryRule}..
   */
  static ObjcCommon common(RuleContext ruleContext, Iterable<SdkFramework> extraSdkFrameworks,
      boolean alwayslink, ExtraImportLibraries extraImportLibraries,
      Iterable<ObjcProvider> extraDepObjcProviders) {
    CompilationArtifacts compilationArtifacts =
        CompilationSupport.compilationArtifacts(ruleContext);

    return new ObjcCommon.Builder(ruleContext)
        .setCompilationAttributes(new CompilationAttributes(ruleContext))
        .setResourceAttributes(new ResourceAttributes(ruleContext))
        .addExtraSdkFrameworks(extraSdkFrameworks)
        .addDefines(ruleContext.getTokenizedStringListAttr("defines"))
        .setCompilationArtifacts(compilationArtifacts)
        .addDepObjcProviders(ruleContext.getPrerequisites("deps", Mode.TARGET, ObjcProvider.class))
        .addDepObjcProviders(
            ruleContext.getPrerequisites("bundles", Mode.TARGET, ObjcProvider.class))
        .addNonPropagatedDepObjcProviders(
            ruleContext.getPrerequisites("non_propagated_deps", Mode.TARGET, ObjcProvider.class))
        .addDepCcHeaderProviders(
            ruleContext.getPrerequisites("deps", Mode.TARGET, CppCompilationContext.class))
        .addDepCcLinkProviders(
            ruleContext.getPrerequisites("deps", Mode.TARGET, CcLinkParamsProvider.class))
        .setIntermediateArtifacts(ObjcRuleClasses.intermediateArtifacts(ruleContext))
        .setAlwayslink(alwayslink)
        .setHasModuleMap()
        .addExtraImportLibraries(extraImportLibraries)
        .addDepObjcProviders(extraDepObjcProviders)
        .build();
  }

  @Override
  public ConfiguredTarget create(RuleContext ruleContext) throws InterruptedException {
    ObjcCommon common = common(
        ruleContext, ImmutableList.<SdkFramework>of(),
        ruleContext.attributes().get("alwayslink", Type.BOOLEAN), new ExtraImportLibraries(),
        ImmutableList.<ObjcProvider>of());

    XcodeProvider.Builder xcodeProviderBuilder = new XcodeProvider.Builder();
    NestedSetBuilder<Artifact> filesToBuild = NestedSetBuilder.<Artifact>stableOrder()
        .addAll(common.getCompiledArchive().asSet());

    CompilationSupport compilationSupport =
        new CompilationSupport(ruleContext)
            .registerCompileAndArchiveActions(common)
            .addXcodeSettings(xcodeProviderBuilder, common)
            .validateAttributes();

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
        .addProvider(XcodeProvider.class, xcodeProviderBuilder.build())
        .addProvider(ObjcProvider.class, common.getObjcProvider())
        .addProvider(J2ObjcSrcsProvider.class, J2ObjcSrcsProvider.buildFrom(ruleContext))
        .addProvider(
            J2ObjcMappingFileProvider.class, ObjcRuleClasses.j2ObjcMappingFileProvider(ruleContext))
        .addProvider(
            InstrumentedFilesProvider.class,
            compilationSupport.getInstrumentedFilesProvider(common))
        .build();
  }
}
