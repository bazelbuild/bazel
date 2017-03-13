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

import static com.google.devtools.build.lib.rules.objc.XcodeProductType.LIBRARY_STATIC;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.rules.cpp.CcLinkParamsProvider;
import com.google.devtools.build.lib.rules.objc.ObjcCommon.ResourceAttributes;
import com.google.devtools.build.lib.rules.test.InstrumentedFilesProvider;
import com.google.devtools.build.lib.syntax.Type;

/** Implementation for experimental_objc_library. */
public class ExperimentalObjcLibrary implements RuleConfiguredTargetFactory {

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

    ObjcCommon common = common(ruleContext);

    CompilationSupport compilationSupport =
        new CrosstoolCompilationSupport(ruleContext)
            .registerCompileAndArchiveActions(common)
            .registerFullyLinkAction(
                common.getObjcProvider(),
                ruleContext.getImplicitOutputArtifact(CompilationSupport.FULLY_LINKED_LIB));

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

    J2ObjcMappingFileProvider j2ObjcMappingFileProvider =
        J2ObjcMappingFileProvider.union(
            ruleContext.getPrerequisites("deps", Mode.TARGET, J2ObjcMappingFileProvider.class));
    J2ObjcEntryClassProvider j2ObjcEntryClassProvider =
        new J2ObjcEntryClassProvider.Builder()
            .addTransitive(
                ruleContext.getPrerequisites("deps", Mode.TARGET, J2ObjcEntryClassProvider.class))
            .build();

    return ObjcRuleClasses.ruleConfiguredTarget(ruleContext, filesToBuild.build())
        .addProvider(ObjcProvider.class, common.getObjcProvider())
        .addProvider(J2ObjcEntryClassProvider.class, j2ObjcEntryClassProvider)
        .addProvider(J2ObjcMappingFileProvider.class, j2ObjcMappingFileProvider)
        .addProvider(InstrumentedFilesProvider.class,
            compilationSupport.getInstrumentedFilesProvider(common))
        .addProvider(
            CcLinkParamsProvider.class,
            new CcLinkParamsProvider(new ObjcLibraryCcLinkParamsStore(common)))
        .addProvider(ObjcProvider.class, common.getObjcProvider())
        .addProvider(XcodeProvider.class, xcodeProviderBuilder.build())
        .build();
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
  private static ObjcCommon common(RuleContext ruleContext) {
    return new ObjcCommon.Builder(ruleContext)
        .setCompilationAttributes(
            CompilationAttributes.Builder.fromRuleContext(ruleContext).build())
        .setResourceAttributes(new ResourceAttributes(ruleContext))
        .addDefines(ruleContext.getTokenizedStringListAttr("defines"))
        .setCompilationArtifacts(CompilationSupport.compilationArtifacts(ruleContext))
        .addDeps(ruleContext.getPrerequisites("deps", Mode.TARGET))
        .addRuntimeDeps(ruleContext.getPrerequisites("runtime_deps", Mode.TARGET))
        .addDepObjcProviders(
            ruleContext.getPrerequisites("bundles", Mode.TARGET, ObjcProvider.class))
        .addNonPropagatedDepObjcProviders(
            ruleContext.getPrerequisites("non_propagated_deps", Mode.TARGET, ObjcProvider.class))
        .setIntermediateArtifacts(ObjcRuleClasses.intermediateArtifacts(ruleContext))
        .setAlwayslink(ruleContext.attributes().get("alwayslink", Type.BOOLEAN))
        .setHasModuleMap()
        .build();
  }
}
