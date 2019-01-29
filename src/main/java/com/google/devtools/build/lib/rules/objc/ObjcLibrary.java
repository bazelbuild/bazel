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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.cpp.CcCompilationContext;
import com.google.devtools.build.lib.rules.cpp.CcInfo;
import com.google.devtools.build.lib.rules.cpp.LibraryToLinkWrapper;
import com.google.devtools.build.lib.rules.cpp.LibraryToLinkWrapper.CcLinkingContext;
import com.google.devtools.build.lib.rules.cpp.LibraryToLinkWrapper.CcLinkingContext.LinkOptions;
import com.google.devtools.build.lib.rules.objc.ObjcCommon.ResourceAttributes;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import java.util.Map;
import java.util.TreeMap;

/**
 * Implementation for {@code objc_library}.
 */
public class ObjcLibrary implements RuleConfiguredTargetFactory {

  /**
   * Constructs an {@link ObjcCommon} instance based on the attributes of the given rule context.
   */
  private ObjcCommon common(RuleContext ruleContext) throws InterruptedException {
    return new ObjcCommon.Builder(ruleContext)
        .setCompilationAttributes(
            CompilationAttributes.Builder.fromRuleContext(ruleContext).build())
        .setResourceAttributes(new ResourceAttributes(ruleContext))
        .addDefines(ruleContext.getExpander().withDataLocations().tokenized("defines"))
        .setCompilationArtifacts(CompilationSupport.compilationArtifacts(ruleContext))
        .addDeps(ruleContext.getPrerequisiteConfiguredTargetAndTargets("deps", Mode.TARGET))
        .addRuntimeDeps(ruleContext.getPrerequisites("runtime_deps", Mode.TARGET))
        .addDepObjcProviders(
            ruleContext.getPrerequisites("bundles", Mode.TARGET, ObjcProvider.SKYLARK_CONSTRUCTOR))
        .setIntermediateArtifacts(ObjcRuleClasses.intermediateArtifacts(ruleContext))
        .setAlwayslink(ruleContext.attributes().get("alwayslink", Type.BOOLEAN))
        .setHasModuleMap()
        .build();
  }

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    validateAttributes(ruleContext);

    ObjcCommon common = common(ruleContext);

    NestedSetBuilder<Artifact> filesToBuild = NestedSetBuilder.<Artifact>stableOrder()
        .addAll(common.getCompiledArchive().asSet());

    Map<String, NestedSet<Artifact>> outputGroupCollector = new TreeMap<>();
    ImmutableList.Builder<Artifact> objectFilesCollector = ImmutableList.builder();
    CompilationSupport compilationSupport =
        new CompilationSupport.Builder()
            .setRuleContext(ruleContext)
            .setOutputGroupCollector(outputGroupCollector)
            .setObjectFilesCollector(objectFilesCollector)
            .build();

    compilationSupport
        .registerCompileAndArchiveActions(common)
        .registerFullyLinkAction(
            common.getObjcProvider(),
            ruleContext.getImplicitOutputArtifact(CompilationSupport.FULLY_LINKED_LIB))
        .validateAttributes();

    new ResourceSupport(ruleContext).validateAttributes();

    J2ObjcMappingFileProvider j2ObjcMappingFileProvider = J2ObjcMappingFileProvider.union(
            ruleContext.getPrerequisites("deps", Mode.TARGET, J2ObjcMappingFileProvider.class));
    J2ObjcEntryClassProvider j2ObjcEntryClassProvider = new J2ObjcEntryClassProvider.Builder()
      .addTransitive(ruleContext.getPrerequisites("deps", Mode.TARGET,
          J2ObjcEntryClassProvider.class)).build();
    CcCompilationContext ccCompilationContext =
        new CcCompilationContext.Builder(ruleContext)
            .addDeclaredIncludeSrcs(
                CompilationAttributes.Builder.fromRuleContext(ruleContext)
                    .build()
                    .hdrs()
                    .toCollection())
            .addTextualHdrs(common.getTextualHdrs())
            .addDeclaredIncludeSrcs(common.getTextualHdrs())
            .build();

    CcLinkingContext ccLinkingContext = buildCcLinkingContext(common);

    return ObjcRuleClasses.ruleConfiguredTarget(ruleContext, filesToBuild.build())
        .addNativeDeclaredProvider(common.getObjcProvider())
        .addProvider(J2ObjcEntryClassProvider.class, j2ObjcEntryClassProvider)
        .addProvider(J2ObjcMappingFileProvider.class, j2ObjcMappingFileProvider)
        .addNativeDeclaredProvider(
            compilationSupport.getInstrumentedFilesProvider(objectFilesCollector.build()))
        .addNativeDeclaredProvider(
            CcInfo.builder()
                .setCcCompilationContext(ccCompilationContext)
                .setCcLinkingContext(ccLinkingContext)
                .build())
        .addOutputGroups(outputGroupCollector)
        .build();
  }

  public CcLinkingContext buildCcLinkingContext(ObjcCommon common) {
    ImmutableSet.Builder<LibraryToLinkWrapper> libraries = new ImmutableSet.Builder<>();
    ObjcProvider objcProvider = common.getObjcProvider();
    for (Artifact library : objcProvider.get(ObjcProvider.LIBRARY)) {
      libraries.add(
          LibraryToLinkWrapper.builder()
              .setStaticLibrary(library)
              .setLibraryIdentifier(
                  FileSystemUtils.removeExtension(library.getRootRelativePathString()))
              .build());
    }
    libraries.addAll(objcProvider.get(ObjcProvider.CC_LIBRARY));

    CcLinkingContext.Builder ccLinkingContext =
        CcLinkingContext.builder()
            .addLibraries(
                NestedSetBuilder.<LibraryToLinkWrapper>linkOrder()
                    .addAll(libraries.build())
                    .build());

    NestedSetBuilder<LinkOptions> userLinkFlags = NestedSetBuilder.linkOrder();
    for (SdkFramework sdkFramework : objcProvider.get(ObjcProvider.SDK_FRAMEWORK)) {
      userLinkFlags.add(
          CcLinkingContext.LinkOptions.of(ImmutableList.of("-framework", sdkFramework.getName())));
    }
    ccLinkingContext.addUserLinkFlags(userLinkFlags.build());

    return ccLinkingContext.build();
  }

  /** Throws errors or warnings for bad attribute state. */
  private static void validateAttributes(RuleContext ruleContext) {
    for (String copt : ObjcCommon.getNonCrosstoolCopts(ruleContext)) {
      if (copt.contains("-fmodules-cache-path")) {
        ruleContext.ruleWarning(CompilationSupport.MODULES_CACHE_PATH_WARNING);
      }
    }
  }
}
