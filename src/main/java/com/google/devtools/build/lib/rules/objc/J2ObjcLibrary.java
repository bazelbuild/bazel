// Copyright 2015 The Bazel Authors. All rights reserved.
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

import static com.google.devtools.build.lib.collect.nestedset.Order.STABLE_ORDER;
import static java.util.stream.Collectors.toList;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.cpp.CcInfo;
import com.google.devtools.build.lib.rules.objc.J2ObjcAspect.J2ObjcCcInfo;
import java.util.List;

/**
 * Implementation for the "j2objc_library" rule, which exports ObjC source files translated from
 * Java source files in java_library rules to dependent objc_binary rules for compilation and
 * linking into the final application bundle. See {@link J2ObjcLibraryBaseRule} for details.
 */
public class J2ObjcLibrary implements RuleConfiguredTargetFactory {

  public static final String NO_ENTRY_CLASS_ERROR_MSG =
      "Entry classes must be specified when flag --compilation_mode=opt is on in order to"
          + " perform J2ObjC dead code stripping.";

  public static final ImmutableList<String> J2OBJC_SUPPORTED_RULES =
      ImmutableList.of("java_import", "java_library", "java_proto_library", "proto_library");

  private ObjcCommon common(RuleContext ruleContext) throws InterruptedException {
    List<J2ObjcCcInfo> j2objcCcInfos = ruleContext.getPrerequisites("deps", J2ObjcCcInfo.class);
    return new ObjcCommon.Builder(ObjcCommon.Purpose.LINK_ONLY, ruleContext)
        .setCompilationAttributes(
            CompilationAttributes.Builder.fromRuleContext(ruleContext).build())
        .addDeps(ruleContext.getPrerequisiteConfiguredTargets("deps"))
        .addDeps(ruleContext.getPrerequisiteConfiguredTargets("jre_deps"))
        .addDirectCcCompilationContexts(
            j2objcCcInfos.stream().map(J2ObjcCcInfo::getCcInfo).collect(toList()))
        .setIntermediateArtifacts(ObjcRuleClasses.intermediateArtifacts(ruleContext))
        .setHasModuleMap()
        .build();
  }

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    checkAttributes(ruleContext);

    if (ruleContext.hasErrors()) {
      return null;
    }

    J2ObjcEntryClassProvider j2ObjcEntryClassProvider = new J2ObjcEntryClassProvider.Builder()
        .addTransitive(ruleContext)
        .addEntryClasses(ruleContext.attributes().get("entry_classes", Type.STRING_LIST))
        .build();

    ObjcCommon common = common(ruleContext);
    ObjcProvider objcProvider = common.getObjcProvider();

    J2ObjcMappingFileProvider j2ObjcMappingFileProvider =
        J2ObjcMappingFileProvider.union(
            ruleContext.getPrerequisites("deps", J2ObjcMappingFileProvider.class));

    CompilationArtifacts moduleMapCompilationArtifacts =
        new CompilationArtifacts.Builder()
            .setIntermediateArtifacts(ObjcRuleClasses.intermediateArtifacts(ruleContext))
            .build();

    new CompilationSupport.Builder()
        .setRuleContext(ruleContext)
        .setIntermediateArtifacts(ObjcRuleClasses.intermediateArtifacts(ruleContext))
        .doNotUsePch()
        .build()
        .registerGenerateModuleMapAction(moduleMapCompilationArtifacts)
        .validateAttributes();

    return new RuleConfiguredTargetBuilder(ruleContext)
        .setFilesToBuild(NestedSetBuilder.<Artifact>emptySet(STABLE_ORDER))
        .add(RunfilesProvider.class, RunfilesProvider.EMPTY)
        .addProvider(J2ObjcEntryClassProvider.class, j2ObjcEntryClassProvider)
        .addProvider(J2ObjcMappingFileProvider.class, j2ObjcMappingFileProvider)
        .addNativeDeclaredProvider(objcProvider)
        .addNativeDeclaredProvider(
            CcInfo.builder().setCcCompilationContext(common.getCcCompilationContext()).build())
        .addStarlarkTransitiveInfo(ObjcProvider.STARLARK_NAME, objcProvider)
        .build();
  }

  private static void checkAttributes(RuleContext ruleContext) {
    checkAttributes(ruleContext, "deps");
    checkAttributes(ruleContext, "exports");
  }

  private static void checkAttributes(RuleContext ruleContext, String attributeName) {
    if (!ruleContext.attributes().has(attributeName, BuildType.LABEL_LIST)) {
      return;
    }

    List<String> entryClasses = ruleContext.attributes().get("entry_classes", Type.STRING_LIST);
    J2ObjcConfiguration j2objcConfiguration = ruleContext.getFragment(J2ObjcConfiguration.class);
    if (j2objcConfiguration.removeDeadCode() && (entryClasses == null || entryClasses.isEmpty())) {
      ruleContext.attributeError("entry_classes", NO_ENTRY_CLASS_ERROR_MSG);
    }
  }
}
