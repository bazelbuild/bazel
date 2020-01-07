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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.vfs.PathFragment;
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
      ImmutableList.of("java_import", "java_library", "proto_library");

  private ObjcCommon common(RuleContext ruleContext) throws InterruptedException {
    return new ObjcCommon.Builder(ruleContext)
        .setCompilationAttributes(
            CompilationAttributes.Builder.fromRuleContext(ruleContext).build())
        .addDeps(ruleContext.getPrerequisiteConfiguredTargetAndTargets("deps", Mode.TARGET))
        .addDeps(ruleContext.getPrerequisiteConfiguredTargetAndTargets("jre_deps", Mode.TARGET))
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

    J2ObjcMappingFileProvider j2ObjcMappingFileProvider = J2ObjcMappingFileProvider.union(
        ruleContext.getPrerequisites("deps", Mode.TARGET, J2ObjcMappingFileProvider.class));

    CompilationArtifacts moduleMapCompilationArtifacts =
        new CompilationArtifacts.Builder()
            .setIntermediateArtifacts(ObjcRuleClasses.intermediateArtifacts(ruleContext))
            .build();

    new CompilationSupport.Builder()
        .setRuleContext(ruleContext)
        .setIntermediateArtifacts(ObjcRuleClasses.intermediateArtifacts(ruleContext))
        .doNotUsePch()
        .build()
        .registerFullyLinkAction(
            objcProvider,
            ruleContext.getImplicitOutputArtifact(CompilationSupport.FULLY_LINKED_LIB))
        .registerGenerateModuleMapAction(moduleMapCompilationArtifacts)
        .validateAttributes();

    return new RuleConfiguredTargetBuilder(ruleContext)
        .setFilesToBuild(NestedSetBuilder.<Artifact>emptySet(STABLE_ORDER))
        .add(RunfilesProvider.class, RunfilesProvider.EMPTY)
        .addProvider(J2ObjcEntryClassProvider.class, j2ObjcEntryClassProvider)
        .addProvider(J2ObjcMappingFileProvider.class, j2ObjcMappingFileProvider)
        .addNativeDeclaredProvider(objcProvider)
        .addSkylarkTransitiveInfo(ObjcProvider.SKYLARK_NAME, objcProvider)
        .build();
  }

  /**
   * Returns header search paths necessary to compile the J2ObjC-generated code from a single
   * target.
   *
   * @param ruleContext the rule context
   * @param objcFileRootExecPath the exec path under which all J2ObjC-generated file resides
   * @param sourcesToTranslate the source files to be translated by J2ObjC in a single target
   */
  public static Iterable<PathFragment> j2objcSourceHeaderSearchPaths(RuleContext ruleContext,
      PathFragment objcFileRootExecPath, Iterable<Artifact> sourcesToTranslate) {
    PathFragment genRoot = ruleContext.getConfiguration().getGenfilesFragment();
    ImmutableList.Builder<PathFragment> headerSearchPaths = ImmutableList.builder();
    headerSearchPaths.add(objcFileRootExecPath);
    // We add another header search path with gen root if we have generated sources to translate.
    for (Artifact sourceToTranslate : sourcesToTranslate) {
      if (!sourceToTranslate.isSourceArtifact()) {
        headerSearchPaths.add(objcFileRootExecPath.getRelative(genRoot));
        return headerSearchPaths.build();
      }
    }

    return headerSearchPaths.build();
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
