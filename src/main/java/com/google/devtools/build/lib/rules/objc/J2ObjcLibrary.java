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
import static com.google.devtools.build.lib.rules.objc.XcodeProductType.LIBRARY_STATIC;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.rules.cpp.CppModuleMap;
import com.google.devtools.build.lib.rules.java.J2ObjcConfiguration;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.List;

/**
 * Implementation for the "j2objc_library" rule, which exports ObjC source files translated from
 * Java source files in java_library rules to dependent objc_binary rules for compilation and
 * linking into the final application bundle. See {@link J2ObjcLibraryBaseRule} for details.
 */
public class J2ObjcLibrary implements RuleConfiguredTargetFactory {

  public static final String NO_ENTRY_CLASS_ERROR_MSG =
      "Entry classes must be specified when flag --compilationMode=opt is on in order to"
          + " perform J2ObjC dead code stripping.";

  @Override
  public ConfiguredTarget create(RuleContext ruleContext) throws InterruptedException {
    checkAttributes(ruleContext);

    if (ruleContext.hasErrors()) {
      return null;
    }

    J2ObjcSrcsProvider j2ObjcSrcsProvider = new J2ObjcSrcsProvider.Builder()
        .addTransitiveJ2ObjcSrcs(ruleContext)
        .addEntryClasses(ruleContext.attributes().get("entry_classes", Type.STRING_LIST))
        .build();

    ObjcProvider.Builder objcProviderBuilder =
        new ObjcProvider.Builder()
            .addJ2ObjcTransitiveAndPropagate(
                ruleContext.getPrerequisite("$jre_emul_lib", Mode.TARGET, ObjcProvider.class))
            .addJ2ObjcTransitiveAndPropagate(
                ruleContext.getPrerequisites("deps", Mode.TARGET, ObjcProvider.class));

    XcodeProvider.Builder xcodeProviderBuilder = new XcodeProvider.Builder();
    XcodeSupport xcodeSupport =
        new XcodeSupport(ruleContext)
            .addDependencies(xcodeProviderBuilder, new Attribute("$jre_emul_lib", Mode.TARGET))
            .addDependencies(xcodeProviderBuilder, new Attribute("deps", Mode.TARGET));

    if (j2ObjcSrcsProvider.hasProtos()) {
      // Public J2 in Bazel provides no protobuf_lib, and if OSS users try to sneakily use
      // undocumented functionality to reach here, the below code will error.
      objcProviderBuilder.addJ2ObjcTransitiveAndPropagate(
          ruleContext.getPrerequisite("$protobuf_lib", Mode.TARGET, ObjcProvider.class));
      xcodeSupport.addDependencies(
          xcodeProviderBuilder, new Attribute("$protobuf_lib", Mode.TARGET));
    }

    for (J2ObjcSource j2objcSource : j2ObjcSrcsProvider.getSrcs()) {
      objcProviderBuilder.addJ2ObjcAll(ObjcProvider.HEADER, j2objcSource.getObjcHdrs());
      objcProviderBuilder.addJ2ObjcAll(ObjcProvider.INCLUDE, j2objcSource.getHeaderSearchPaths());
      xcodeProviderBuilder.addHeaders(j2objcSource.getObjcHdrs());
      xcodeProviderBuilder.addUserHeaderSearchPaths(j2objcSource.getHeaderSearchPaths());
    }

    if (ObjcRuleClasses.objcConfiguration(ruleContext).moduleMapsEnabled()) {
      configureModuleMap(ruleContext, objcProviderBuilder, j2ObjcSrcsProvider);
    }

    ObjcProvider objcProvider = objcProviderBuilder.build();
    xcodeSupport.addXcodeSettings(xcodeProviderBuilder, objcProvider, LIBRARY_STATIC);

    return new RuleConfiguredTargetBuilder(ruleContext)
        .setFilesToBuild(NestedSetBuilder.<Artifact>emptySet(STABLE_ORDER))
        .add(RunfilesProvider.class, RunfilesProvider.EMPTY)
        .addProvider(J2ObjcSrcsProvider.class, j2ObjcSrcsProvider)
        .addProvider(
            J2ObjcMappingFileProvider.class, ObjcRuleClasses.j2ObjcMappingFileProvider(ruleContext))
        .addProvider(ObjcProvider.class, objcProvider)
        .addProvider(XcodeProvider.class, xcodeProviderBuilder.build())
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
        headerSearchPaths.add(new PathFragment(objcFileRootExecPath, genRoot));
        return headerSearchPaths.build();
      }
    }

    return headerSearchPaths.build();
  }

  /**
   * Configures a module map for all the sources in {@code j2ObjcSrcsProvider}, registering
   * an action to generate the module map and exposing that module map through {@code objcProvider}.
   */
  private void configureModuleMap(
      RuleContext ruleContext,
      ObjcProvider.Builder objcProvider,
      J2ObjcSrcsProvider j2ObjcSrcsProvider) {
    new CompilationSupport(ruleContext).registerJ2ObjcGenerateModuleMapAction(j2ObjcSrcsProvider);

    CppModuleMap moduleMap = ObjcRuleClasses.intermediateArtifacts(ruleContext).moduleMap();
    objcProvider.add(ObjcProvider.MODULE_MAP, moduleMap.getArtifact());
    objcProvider.add(ObjcProvider.TOP_LEVEL_MODULE_MAP, moduleMap);
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
