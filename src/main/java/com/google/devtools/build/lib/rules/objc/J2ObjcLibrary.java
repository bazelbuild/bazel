// Copyright 2015 Google Inc. All rights reserved.
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
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.rules.cpp.CppModuleMap;
import com.google.devtools.build.lib.rules.java.J2ObjcConfiguration;
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
        .addTransitiveFromDeps(ruleContext)
        .addEntryClasses(ruleContext.attributes().get("entry_classes", Type.STRING_LIST))
        .build();

    ObjcProvider.Builder objcProviderBuilder =
        new ObjcProvider.Builder()
            .addTransitiveAndPropagate(
                ruleContext.getPrerequisite("$jre_emul_lib", Mode.TARGET, ObjcProvider.class))
            .addTransitiveAndPropagate(
                ruleContext.getPrerequisites("deps", Mode.TARGET, ObjcProvider.class));

    XcodeProvider.Builder xcodeProviderBuilder = new XcodeProvider.Builder();
    XcodeSupport xcodeSupport =
        new XcodeSupport(ruleContext)
            .addDependencies(xcodeProviderBuilder, new Attribute("$jre_emul_lib", Mode.TARGET))
            .addDependencies(xcodeProviderBuilder, new Attribute("deps", Mode.TARGET));

    if (j2ObjcSrcsProvider.hasProtos()) {
      if (ruleContext.attributes().has("$protobuf_lib", Type.LABEL)) {
        objcProviderBuilder.addTransitiveAndPropagate(
            ruleContext.getPrerequisite("$protobuf_lib", Mode.TARGET, ObjcProvider.class));
        xcodeSupport.addDependencies(
            xcodeProviderBuilder, new Attribute("$protobuf_lib", Mode.TARGET));
      } else {
        // In theory no Bazel rule should ever provide protos, because they're not supported yet.
        // If we reach here, it's a programming error, not a rule error.
        throw new IllegalStateException(
            "Found protos in the dependencies of rule " + ruleContext.getLabel() + ", "
                + "but protos are not supported in Bazel.");
      }
    }

    for (J2ObjcSource j2objcSource : j2ObjcSrcsProvider.getSrcs()) {
      PathFragment genDirHeaderSearchPath =
          new PathFragment(
              j2objcSource.getObjcFilePath(), ruleContext.getConfiguration().getGenfilesFragment());

      objcProviderBuilder.addAll(ObjcProvider.HEADER, j2objcSource.getObjcHdrs());
      objcProviderBuilder.add(ObjcProvider.INCLUDE, j2objcSource.getObjcFilePath());
      objcProviderBuilder.add(ObjcProvider.INCLUDE, genDirHeaderSearchPath);
      xcodeProviderBuilder.addHeaders(j2objcSource.getObjcHdrs());
      xcodeProviderBuilder.addUserHeaderSearchPaths(
          ImmutableList.of(j2objcSource.getObjcFilePath(), genDirHeaderSearchPath));
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
    if (!ruleContext.attributes().has(attributeName, Type.LABEL_LIST)) {
      return;
    }

    List<String> entryClasses = ruleContext.attributes().get("entry_classes", Type.STRING_LIST);
    J2ObjcConfiguration j2objcConfiguration = ruleContext.getFragment(J2ObjcConfiguration.class);
    if (j2objcConfiguration.removeDeadCode() && (entryClasses == null || entryClasses.isEmpty())) {
      ruleContext.attributeError("entry_classes", NO_ENTRY_CLASS_ERROR_MSG);
    }
  }
}
