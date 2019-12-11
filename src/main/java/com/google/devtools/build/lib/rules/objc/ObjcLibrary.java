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
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.SymbolGenerator;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.cpp.CcCommon;
import com.google.devtools.build.lib.rules.cpp.CcCompilationContext;
import com.google.devtools.build.lib.rules.cpp.CcInfo;
import com.google.devtools.build.lib.rules.cpp.CcLinkingContext;
import com.google.devtools.build.lib.rules.cpp.CcLinkingContext.LinkOptions;
import com.google.devtools.build.lib.rules.cpp.LibraryToLink;
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
        .setCompilationArtifacts(CompilationSupport.compilationArtifacts(ruleContext))
        .addDeps(ruleContext.getPrerequisiteConfiguredTargetAndTargets("deps", Mode.TARGET))
        .addRuntimeDeps(ruleContext.getPrerequisites("runtime_deps", Mode.TARGET))
        .setIntermediateArtifacts(ObjcRuleClasses.intermediateArtifacts(ruleContext))
        .setAlwayslink(ruleContext.attributes().get("alwayslink", Type.BOOLEAN))
        .setHasModuleMap()
        .build();
  }

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    CcCommon.checkRuleLoadedThroughMacro(ruleContext);
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

    J2ObjcMappingFileProvider j2ObjcMappingFileProvider = J2ObjcMappingFileProvider.union(
            ruleContext.getPrerequisites("deps", Mode.TARGET, J2ObjcMappingFileProvider.class));
    J2ObjcEntryClassProvider j2ObjcEntryClassProvider = new J2ObjcEntryClassProvider.Builder()
      .addTransitive(ruleContext.getPrerequisites("deps", Mode.TARGET,
          J2ObjcEntryClassProvider.class)).build();
    ObjcProvider objcProvider = common.getObjcProvider();
    CcCompilationContext ccCompilationContext =
        CcCompilationContext.builder(
                ruleContext, ruleContext.getConfiguration(), ruleContext.getLabel())
            .addDeclaredIncludeSrcs(
                CompilationAttributes.Builder.fromRuleContext(ruleContext).build().hdrs().toList())
            .addTextualHdrs(common.getTextualHdrs())
            .addDeclaredIncludeSrcs(common.getTextualHdrs())
            .setPurpose(
                compilationSupport
                    .createObjcCppSemantics(
                        objcProvider, /* privateHdrs= */ ImmutableList.of(), /* pchHdr= */ null)
                    .getPurpose())
            .build();

    CcLinkingContext ccLinkingContext =
        buildCcLinkingContext(ruleContext.getLabel(), common, ruleContext.getSymbolGenerator());

    return ObjcRuleClasses.ruleConfiguredTarget(ruleContext, filesToBuild.build())
        .addNativeDeclaredProvider(objcProvider)
        .addSkylarkTransitiveInfo(ObjcProvider.SKYLARK_NAME, objcProvider)
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

  private CcLinkingContext buildCcLinkingContext(
      Label label, ObjcCommon common, SymbolGenerator<?> symbolGenerator) {
    ImmutableSet.Builder<LibraryToLink> libraries = new ImmutableSet.Builder<>();
    ObjcProvider objcProvider = common.getObjcProvider();
    for (Artifact library : objcProvider.get(ObjcProvider.LIBRARY)) {
      libraries.add(
          LibraryToLink.builder()
              .setStaticLibrary(library)
              .setLibraryIdentifier(
                  FileSystemUtils.removeExtension(library.getRootRelativePathString()))
              .build());
    }

    libraries.addAll(convertLibrariesToStaticLibraries(objcProvider.get(ObjcProvider.CC_LIBRARY)));

    CcLinkingContext.Builder ccLinkingContext =
        CcLinkingContext.builder()
            .setOwner(label)
            .addLibraries(ImmutableList.copyOf(libraries.build()));

    ImmutableList.Builder<LinkOptions> userLinkFlags = ImmutableList.builder();
    for (SdkFramework sdkFramework : objcProvider.get(ObjcProvider.SDK_FRAMEWORK)) {
      userLinkFlags.add(
          LinkOptions.of(ImmutableList.of("-framework", sdkFramework.getName()), symbolGenerator));
    }
    ccLinkingContext.addUserLinkFlags(userLinkFlags.build());

    return ccLinkingContext.build();
  }

  /**
   * This method removes dynamic libraries from LibraryToLink objects coming from C++ dependencies.
   * The reason for this is that objective-C rules do not support linking the dynamic version of the
   * libraries.
   */
  private ImmutableList<LibraryToLink> convertLibrariesToStaticLibraries(
      Iterable<LibraryToLink> librariesToLink) {
    ImmutableList.Builder<LibraryToLink> libraries = ImmutableList.builder();
    for (LibraryToLink libraryToLink : librariesToLink) {
      LibraryToLink.Builder staticLibraryToLink = libraryToLink.toBuilder();
      if (libraryToLink.getPicStaticLibrary() != null || libraryToLink.getStaticLibrary() != null) {
        staticLibraryToLink.setDynamicLibrary(null);
        staticLibraryToLink.setResolvedSymlinkDynamicLibrary(null);
        staticLibraryToLink.setInterfaceLibrary(null);
        staticLibraryToLink.setResolvedSymlinkInterfaceLibrary(null);
      }
      libraries.add(staticLibraryToLink.build());
    }
    return libraries.build();
  }

  /** Throws errors or warnings for bad attribute state. */
  private static void validateAttributes(RuleContext ruleContext) throws RuleErrorException {
    // TODO(b/129469095): objc_library cannot handle target names with slashes.  Rather than
    // crashing bazel, we emit a useful error message.
    if (ruleContext.getTarget().getName().indexOf('/') != -1) {
      ruleContext.attributeError("name", "this attribute has unsupported character '/'");
    }
    for (String copt : ObjcCommon.getNonCrosstoolCopts(ruleContext)) {
      if (copt.contains("-fmodules-cache-path")) {
        ruleContext.ruleWarning(CompilationSupport.MODULES_CACHE_PATH_WARNING);
      }
    }
  }
}
