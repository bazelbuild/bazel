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
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.compacthashset.CompactHashSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.SymbolGenerator;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.cpp.CcCompilationContext;
import com.google.devtools.build.lib.rules.cpp.CcInfo;
import com.google.devtools.build.lib.rules.cpp.CcLinkingContext;
import com.google.devtools.build.lib.rules.cpp.CcLinkingContext.LinkOptions;
import com.google.devtools.build.lib.rules.cpp.CcLinkingContext.LinkerInput;
import com.google.devtools.build.lib.rules.cpp.CppSemantics;
import com.google.devtools.build.lib.rules.cpp.LibraryToLink;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

/**
 * Implementation for {@code objc_library}.
 */
public class ObjcLibrary implements RuleConfiguredTargetFactory {
  private final CppSemantics cppSemantics;

  protected ObjcLibrary(CppSemantics cppSemantics) {
    this.cppSemantics = cppSemantics;
  }

  /**
   * Constructs an {@link ObjcCommon} instance based on the attributes of the given rule context.
   */
  private static ObjcCommon common(RuleContext ruleContext) throws InterruptedException {
    return new ObjcCommon.Builder(ObjcCommon.Purpose.COMPILE_AND_LINK, ruleContext)
        .setCompilationAttributes(
            CompilationAttributes.Builder.fromRuleContext(ruleContext).build())
        .setCompilationArtifacts(CompilationSupport.compilationArtifacts(ruleContext))
        .addDeps(ruleContext.getPrerequisiteConfiguredTargets("deps"))
        .addRuntimeDeps(ruleContext.getPrerequisites("runtime_deps"))
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
        new CompilationSupport.Builder(ruleContext, cppSemantics)
            .setOutputGroupCollector(outputGroupCollector)
            .setObjectFilesCollector(objectFilesCollector)
            .build();

    compilationSupport
        .registerCompileAndArchiveActions(common)
        .validateAttributes();

    J2ObjcMappingFileProvider j2ObjcMappingFileProvider =
        J2ObjcMappingFileProvider.union(
            ruleContext.getPrerequisites("deps", J2ObjcMappingFileProvider.PROVIDER));
    J2ObjcEntryClassProvider j2ObjcEntryClassProvider =
        new J2ObjcEntryClassProvider.Builder()
            .addTransitive(ruleContext.getPrerequisites("deps", J2ObjcEntryClassProvider.PROVIDER))
            .build();
    ObjcProvider objcProvider = common.getObjcProvider();
    CcCompilationContext ccCompilationContext = compilationSupport.getCcCompilationContext();
    CcLinkingContext ccLinkingContext =
        buildCcLinkingContext(
            ruleContext.getLabel(), objcProvider, ruleContext.getSymbolGenerator());

    return ObjcRuleClasses.ruleConfiguredTarget(ruleContext, filesToBuild.build())
        .addNativeDeclaredProvider(objcProvider)
        .addStarlarkTransitiveInfo(ObjcProvider.STARLARK_NAME, objcProvider)
        .addNativeDeclaredProvider(j2ObjcEntryClassProvider)
        .addNativeDeclaredProvider(j2ObjcMappingFileProvider)
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

  private static CcLinkingContext buildCcLinkingContext(
      Label label, ObjcProvider objcProvider, SymbolGenerator<?> symbolGenerator) {
    List<Artifact> libraries = objcProvider.get(ObjcProvider.LIBRARY).toList();
    List<LibraryToLink> ccLibraries = objcProvider.get(ObjcProvider.CC_LIBRARY).toList();

    Set<LibraryToLink> librariesToLink =
        CompactHashSet.createWithExpectedSize(libraries.size() + ccLibraries.size());
    for (Artifact library : libraries) {
      librariesToLink.add(LibraryToLink.staticOnly(library));
    }

    for (LibraryToLink library : ccLibraries) {
      librariesToLink.add(convertToStaticLibrary(library));
    }

    List<SdkFramework> sdkFrameworks = objcProvider.get(ObjcProvider.SDK_FRAMEWORK).toList();
    ImmutableList.Builder<LinkOptions> userLinkFlags =
        ImmutableList.builderWithExpectedSize(sdkFrameworks.size());
    for (SdkFramework sdkFramework : sdkFrameworks) {
      userLinkFlags.add(
          LinkOptions.of(ImmutableList.of("-framework", sdkFramework.getName()), symbolGenerator));
    }

    LinkerInput linkerInput =
        new LinkerInput(
            label,
            ImmutableList.copyOf(librariesToLink),
            userLinkFlags.build(),
            /*nonCodeInputs=*/ ImmutableList.of(),
            objcProvider.get(ObjcProvider.LINKSTAMP).toList());

    return new CcLinkingContext(
        NestedSetBuilder.create(Order.LINK_ORDER, linkerInput), /*extraLinkTimeLibraries=*/ null);
  }

  /**
   * Removes dynamic libraries from {@link LibraryToLink} objects coming from C++ dependencies. The
   * reason for this is that objective-C rules do not support linking the dynamic version of the
   * libraries.
   *
   * <p>Returns the same object if nothing would be changed.
   */
  private static LibraryToLink convertToStaticLibrary(LibraryToLink library) {
    if ((library.getPicStaticLibrary() == null && library.getStaticLibrary() == null)
        || (library.getDynamicLibrary() == null && library.getInterfaceLibrary() == null)) {
      return library;
    }

    return library.toBuilder()
        .setDynamicLibrary(null)
        .setResolvedSymlinkDynamicLibrary(null)
        .setInterfaceLibrary(null)
        .setResolvedSymlinkInterfaceLibrary(null)
        .build();
  }

  /** Throws errors or warnings for bad attribute state. */
  private static void validateAttributes(RuleContext ruleContext) {
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
