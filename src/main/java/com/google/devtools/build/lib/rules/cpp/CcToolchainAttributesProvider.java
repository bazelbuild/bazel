// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.cpp;

import static com.google.devtools.build.lib.packages.BuildType.NODEP_LABEL;
import static com.google.devtools.build.lib.syntax.Type.BOOLEAN;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Actions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.LicensesProvider;
import com.google.devtools.build.lib.analysis.LicensesProvider.TargetLicense;
import com.google.devtools.build.lib.analysis.LicensesProviderImpl;
import com.google.devtools.build.lib.analysis.MiddlemanProvider;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.License;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.packages.NativeProvider;
import com.google.devtools.build.lib.syntax.Type;

/**
 * Provider encapsulating all the information from the cc_toolchain rule that affects creation of
 * {@link CcToolchainProvider}
 */
public class CcToolchainAttributesProvider extends NativeInfo {

  public static final NativeProvider<CcToolchainAttributesProvider> PROVIDER =
      new NativeProvider<CcToolchainAttributesProvider>(
          CcToolchainAttributesProvider.class, "CcToolchainAttributesInfo") {};

  private final boolean supportsParamFiles;
  private final boolean supportsHeaderParsing;
  private final NestedSet<Artifact> crosstool;
  private final NestedSet<Artifact> crosstoolMiddleman;
  private final NestedSet<Artifact> compile;
  private final NestedSet<Artifact> compileWithoutIncludes;
  private final NestedSet<Artifact> strip;
  private final NestedSet<Artifact> objcopy;
  private final NestedSet<Artifact> as;
  private final NestedSet<Artifact> ar;
  private final NestedSet<Artifact> link;
  private final NestedSet<Artifact> dwp;
  private final NestedSet<Artifact> libc;
  private final NestedSet<Artifact> libcMiddleman;
  private final NestedSet<Artifact> fullInputsForCrosstool;
  private final NestedSet<Artifact> fullInputsForLink;
  private final NestedSet<Artifact> coverage;
  private final String compiler;
  private final String proto;
  private final String cpu;
  private final Artifact ifsoBuilder;
  private final Artifact linkDynamicLibraryTool;
  private final FdoProfileProvider fdoOptimizeProvider;
  private final TransitiveInfoCollection fdoOptimize;
  private final ImmutableList<Artifact> fdoOptimizeArtifacts;
  private final FdoPrefetchHintsProvider fdoPrefetch;
  private final TransitiveInfoCollection libcTop;
  private final TransitiveInfoCollection moduleMap;
  private final Artifact moduleMapArtifact;
  private final Artifact zipper;
  private final String purposePrefix;
  private final String runtimeSolibDirBase;
  private final ImmutableList<? extends TransitiveInfoCollection> staticRuntimesLibs;
  private final ImmutableList<? extends TransitiveInfoCollection> dynamicRuntimesLibs;
  private final LicensesProvider licensesProvider;
  private final Label toolchainType;
  private final CcToolchainVariables additionalBuildVariables;
  private final CcToolchainConfigInfo ccToolchainConfigInfo;
  private final String toolchainIdentifier;
  private final FdoProfileProvider fdoProfileProvider;
  private final Label ccToolchainLabel;

  public CcToolchainAttributesProvider(
      RuleContext ruleContext,
      boolean isAppleToolchain,
      CcToolchainVariables additionalBuildVariables) {
    super(PROVIDER);
    this.ccToolchainLabel = ruleContext.getLabel();
    this.toolchainIdentifier = ruleContext.attributes().get("toolchain_identifier", Type.STRING);
    this.cpu = ruleContext.attributes().get("cpu", Type.STRING);
    this.compiler = ruleContext.attributes().get("compiler", Type.STRING);
    this.proto = ruleContext.attributes().get("proto", Type.STRING);
    this.supportsParamFiles = ruleContext.attributes().get("supports_param_files", BOOLEAN);
    this.supportsHeaderParsing = ruleContext.attributes().get("supports_header_parsing", BOOLEAN);
    this.crosstool =
        ruleContext
            .getPrerequisite("all_files", Mode.HOST)
            .getProvider(FileProvider.class)
            .getFilesToBuild();
    this.crosstoolMiddleman = getMiddlemanOrFiles(ruleContext, "all_files");
    this.compile = getMiddlemanOrFiles(ruleContext, "compiler_files");
    this.compileWithoutIncludes =
        getOptionalMiddlemanOrFiles(ruleContext, "compiler_files_without_includes");
    this.strip = getMiddlemanOrFiles(ruleContext, "strip_files");
    this.objcopy = getMiddlemanOrFiles(ruleContext, "objcopy_files");
    this.as = getOptionalMiddlemanOrFiles(ruleContext, "as_files");
    this.ar = getOptionalMiddlemanOrFiles(ruleContext, "ar_files");
    this.link = getMiddlemanOrFiles(ruleContext, "linker_files");
    this.dwp = getMiddlemanOrFiles(ruleContext, "dwp_files");
    this.libcMiddleman =
        getOptionalMiddlemanOrFiles(ruleContext, CcToolchainRule.LIBC_TOP_ATTR, Mode.TARGET);
    this.libc = getOptionalFiles(ruleContext, CcToolchainRule.LIBC_TOP_ATTR, Mode.TARGET);
    this.fullInputsForCrosstool =
        NestedSetBuilder.<Artifact>stableOrder()
            .addTransitive(crosstoolMiddleman)
            .addTransitive(libcMiddleman)
            .build();
    ;
    this.fullInputsForLink = fullInputsForLink(ruleContext, link, libcMiddleman, isAppleToolchain);
    NestedSet<Artifact> coverageAttribute =
        getOptionalMiddlemanOrFiles(ruleContext, "coverage_files");
    if (coverageAttribute.isEmpty()) {
      this.coverage = Preconditions.checkNotNull(this.crosstool);
    } else {
      this.coverage = coverageAttribute;
    }
    this.ifsoBuilder = ruleContext.getPrerequisiteArtifact("$interface_library_builder", Mode.HOST);
    this.linkDynamicLibraryTool =
        ruleContext.getPrerequisiteArtifact("$link_dynamic_library_tool", Mode.HOST);
    this.fdoProfileProvider =
        ruleContext.getPrerequisite(
            CcToolchainRule.FDO_PROFILE_ATTR, Mode.TARGET, FdoProfileProvider.PROVIDER);
    this.fdoOptimizeProvider =
        ruleContext.getPrerequisite(
            CcToolchainRule.FDO_OPTIMIZE_ATTR, Mode.TARGET, FdoProfileProvider.PROVIDER);
    this.fdoOptimize = ruleContext.getPrerequisite(CcToolchainRule.FDO_OPTIMIZE_ATTR, Mode.TARGET);
    this.fdoOptimizeArtifacts =
        ruleContext.getPrerequisiteArtifacts(CcToolchainRule.FDO_OPTIMIZE_ATTR, Mode.TARGET).list();
    this.fdoPrefetch =
        ruleContext.getPrerequisite(
            ":fdo_prefetch_hints", Mode.TARGET, FdoPrefetchHintsProvider.PROVIDER);
    this.libcTop = ruleContext.getPrerequisite(CcToolchainRule.LIBC_TOP_ATTR, Mode.TARGET);
    this.moduleMap = ruleContext.getPrerequisite("module_map", Mode.HOST);
    this.moduleMapArtifact = ruleContext.getPrerequisiteArtifact("module_map", Mode.HOST);
    this.zipper = ruleContext.getPrerequisiteArtifact(":zipper", Mode.HOST);
    this.purposePrefix = Actions.escapeLabel(ruleContext.getLabel()) + "_";
    this.runtimeSolibDirBase = "_solib_" + "_" + Actions.escapeLabel(ruleContext.getLabel());
    this.staticRuntimesLibs =
        ImmutableList.copyOf(ruleContext.getPrerequisites("static_runtime_libs", Mode.TARGET));
    this.dynamicRuntimesLibs =
        ImmutableList.copyOf(ruleContext.getPrerequisites("dynamic_runtime_libs", Mode.TARGET));
    this.ccToolchainConfigInfo =
        ruleContext.getPrerequisite(
            CcToolchainRule.TOOLCHAIN_CONFIG_ATTR, Mode.TARGET, CcToolchainConfigInfo.PROVIDER);

    // If output_license is specified on the cc_toolchain rule, override the transitive licenses
    // with that one. This is necessary because cc_toolchain is used in the target configuration,
    // but it is sort-of-kind-of a tool, but various parts of it are linked into the output...
    // ...so we trust the judgment of the author of the cc_toolchain rule to figure out what
    // licenses should be propagated to C++ targets.
    // TODO(elenairina): Remove this and use Attribute.Builder.useOutputLicenses() on the
    // :cc_toolchain attribute instead.
    final License outputLicense =
        ruleContext.getRule().getToolOutputLicense(ruleContext.attributes());
    if (outputLicense != null && !outputLicense.equals(License.NO_LICENSE)) {
      final NestedSet<TargetLicense> license =
          NestedSetBuilder.create(
              Order.STABLE_ORDER, new TargetLicense(ruleContext.getLabel(), outputLicense));
      this.licensesProvider =
          new LicensesProviderImpl(
              license, new TargetLicense(ruleContext.getLabel(), outputLicense));
    } else {
      this.licensesProvider = null;
    }
    // TODO(b/65835260): Remove this conditional once j2objc can learn the toolchain type.
    if (ruleContext.attributes().has(CcToolchain.CC_TOOLCHAIN_TYPE_ATTRIBUTE_NAME)) {
      this.toolchainType =
          ruleContext.attributes().get(CcToolchain.CC_TOOLCHAIN_TYPE_ATTRIBUTE_NAME, NODEP_LABEL);
    } else {
      this.toolchainType = null;
    }
    this.additionalBuildVariables = additionalBuildVariables;
  }

  public String getCpu() {
    return cpu;
  }

  public boolean isSupportsParamFiles() {
    return supportsParamFiles;
  }

  public String getPurposePrefix() {
    return purposePrefix;
  }

  public String getRuntimeSolibDirBase() {
    return runtimeSolibDirBase;
  }

  public FdoPrefetchHintsProvider getFdoPrefetch() {
    return fdoPrefetch;
  }

  public String getToolchainIdentifier() {
    return toolchainIdentifier;
  }

  public Label getToolchainType() {
    return toolchainType;
  }

  public CcToolchainConfigInfo getCcToolchainConfigInfo() {
    return ccToolchainConfigInfo;
  }

  public ImmutableList<Artifact> getFdoOptimizeArtifacts() {
    return fdoOptimizeArtifacts;
  }

  public ImmutableList<? extends TransitiveInfoCollection> getStaticRuntimesLibs() {
    return staticRuntimesLibs;
  }

  public LicensesProvider getLicensesProvider() {
    return licensesProvider;
  }

  public ImmutableList<? extends TransitiveInfoCollection> getDynamicRuntimesLibs() {
    return dynamicRuntimesLibs;
  }

  public boolean isSupportsHeaderParsing() {
    return supportsHeaderParsing;
  }

  public CcToolchainVariables getAdditionalBuildVariables() {
    return additionalBuildVariables;
  }

  public NestedSet<Artifact> getCrosstool() {
    return crosstool;
  }

  public NestedSet<Artifact> getCrosstoolMiddleman() {
    return crosstoolMiddleman;
  }

  public NestedSet<Artifact> getCompile() {
    return compile;
  }

  public NestedSet<Artifact> getStrip() {
    return strip;
  }

  public NestedSet<Artifact> getObjcopy() {
    return objcopy;
  }

  public TransitiveInfoCollection getFdoOptimize() {
    return fdoOptimize;
  }

  public Artifact getLinkDynamicLibraryTool() {

    return linkDynamicLibraryTool;
  }

  public TransitiveInfoCollection getModuleMap() {
    return moduleMap;
  }

  public NestedSet<Artifact> getAs() {
    return as;
  }

  public NestedSet<Artifact> getAr() {
    return ar;
  }

  public TransitiveInfoCollection getLibcTop() {
    return libcTop;
  }

  public NestedSet<Artifact> getLink() {
    return link;
  }

  public NestedSet<Artifact> getDwp() {
    return dwp;
  }

  public FdoProfileProvider getFdoOptimizeProvider() {
    return fdoOptimizeProvider;
  }

  public Artifact getModuleMapArtifact() {
    return moduleMapArtifact;
  }

  public String getProto() {
    return proto;
  }

  public NestedSet<Artifact> getFullInputsForCrosstool() {
    return fullInputsForCrosstool;
  }

  public FdoProfileProvider getFdoProfileProvider() {
    return fdoProfileProvider;
  }

  public Artifact getZipper() {
    return zipper;
  }

  public NestedSet<Artifact> getFullInputsForLink() {
    return fullInputsForLink;
  }

  public Label getCcToolchainLabel() {
    return ccToolchainLabel;
  }

  public NestedSet<Artifact> getCoverage() {
    return coverage;
  }

  public NestedSet<Artifact> getCompileWithoutIncludes() {
    return compileWithoutIncludes;
  }

  public NestedSet<Artifact> getLibc() {
    return libc;
  }

  public String getCompiler() {
    return compiler;
  }

  public Artifact getIfsoBuilder() {
    return ifsoBuilder;
  }

  private static NestedSet<Artifact> getMiddlemanOrFiles(RuleContext context, String attribute) {
    return getMiddlemanOrFiles(context, attribute, Mode.HOST);
  }

  private static NestedSet<Artifact> getMiddlemanOrFiles(
      RuleContext context, String attribute, Mode mode) {
    TransitiveInfoCollection dep = context.getPrerequisite(attribute, mode);
    MiddlemanProvider middlemanProvider = dep.getProvider(MiddlemanProvider.class);
    // We use the middleman if we can (if the dep is a filegroup), otherwise, just the regular
    // filesToBuild (e.g. if it is a simple input file)
    return middlemanProvider != null
        ? middlemanProvider.getMiddlemanArtifact()
        : dep.getProvider(FileProvider.class).getFilesToBuild();
  }

  private static NestedSet<Artifact> getOptionalMiddlemanOrFiles(
      RuleContext context, String attribute) {
    return getOptionalMiddlemanOrFiles(context, attribute, Mode.HOST);
  }

  private static NestedSet<Artifact> getOptionalMiddlemanOrFiles(
      RuleContext context, String attribute, Mode mode) {
    TransitiveInfoCollection dep = context.getPrerequisite(attribute, mode);
    return dep != null
        ? getMiddlemanOrFiles(context, attribute, mode)
        : NestedSetBuilder.emptySet(Order.STABLE_ORDER);
  }

  private static NestedSet<Artifact> getOptionalFiles(
      RuleContext ruleContext, String attribute, Mode mode) {
    TransitiveInfoCollection dep = ruleContext.getPrerequisite(attribute, mode);
    return dep != null
        ? dep.getProvider(FileProvider.class).getFilesToBuild()
        : NestedSetBuilder.emptySet(Order.STABLE_ORDER);
  }

  /**
   * Returns the crosstool-derived link action inputs for a given rule. Adds the given set of
   * artifacts as extra inputs.
   */
  private static NestedSet<Artifact> fullInputsForLink(
      RuleContext ruleContext,
      NestedSet<Artifact> link,
      NestedSet<Artifact> libcMiddleman,
      boolean isAppleToolchain) {
    NestedSetBuilder<Artifact> builder =
        NestedSetBuilder.<Artifact>stableOrder().addTransitive(link).addTransitive(libcMiddleman);
    if (!isAppleToolchain) {
      builder
          .add(ruleContext.getPrerequisiteArtifact("$interface_library_builder", Mode.HOST))
          .add(ruleContext.getPrerequisiteArtifact("$link_dynamic_library_tool", Mode.HOST));
    }
    return builder.build();
  }
}
