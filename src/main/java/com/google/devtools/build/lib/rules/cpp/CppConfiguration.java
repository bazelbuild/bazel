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

package com.google.devtools.build.lib.rules.cpp;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.base.Verify;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.config.AutoCpuConverter;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.Options.MakeVariableSource;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.CompilationMode;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.config.PerLabelOptions;
import com.google.devtools.build.lib.analysis.skylark.annotations.SkylarkConfigurationField;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.rules.cpp.CppConfigurationLoader.CppConfigurationParameters;
import com.google.devtools.build.lib.rules.cpp.Link.LinkingMode;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.CppConfigurationApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CrosstoolRelease;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * This class represents the C/C++ parts of the {@link BuildConfiguration}, including the host
 * architecture, target architecture, compiler version, and a standard library version. It has
 * information about the tools locations and the flags required for compiling.
 *
 * <p>Before {@link CppConfiguration} is created, two things need to be done:
 *
 * <ol>
 *   <li>choosing a {@link CcToolchainRule} label from {@code toolchains} map attribute of {@link
 *       CcToolchainSuiteRule}.
 *   <li>selection of a {@link
 *       com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain} from the
 *       CROSSTOOL file.
 * </ol>
 *
 * <p>The process goes as follows:
 *
 * <p>Check for existence of {@link CcToolchainSuiteRule}.toolchains[<cpu>|<compiler>], if
 * --compiler is specified, otherwise check for {@link CcToolchainSuiteRule}.toolchains[<cpu>].
 *
 * <ul>
 *   <li>if a value is found, load the {@link CcToolchainRule} rule and look for the {@code
 *       toolchain_identifier} attribute.
 *   <li>
 *       <ul>
 *         <li>if the attribute exists, loop through all the {@code CToolchain}s in CROSSTOOL and
 *             select the one with the matching toolchain identifier.
 *         <li>otherwise fall back to selecting the CToolchain from CROSSTOOL by matching the --cpu
 *             and --compiler values.
 *       </ul>
 *   <li>If a value is not found, select the CToolchain from CROSSTOOL by matching the --cpu and
 *       --compiler values, and construct the key as follows: <toolchain.cpu>|<toolchain.compiler>.
 * </ul>
 */
@Immutable
public final class CppConfiguration extends BuildConfiguration.Fragment
    implements CppConfigurationApi<InvalidConfigurationException> {
  /**
   * String indicating a Mac system, for example when used in a crosstool configuration's host or
   * target system name.
   */
  public static final String MAC_SYSTEM_NAME = "x86_64-apple-macosx";

  /** String constant for CC_FLAGS make variable name */
  public static final String CC_FLAGS_MAKE_VARIABLE_NAME = "CC_FLAGS";

  /**
   * An enumeration of all the tools that comprise a toolchain.
   */
  public enum Tool {
    AR("ar"),
    CPP("cpp"),
    GCC("gcc"),
    GCOV("gcov"),
    GCOVTOOL("gcov-tool"),
    LD("ld"),
    NM("nm"),
    OBJCOPY("objcopy"),
    OBJDUMP("objdump"),
    STRIP("strip"),
    DWP("dwp"),
    LLVM_PROFDATA("llvm-profdata");

    private final String namePart;

    private Tool(String namePart) {
      this.namePart = namePart;
    }

    public String getNamePart() {
      return namePart;
    }
  }

  /**
   * Values for the --hdrs_check option. Note that Bazel only supports and will default to "strict".
   */
  public enum HeadersCheckingMode {
    /**
     * Legacy behavior: Silently allow any source header file in any of the directories of the
     * containing package to be included by sources in this rule and dependent rules.
     */
    LOOSE,
    /** Disallow undeclared headers. */
    STRICT;

    public static HeadersCheckingMode getValue(String value) {
      if (value.equalsIgnoreCase("loose") || value.equalsIgnoreCase("warn")) {
        return LOOSE;
      }
      if (value.equalsIgnoreCase("strict")) {
        return STRICT;
      }
      throw new IllegalArgumentException();
    }
  }

  /**
   * --dynamic_mode parses to DynamicModeFlag, but AUTO will be translated based on platform,
   * resulting in a DynamicMode value.
   */
  public enum DynamicMode     { OFF, DEFAULT, FULLY }

  /**
   * This enumeration is used for the --strip option.
   */
  public enum StripMode {

    ALWAYS("always"),       // Always strip.
    SOMETIMES("sometimes"), // Strip iff compilationMode == FASTBUILD.
    NEVER("never");         // Never strip.

    private final String mode;

    private StripMode(String mode) {
      this.mode = mode;
    }

    @Override
    public String toString() {
      return mode;
    }
  }

  /**
   * This macro will be passed as a command-line parameter (eg. -DBUILD_FDO_TYPE="AUTOFDO"). For
   * possible values see {@code CppModel.getFdoBuildStamp()}.
   */
  public static final String FDO_STAMP_MACRO = "BUILD_FDO_TYPE";

  private final Label crosstoolTop;
  /**
   * cc_toolchain_suite allows to override CROSSTOOL by using proto attribute. This attribute value
   * is stored here so cc_toolchain can access it in the analysis. Don't use this for anything, it
   * will be removed when b/113849758 is fixed. If you do, I'll send bubo to take your keyboard
   * away.
   */
  @Deprecated private final CrosstoolRelease crosstoolFromCcToolchainProtoAttribute;

  private final String transformedCpuFromOptions;
  private final String compilerFromOptions;
  // TODO(lberki): desiredCpu *should* be always the same as targetCpu, except that we don't check
  // that the CPU we get from the toolchain matches BuildConfiguration.Options.cpu . So we store
  // it here so that the output directory doesn't depend on the CToolchain. When we will eventually
  // verify that the two are the same, we can remove one of desiredCpu and targetCpu.
  private final String desiredCpu;
  private final PathFragment crosstoolTopPathFragment;

  private final PathFragment fdoPath;
  private final Label fdoOptimizeLabel;

  // TODO(bazel-team): All these labels (except for ccCompilerRuleLabel) can be removed once the
  // transition to the cc_compiler rule is complete.
  private final Label ccToolchainLabel;
  private final Label stlLabel;

  // TODO(kmensah): This is temporary until all the Skylark functions that need this can be removed.
  private final PathFragment nonConfiguredSysroot;
  private final Label sysrootLabel;

  private final ImmutableList<String> compilerFlags;
  private final ImmutableList<String> cxxFlags;
  private final ImmutableList<String> unfilteredCompilerFlags;
  private final ImmutableList<String> conlyopts;

  private final ImmutableList<String> mostlyStaticLinkFlags;
  private final ImmutableList<String> mostlyStaticSharedLinkFlags;
  private final ImmutableList<String> dynamicLinkFlags;
  private final ImmutableList<String> copts;
  private final ImmutableList<String> cxxopts;

  private final ImmutableList<String> linkopts;
  private final ImmutableList<String> ltoindexOptions;
  private final ImmutableList<String> ltobackendOptions;

  private final CppOptions cppOptions;

  // The dynamic mode for linking.
  private final boolean stripBinaries;
  private final CompilationMode compilationMode;

  private final boolean shouldProvideMakeVariables;

  private final CppToolchainInfo cppToolchainInfo;

  static CppConfiguration create(CppConfigurationParameters params)
      throws InvalidConfigurationException {
    CppOptions cppOptions = params.cppOptions;
    PathFragment crosstoolTopPathFragment =
        params.crosstoolTop.getPackageIdentifier().getPathUnderExecRoot();
    CppToolchainInfo cppToolchainInfo =
        CppToolchainInfo.create(
            crosstoolTopPathFragment,
            params.ccToolchainLabel,
            params.ccToolchainConfigInfo,
            cppOptions.disableLegacyCrosstoolFields,
            cppOptions.disableCompilationModeFlags,
            cppOptions.disableLinkingModeFlags);

    CompilationMode compilationMode = params.commonOptions.compilationMode;

    ImmutableList.Builder<String> coptsBuilder =
        ImmutableList.<String>builder()
            .addAll(cppToolchainInfo.getCompilerFlags())
            .addAll(cppToolchainInfo.getCFlagsByCompilationMode().get(compilationMode));
    if (cppOptions.experimentalOmitfp) {
      coptsBuilder.add("-fomit-frame-pointer");
      coptsBuilder.add("-fasynchronous-unwind-tables");
      coptsBuilder.add("-DNO_FRAME_POINTER");
    }
    coptsBuilder.addAll(cppOptions.coptList);

    ImmutableList<String> cxxOpts =
        ImmutableList.<String>builder()
            .addAll(cppToolchainInfo.getCxxFlags())
            .addAll(cppToolchainInfo.getCxxFlagsByCompilationMode().get(compilationMode))
            .addAll(cppOptions.cxxoptList)
            .build();

    ImmutableList.Builder<String> linkoptsBuilder = ImmutableList.builder();
    linkoptsBuilder.addAll(cppOptions.linkoptList);
    if (cppOptions.experimentalOmitfp) {
      linkoptsBuilder.add("-Wl,--eh-frame-hdr");
    }

    return new CppConfiguration(
        params.crosstoolTop,
        params.crosstoolFromCcToolchainProtoAttribute,
        params.transformedCpu,
        params.compiler,
        Preconditions.checkNotNull(params.commonOptions.cpu),
        crosstoolTopPathFragment,
        params.fdoPath,
        params.fdoOptimizeLabel,
        params.ccToolchainLabel,
        params.stlLabel,
        params.sysrootLabel == null
            ? cppToolchainInfo.getDefaultSysroot()
            : params.sysrootLabel.getPackageFragment(),
        params.sysrootLabel,
        coptsBuilder.build(),
        cxxOpts,
        cppToolchainInfo.getUnfilteredCompilerOptions(/* sysroot= */ null),
        ImmutableList.copyOf(cppOptions.conlyoptList),
        cppToolchainInfo.configureAllLegacyLinkOptions(compilationMode, LinkingMode.STATIC),
        cppToolchainInfo.configureAllLegacyLinkOptions(
            compilationMode, LinkingMode.LEGACY_MOSTLY_STATIC_LIBRARIES),
        cppToolchainInfo.configureAllLegacyLinkOptions(compilationMode, LinkingMode.DYNAMIC),
        ImmutableList.copyOf(cppOptions.coptList),
        ImmutableList.copyOf(cppOptions.cxxoptList),
        linkoptsBuilder.build(),
        ImmutableList.copyOf(cppOptions.ltoindexoptList),
        ImmutableList.copyOf(cppOptions.ltobackendoptList),
        cppOptions,
        (cppOptions.stripBinaries == StripMode.ALWAYS
            || (cppOptions.stripBinaries == StripMode.SOMETIMES
                && compilationMode == CompilationMode.FASTBUILD)),
        compilationMode,
        params.commonOptions.makeVariableSource == MakeVariableSource.CONFIGURATION,
        cppToolchainInfo);
  }

  private CppConfiguration(
      Label crosstoolTop,
      CrosstoolRelease crosstoolFromCcToolchainProtoAttribute,
      String transformedCpuFromOptions,
      String compilerFromOptions,
      String desiredCpu,
      PathFragment crosstoolTopPathFragment,
      PathFragment fdoPath,
      Label fdoOptimizeLabel,
      Label ccToolchainLabel,
      Label stlLabel,
      PathFragment nonConfiguredSysroot,
      Label sysrootLabel,
      ImmutableList<String> compilerFlags,
      ImmutableList<String> cxxFlags,
      ImmutableList<String> unfilteredCompilerFlags,
      ImmutableList<String> conlyopts,
      ImmutableList<String> mostlyStaticLinkFlags,
      ImmutableList<String> mostlyStaticSharedLinkFlags,
      ImmutableList<String> dynamicLinkFlags,
      ImmutableList<String> copts,
      ImmutableList<String> cxxopts,
      ImmutableList<String> linkopts,
      ImmutableList<String> ltoindexOptions,
      ImmutableList<String> ltobackendOptions,
      CppOptions cppOptions,
      boolean stripBinaries,
      CompilationMode compilationMode,
      boolean shouldProvideMakeVariables,
      CppToolchainInfo cppToolchainInfo) {
    this.crosstoolTop = crosstoolTop;
    this.crosstoolFromCcToolchainProtoAttribute = crosstoolFromCcToolchainProtoAttribute;
    this.transformedCpuFromOptions = transformedCpuFromOptions;
    this.compilerFromOptions = compilerFromOptions;
    this.desiredCpu = desiredCpu;
    this.crosstoolTopPathFragment = crosstoolTopPathFragment;
    this.fdoPath = fdoPath;
    this.fdoOptimizeLabel = fdoOptimizeLabel;
    this.ccToolchainLabel = ccToolchainLabel;
    this.stlLabel = stlLabel;
    this.nonConfiguredSysroot = nonConfiguredSysroot;
    this.sysrootLabel = sysrootLabel;
    this.compilerFlags = compilerFlags;
    this.cxxFlags = cxxFlags;
    this.unfilteredCompilerFlags = unfilteredCompilerFlags;
    this.conlyopts = conlyopts;
    this.mostlyStaticLinkFlags = mostlyStaticLinkFlags;
    this.mostlyStaticSharedLinkFlags = mostlyStaticSharedLinkFlags;
    this.dynamicLinkFlags = dynamicLinkFlags;
    this.copts = copts;
    this.cxxopts = cxxopts;
    this.linkopts = linkopts;
    this.ltoindexOptions = ltoindexOptions;
    this.ltobackendOptions = ltobackendOptions;
    this.cppOptions = cppOptions;
    this.stripBinaries = stripBinaries;
    this.compilationMode = compilationMode;
    this.shouldProvideMakeVariables = shouldProvideMakeVariables;
    this.cppToolchainInfo = cppToolchainInfo;
  }

  @VisibleForTesting
  static LinkingMode importLinkingMode(CrosstoolConfig.LinkingMode mode) {
    switch (mode.name()) {
      case "FULLY_STATIC":
        return LinkingMode.LEGACY_FULLY_STATIC;
      case "MOSTLY_STATIC_LIBRARIES":
        return LinkingMode.LEGACY_MOSTLY_STATIC_LIBRARIES;
      case "MOSTLY_STATIC":
        return LinkingMode.STATIC;
      case "DYNAMIC":
        return LinkingMode.DYNAMIC;
      default:
        throw new IllegalArgumentException(
            String.format("Linking mode '%s' not known.", mode.name()));
    }
  }

  /** Returns the {@link CppToolchainInfo} used by this configuration. */
  public CppToolchainInfo getCppToolchainInfo() {
    return cppToolchainInfo;
  }

  /**
   * Returns the toolchain identifier, which uniquely identifies the compiler version, target libc
   * version, and target cpu.
   */
  public String getToolchainIdentifier() {
    return cppToolchainInfo.getToolchainIdentifier();
  }

  /** Returns the label of the CROSSTOOL for this configuration. */
  public Label getCrosstoolTop() {
    return crosstoolTop;
  }

  /**
   * Returns the path of the crosstool.
   */
  public PathFragment getCrosstoolTopPathFragment() {
    return cppToolchainInfo.getCrosstoolTopPathFragment();
  }

  @Override
  public String toString() {
    return cppToolchainInfo.toString();
  }

  /**
   * Returns the compiler version string (e.g. "gcc-4.1.1").
   *
   * <p>Deprecated: Use {@link CcToolchainProvider#getCompiler()}
   */
  // TODO(b/68038647): Remove once make variables are no longer derived from CppConfiguration.
  @Override
  @Deprecated
  public String getCompiler() throws EvalException {
    checkForToolchainSkylarkApiAvailability();
    return cppToolchainInfo.getCompiler();
  }

  /**
   * Returns the libc version string (e.g. "glibc-2.2.2").
   *
   * <p>Deprecated: Use {@link CcToolchainProvider#getTargetLibc()}
   */
  // TODO(b/68038647): Remove once make variables are no longer derived from CppConfiguration.
  @Override
  @Deprecated
  public String getTargetLibc() throws EvalException {
    checkForToolchainSkylarkApiAvailability();
    return cppToolchainInfo.getTargetLibc();
  }

  /**
   * Returns the target architecture using blaze-specific constants (e.g. "piii").
   *
   * <p>Deprecated: Use {@link CcToolchainProvider#getTargetCpu()}
   */
  // TODO(b/68038647): Remove once skylark callers are migrated.
  @Override
  @Deprecated
  public String getTargetCpu() throws EvalException {
    checkForToolchainSkylarkApiAvailability();
    return cppToolchainInfo.getTargetCpu();
  }

  /**
   * Returns the path fragment that is either absolute or relative to the execution root that can be
   * used to execute the given tool.
   *
   * <p>Deprecated: Use {@link CcToolchainProvider#getToolPathFragment(Tool)}
   */
  @Deprecated
  public PathFragment getToolPathFragment(CppConfiguration.Tool tool) {
    return cppToolchainInfo.getToolPathFragment(tool);
  }

  /** Returns the label of the <code>cc_compiler</code> rule for the C++ configuration. */
  @SkylarkConfigurationField(
      name = "cc_toolchain",
      doc = "The label of the target describing the C++ toolchain",
      defaultLabel = "//tools/cpp:crosstool",
      defaultInToolRepository = true)
  public Label getRuleProvidingCcToolchainProvider() {
    return ccToolchainLabel;
  }

  /**
   * Returns the configured features of the toolchain. Rules should not call this directly, but
   * instead use {@code CcToolchainProvider.getFeatures}.
   */
  public CcToolchainFeatures getFeatures() {
    return cppToolchainInfo.getFeatures();
  }

  /**
   * Returns the configured current compilation mode. Rules should not call this directly, but
   * instead use {@code CcToolchainProvider.getCompilationMode}.
   */
  public CompilationMode getCompilationMode() {
    return compilationMode;
  }

  @Override
  @Deprecated
  public ImmutableList<String> getBuiltInIncludeDirectoriesForSkylark()
      throws InvalidConfigurationException, EvalException {
    checkForToolchainSkylarkApiAvailability();
    return getBuiltInIncludeDirectories(nonConfiguredSysroot)
            .stream()
            .map(PathFragment::getPathString)
            .collect(ImmutableList.toImmutableList());
  }

  /**
   * Returns the built-in list of system include paths for the toolchain compiler. All paths in this
   * list should be relative to the exec directory. They may be absolute if they are also installed
   * on the remote build nodes or for local compilation.
   *
   * <p>TODO(b/64384912): Migrate skylark callers to
   * CcToolchainProvider#getBuiltinIncludeDirectories and delete this method.
   */
  private ImmutableList<PathFragment> getBuiltInIncludeDirectories(PathFragment sysroot)
      throws InvalidConfigurationException {
    ImmutableList.Builder<PathFragment> builtInIncludeDirectoriesBuilder = ImmutableList.builder();
    for (String s : cppToolchainInfo.getRawBuiltInIncludeDirectories()) {
      builtInIncludeDirectoriesBuilder.add(
          CcToolchainProviderHelper.resolveIncludeDir(s, sysroot, crosstoolTopPathFragment));
    }
    return builtInIncludeDirectoriesBuilder.build();
  }

  /**
   * Returns the sysroot to be used. If the toolchain compiler does not support
   * different sysroots, or the sysroot is the same as the default sysroot, then
   * this method returns <code>null</code>.
   */
  @Override
  @Deprecated
  public String getSysroot() throws EvalException {
    checkForToolchainSkylarkApiAvailability();
    return nonConfiguredSysroot.getPathString();
  }

  public Label getSysrootLabel() {
    return sysrootLabel;
  }

  /**
   * Returns the default options to use for compiling C, C++, and assembler. This is just the
   * options that should be used for all three languages. There may be additional C-specific or
   * C++-specific options that should be used, in addition to the ones returned by this method.
   *
   * <p>Deprecated: Use {@link CcToolchainProvider#getLegacyCompileOptionsWithCopts()}
   */
  // TODO(b/64384912): Migrate skylark callers and remove.
  @Override
  @Deprecated
  public ImmutableList<String> getCompilerOptions(Iterable<String> featuresNotUsedAnymore)
    throws EvalException {
    checkForToolchainSkylarkApiAvailability();
    checkForLegacyCompilationApiAvailability();
    return compilerFlags;
  }

  /**
   * Returns the list of additional C-specific options to use for compiling C. These should be go on
   * the command line after the common options returned by {@link #getCompilerOptions}.
   */
  // TODO(b/64384912): Migrate skylark callers and remove.
  @Override
  @Deprecated
  public ImmutableList<String> getCOptionsForSkylark() throws EvalException {
    checkForToolchainSkylarkApiAvailability();
    checkForLegacyCompilationApiAvailability();
    return getCOptions();
  }

  public ImmutableList<String> getCOptions() {
    return conlyopts;
  }

  /**
   * Returns the list of additional C++-specific options to use for compiling C++. These should be
   * on the command line after the common options returned by {@link #getCompilerOptions}.
   *
   * <p>Deprecated: Use {@link CcToolchainProvider#getCxxOptionsWithCopts}
   */
  // TODO(b/64384912): Migrate skylark callers and remove.
  @Override
  @Deprecated
  public ImmutableList<String> getCxxOptions(Iterable<String> featuresNotUsedAnymore)
    throws EvalException {
    checkForToolchainSkylarkApiAvailability();
    checkForLegacyCompilationApiAvailability();
    return cxxFlags;
  }

  /**
   * Returns the default list of options which cannot be filtered by BUILD rules. These should be
   * appended to the command line after filtering.
   *
   * @deprecated since it uses nonconfigured sysroot. Use {@link
   *     CcToolchainProvider#getUnfilteredCompilerOptionsWithSysroot(Iterable)} if you *really* need
   *     to.
   */
  // TODO(b/65401585): Migrate existing uses to cc_toolchain and cleanup here.
  @Deprecated
  @Override
  public ImmutableList<String> getUnfilteredCompilerOptionsWithLegacySysroot(
      Iterable<String> featuresNotUsedAnymore) throws EvalException {
    checkForToolchainSkylarkApiAvailability();
    checkForLegacyCompilationApiAvailability();
    return getUnfilteredCompilerOptionsDoNotUse(nonConfiguredSysroot);
  }

  /**
   * @deprecated since it hardcodes --sysroot flag. Use {@link
   *     com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration} instead.
   */
  // TODO(b/65401585): Migrate existing uses to cc_toolchain and cleanup here.
  @Deprecated
  ImmutableList<String> getUnfilteredCompilerOptionsDoNotUse(@Nullable PathFragment sysroot)
    throws EvalException {
    checkForToolchainSkylarkApiAvailability();
    if (sysroot == null) {
      return unfilteredCompilerFlags;
    }
    return ImmutableList.<String>builder()
        .add("--sysroot=" + sysroot)
        .addAll(unfilteredCompilerFlags)
        .build();
  }

  /**
   * Returns the set of command-line linker options, including any flags inferred from the
   * command-line options.
   *
   * @see Link
   * @deprecated since it uses nonconfigured sysroot. Use
   * {@link CcToolchainProvider#getLinkOptionsWithSysroot()} if you *really* need to.
   */
  // TODO(b/65401585): Migrate existing uses to cc_toolchain and cleanup here.
  @Deprecated
  @Override
  public ImmutableList<String> getLinkOptionsWithLegacySysroot() throws EvalException {
    checkForToolchainSkylarkApiAvailability();
    checkForLegacyLinkingApiAvailability();
    return getLinkOptionsDoNotUse(nonConfiguredSysroot);
  }

  /**
   * @deprecated since it hardcodes --sysroot flag. Use
   * {@link com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration}
   * instead.
   */
  // TODO(b/65401585): Migrate existing uses to cc_toolchain and cleanup here.
  @Deprecated
  ImmutableList<String> getLinkOptionsDoNotUse(@Nullable PathFragment sysroot) {
    if (sysroot == null) {
      return linkopts;
    } else {
      return ImmutableList.<String>builder().addAll(linkopts).add("--sysroot=" + sysroot).build();
    }
  }

  public boolean hasSharedLinkOption() {
    return linkopts.contains("-shared");
  }

  /** Returns the set of command-line LTO indexing options. */
  public ImmutableList<String> getLtoIndexOptions() {
    return ltoindexOptions;
  }

  /** Returns the set of command-line LTO backend options. */
  public ImmutableList<String> getLtoBackendOptions() {
    return ltobackendOptions;
  }

  /**
   * Returns the immutable list of linker options for fully statically linked outputs. Does not
   * include command-line options passed via --linkopt or --linkopts.
   *
   * @param featuresNotUsedAnymore
   * @param sharedLib true if the output is a shared lib, false if it's an executable
   *     <p>Deprecated: Use {@link CppHelper#getFullyStaticLinkOptions(CppConfiguration,
   *     CcToolchainProvider, boolean)}
   */
  // TODO(b/64384912): Migrate skylark users to cc_common and remove.
  @Override
  @Deprecated
  public ImmutableList<String> getFullyStaticLinkOptions(
      Iterable<String> featuresNotUsedAnymore, Boolean sharedLib) throws EvalException {
    checkForToolchainSkylarkApiAvailability();
    checkForLegacyLinkingApiAvailability();
    if (!sharedLib) {
      throw new EvalException(
          Location.BUILTIN, "fully_static_link_options is deprecated, new uses are not allowed.");
    }
    return getSharedLibraryLinkOptions(mostlyStaticLinkFlags);
  }

  /**
   * Returns the immutable list of linker options for mostly statically linked outputs. Does not
   * include command-line options passed via --linkopt or --linkopts.
   *
   * @param featuresNotUsedAnymore
   * @param sharedLib true if the output is a shared lib, false if it's an executable
   *     <p>Deprecated: Use {@link CppHelper#getMostlyStaticLinkOptions( CppConfiguration,
   *     CcToolchainProvider, boolean, boolean)}
   */
  // TODO(b/64384912): Migrate skylark users to cc_common and remove.
  @Override
  @Deprecated
  public ImmutableList<String> getMostlyStaticLinkOptions(
      Iterable<String> featuresNotUsedAnymore, Boolean sharedLib) throws EvalException {
    checkForToolchainSkylarkApiAvailability();
    checkForLegacyLinkingApiAvailability();
    if (sharedLib) {
      return getSharedLibraryLinkOptions(
          cppToolchainInfo.supportsEmbeddedRuntimes()
              ? mostlyStaticSharedLinkFlags
              : dynamicLinkFlags);
    } else {
      return mostlyStaticLinkFlags;
    }
  }

  /**
   * Returns the immutable list of linker options for artifacts that are not fully or mostly
   * statically linked. Does not include command-line options passed via --linkopt or --linkopts.
   *
   * @param featuresNotUsedAnymore
   * @param sharedLib true if the output is a shared lib, false if it's an executable
   *     <p>Deprecated: Use {@link CppHelper#getDynamicLinkOptions(CppConfiguration,
   *     CcToolchainProvider, Boolean)}
   */
  // TODO(b/64384912): Migrate skylark users to cc_common and remove.
  @Override
  @Deprecated
  public ImmutableList<String> getDynamicLinkOptions(
      Iterable<String> featuresNotUsedAnymore, Boolean sharedLib) throws EvalException {
    checkForToolchainSkylarkApiAvailability();
    checkForLegacyLinkingApiAvailability();
    if (sharedLib) {
      return getSharedLibraryLinkOptions(dynamicLinkFlags);
    } else {
      return dynamicLinkFlags;
    }
  }

  /**
   * Returns link options for the specified flag list, combined with universal options for all
   * shared libraries (regardless of link staticness).
   *
   * <p>Deprecated: Use {@link CcToolchainProvider#getSharedLibraryLinkOptions}
   */
  // TODO(b/64384912): Migrate skylark dependants and delete.
  private ImmutableList<String> getSharedLibraryLinkOptions(ImmutableList<String> flags) {
    return cppToolchainInfo.getSharedLibraryLinkOptions(flags);
  }

  /**
   * Returns a map of additional make variables for use by {@link
   * BuildConfiguration}. These are to used to allow some build rules to
   * avoid the limits on stack frame sizes and variable-length arrays.
   *
   * <p>The returned map must contain an entry for {@code STACK_FRAME_UNLIMITED},
   * though the entry may be an empty string.
   */
  public ImmutableMap<String, String> getAdditionalMakeVariables() {
    return cppToolchainInfo.getAdditionalMakeVariables();
  }

  /**
   * Returns the execution path to the linker binary to use for this build. Relative paths are
   * relative to the execution root.
   */
  @Override
  @Deprecated
  public String getLdExecutableForSkylark() throws EvalException {
    checkForToolchainSkylarkApiAvailability();
    PathFragment ldExecutable = getToolPathFragment(CppConfiguration.Tool.LD);
    return ldExecutable != null ? ldExecutable.getPathString() : "";
  }

  @SkylarkCallable(
      name = "minimum_os_version",
      doc = "The minimum OS version for C/C++ compilation.")
  public String getMinimumOsVersion() {
    return cppOptions.minimumOsVersion;
  }

  /** Returns the value of the --dynamic_mode flag. */
  public DynamicMode getDynamicModeFlag() {
    return cppOptions.dynamicMode;
  }

  public boolean getLinkCompileOutputSeparately() {
    return cppOptions.linkCompileOutputSeparately;
  }

  /**
   * Returns the STL label if given on the command line. {@code null}
   * otherwise.
   */
  public Label getStl() {
    return stlLabel;
  }

  @SkylarkConfigurationField(
      name = "stl",
      doc = "The label of the STL target",
      defaultLabel = "//third_party/stl",
      defaultInToolRepository = false
  )
  public Label getSkylarkStl() {
    if (stlLabel == null) {
      try {
        return Label.parseAbsolute("//third_party/stl", ImmutableMap.of());
      } catch (LabelSyntaxException e) {
        throw new IllegalStateException("STL label not formatted correctly", e);
      }
    }
    return stlLabel;
  }

  public boolean isFdo() {
    return cppOptions.isFdo();
  }

  /**
   * Returns whether or not to strip the binaries.
   */
  public boolean shouldStripBinaries() {
    return stripBinaries;
  }

  /**
   * Returns the additional options to pass to strip when generating a
   * {@code <name>.stripped} binary by this build.
   */
  public ImmutableList<String> getStripOpts() {
    return ImmutableList.copyOf(cppOptions.stripoptList);
  }

  /**
   * Returns whether temporary outputs from gcc will be saved.
   */
  public boolean getSaveTemps() {
    return cppOptions.saveTemps;
  }

  /**
   * Returns the {@link PerLabelOptions} to apply to the gcc command line, if
   * the label of the compiled file matches the regular expression.
   */
  public ImmutableList<PerLabelOptions> getPerFileCopts() {
    return ImmutableList.copyOf(cppOptions.perFileCopts);
  }

  /**
   * Returns the {@link PerLabelOptions} to apply to the LTO Backend command line, if the compiled
   * object matches the regular expression.
   */
  public ImmutableList<PerLabelOptions> getPerFileLtoBackendOpts() {
    return ImmutableList.copyOf(cppOptions.perFileLtoBackendOpts);
  }

  /**
   * Returns the custom malloc library label.
   */
  public Label customMalloc() {
    return cppOptions.customMalloc;
  }

  /**
   * Returns whether we are processing headers in dependencies of built C++ targets.
   */
  public boolean processHeadersInDependencies() {
    return cppOptions.processHeadersInDependencies;
  }

  /** Returns true if --fission contains the current compilation mode. */
  public boolean fissionIsActiveForCurrentCompilationMode() {
    return cppOptions.fissionModes.contains(compilationMode);
  }

  /** Returns true if --build_test_dwp is set on this build. */
  public boolean buildTestDwpIsActivated() {
    return cppOptions.buildTestDwp;
  }

  /**
   * Returns true if all C++ compilations should produce position-independent code, links should
   * produce position-independent executables, and dependencies with equivalent pre-built pic and
   * nopic versions should apply the pic versions. Returns false if default settings should be
   * applied (i.e. make no special provisions for pic code).
   */
  public boolean forcePic() {
    return cppOptions.forcePic;
  }

  /** Returns true if --start_end_lib is set on this build. */
  public boolean startEndLibIsRequested() {
    return cppOptions.useStartEndLib;
  }

  /**
   * @return value from the --cpu option transformed using {@link CpuTransformer}. If it was not
   *     passed explicitly, {@link AutoCpuConverter} will try to guess something reasonable.
   */
  public String getTransformedCpuFromOptions() {
    return transformedCpuFromOptions;
  }

  /** @return value from --compiler option, null if the option was not passed. */
  @Nullable
  public String getCompilerFromOptions() {
    return compilerFromOptions;
  }

  public boolean legacyWholeArchive() {
    return cppOptions.legacyWholeArchive;
  }

  public boolean getSymbolCounts() {
    return cppOptions.symbolCounts;
  }

  public boolean getInmemoryDotdFiles() {
    return cppOptions.inmemoryDotdFiles;
  }

  public boolean getPruneCppModules() {
    return cppOptions.pruneCppModules;
  }

  public boolean getPruneCppInputDiscovery() {
    return cppOptions.pruneCppInputDiscovery;
  }

  public boolean getParseHeadersVerifiesModules() {
    return cppOptions.parseHeadersVerifiesModules;
  }

  public boolean getUseInterfaceSharedObjects() {
    return cppOptions.useInterfaceSharedObjects;
  }

  /**
   * Returns the path to the GNU binutils 'objcopy' binary to use for this build. (Corresponds to
   * $(OBJCOPY) in make-dbg.) Relative paths are relative to the execution root.
   */
  @Override
  @Deprecated
  public String getObjCopyExecutableForSkylark() throws EvalException {
    checkForToolchainSkylarkApiAvailability();
    PathFragment objCopyExecutable = getToolPathFragment(Tool.OBJCOPY);
    return objCopyExecutable != null ? objCopyExecutable.getPathString() : "";
  }

  @Override
  @Deprecated
  public String getCppExecutableForSkylark() throws EvalException {
    checkForToolchainSkylarkApiAvailability();
    PathFragment cppExecutable = getToolPathFragment(Tool.GCC);
    return cppExecutable != null ? cppExecutable.getPathString() : "";
  }

  @Override
  @Deprecated
  public String getCpreprocessorExecutableForSkylark() throws EvalException {
    checkForToolchainSkylarkApiAvailability();
    PathFragment cpreprocessorExecutable = getToolPathFragment(Tool.CPP);
    return cpreprocessorExecutable != null ? cpreprocessorExecutable.getPathString() : "";
  }

  @Override
  @Deprecated
  public String getNmExecutableForSkylark() throws EvalException {
    checkForToolchainSkylarkApiAvailability();
    PathFragment nmExecutable = getToolPathFragment(Tool.NM);
    return nmExecutable != null ? nmExecutable.getPathString() : "";
  }

  @Override
  @Deprecated
  public String getObjdumpExecutableForSkylark() throws EvalException {
    checkForToolchainSkylarkApiAvailability();
    PathFragment objdumpExecutable = getToolPathFragment(Tool.OBJDUMP);
    return objdumpExecutable != null ? objdumpExecutable.getPathString() : "";
  }

  @Override
  @Deprecated
  public String getArExecutableForSkylark() throws EvalException {
    checkForToolchainSkylarkApiAvailability();
    PathFragment arExecutable = getToolPathFragment(Tool.AR);
    return arExecutable != null ? arExecutable.getPathString() : "";
  }

  @Override
  @Deprecated
  public String getStripExecutableForSkylark() throws EvalException {
    checkForToolchainSkylarkApiAvailability();
    PathFragment stripExecutable = getToolPathFragment(Tool.STRIP);
    return stripExecutable != null ? stripExecutable.getPathString() : "";
  }

  /**
   * Returns the GNU System Name
   *
   */
  //TODO(b/70225490): Migrate skylark dependants to CcToolchainProvider and delete.
  @Override
  @Deprecated
  public String getTargetGnuSystemName() throws EvalException {
    checkForToolchainSkylarkApiAvailability();
    return cppToolchainInfo.getTargetGnuSystemName();
  }

  /** Returns whether this configuration will use libunwind for stack unwinding. */
  public boolean isOmitfp() {
    return cppOptions.experimentalOmitfp;
  }

  /** Returns flags passed to Bazel by --copt option. */
  @Override
  public ImmutableList<String> getCopts() {
    return copts;
  }

  /** Returns flags passed to Bazel by --cxxopt option. */
  @Override
  public ImmutableList<String> getCxxopts() {
    return cxxopts;
  }

  /** Returns flags passed to Bazel by --conlyopt option. */
  @Override
  public ImmutableList<String> getConlyopts() {
    return conlyopts;
  }

  /** Returns flags passed to Bazel by --linkopt option. */
  @Override
  public ImmutableList<String> getLinkopts() {
    return linkopts;
  }

  @Override
  public void reportInvalidOptions(EventHandler reporter, BuildOptions buildOptions) {
    CppOptions cppOptions = buildOptions.get(CppOptions.class);
    if (stripBinaries) {
      boolean warn = cppOptions.coptList.contains("-g");
      for (PerLabelOptions opt : cppOptions.perFileCopts) {
        warn |= opt.getOptions().contains("-g");
      }
      if (warn) {
        reporter.handle(
            Event.warn(
                "Stripping enabled, but '--copt=-g' (or --per_file_copt=...@-g) specified. "
                    + "Debug information will be generated and then stripped away. This is "
                    + "probably not what you want! Use '-c dbg' for debug mode, or use "
                    + "'--strip=never' to disable stripping"));
      }
    }

    // FDO
    if (cppOptions.getFdoOptimize() != null && cppOptions.fdoProfileLabel != null) {
      reporter.handle(Event.error("Both --fdo_optimize and --fdo_profile specified"));
    }

    if (cppOptions.fdoInstrumentForBuild != null) {
      if (cppOptions.getFdoOptimize() != null || cppOptions.fdoProfileLabel != null) {
        reporter.handle(
            Event.error(
                "Cannot instrument and optimize for FDO at the same time. Remove one of the "
                    + "'--fdo_instrument' and '--fdo_optimize/--fdo_profile' options"));
      }
      if (!cppOptions.coptList.contains("-Wno-error")) {
        // This is effectively impossible. --fdo_instrument adds this value, and only invocation
        // policy could remove it.
        reporter.handle(Event.error("Cannot instrument FDO without --copt including -Wno-error."));
      }
    }

    // This is an assertion check vs. user error because users can't trigger this state.
    Verify.verify(
        !(buildOptions.get(BuildConfiguration.Options.class).isHost && cppOptions.isFdo()),
        "FDO state should not propagate to the host configuration");
  }

  @Override
  public void addGlobalMakeVariables(ImmutableMap.Builder<String, String> globalMakeEnvBuilder) {
    if (cppOptions.disableMakeVariables) {
      return;
    }

    if (!shouldProvideMakeVariables) {
      return;
    }
    globalMakeEnvBuilder.putAll(
        CcToolchainProvider.getCppBuildVariables(
            this::getToolPathFragment,
            cppToolchainInfo.getTargetLibc(),
            cppToolchainInfo.getCompiler(),
            desiredCpu,
            crosstoolTopPathFragment,
            cppToolchainInfo.getAbiGlibcVersion(),
            cppToolchainInfo.getAbi(),
            getAdditionalMakeVariables()));
  }

  @Override
  public String getOutputDirectoryName() {
    String toolchainPrefix = desiredCpu;
    if (!cppOptions.outputDirectoryTag.isEmpty()) {
      toolchainPrefix += "-" + cppOptions.outputDirectoryTag;
    }

    return toolchainPrefix;
  }

  /**
   * Returns true if we should share identical native libraries between different targets.
   */
  public boolean shareNativeDeps() {
    return cppOptions.shareNativeDeps;
  }

  public boolean isStrictSystemIncludes() {
    return cppOptions.strictSystemIncludes;
  }

  @Override
  public Map<String, Object> lateBoundOptionDefaults() {
    // --compiler initially defaults to null because its *actual* default isn't known
    // until it's read from the CROSSTOOL. Feed the CROSSTOOL defaults in here.
    return ImmutableMap.of("compiler", cppToolchainInfo.getCompiler());
  }

  public String getFdoInstrument() {
    return cppOptions.fdoInstrumentForBuild;
  }

  public PathFragment getFdoPath() {
    return fdoPath;
  }

  public Label getFdoOptimizeLabel() {
    return fdoOptimizeLabel;
  }

  public Label getFdoPrefetchHintsLabel() {
    return cppOptions.getFdoPrefetchHintsLabel();
  }

  public Label getFdoProfileLabel() {
    return cppOptions.fdoProfileLabel;
  }

  public boolean isFdoAbsolutePathEnabled() {
    return cppOptions.enableFdoProfileAbsolutePath;
  }

  public boolean useLLVMCoverageMapFormat() {
    return cppOptions.useLLVMCoverageMapFormat;
  }

  public boolean disableLegacyCrosstoolFields() {
    return cppOptions.disableLegacyCrosstoolFields;
  }

  public boolean disableCompilationModeFlags() {
    return cppOptions.disableCompilationModeFlags;
  }

  public boolean disableLinkingModeFlags() {
    return cppOptions.disableLinkingModeFlags;
  }

  public boolean disableMakeVariables() {
    return cppOptions.disableMakeVariables;
  }

  /**
   * cc_toolchain_suite allows to override CROSSTOOL by using proto attribute. This attribute value
   * is stored here so cc_toolchain can access it in the analysis. Don't use this for anything, it
   * will be removed when b/113849758 is fixed. If you do, I'll send bubo to take your keyboard
   * away.
   */
  @Deprecated
  public CrosstoolRelease getCrosstoolFromCcToolchainProtoAttribute() {
    return crosstoolFromCcToolchainProtoAttribute;
  }

  public boolean enableLinkoptsInUserLinkFlags() {
    return cppOptions.enableLinkoptsInUserLinkFlags;
  }

  public boolean disableEmittingStaticLibgcc() {
    return cppOptions.disableEmittingStaticLibgcc;
  }

  public boolean disableDepsetInUserFlags() {
    return cppOptions.disableDepsetInUserFlags;
  }

  private void checkForToolchainSkylarkApiAvailability() throws EvalException {
    if (cppOptions.disableLegacyToolchainSkylarkApi
        || !cppOptions.enableLegacyToolchainSkylarkApi) {
      throw new EvalException(
          null,
          "Information about the C++ toolchain API is not accessible "
              + "anymore through ctx.fragments.cpp "
              + "(see --incompatible_disable_legacy_cpp_toolchain_skylark_api on "
              + "http://docs.bazel.build/versions/master/skylark/backward-compatibility.html"
              + "#disable-legacy-c-configuration-api for migration notes). "
              + "Use CcToolchainInfo instead.");
    }
  }

  public void checkForLegacyCompilationApiAvailability() throws EvalException {
    if (cppOptions.disableLegacyCompilationApi || cppOptions.disableLegacyFlagsCcToolchainApi) {
      throw new EvalException(
          null,
          "Skylark APIs accessing compilation flags has been removed. "
              + "Use the new API on cc_common (see "
              + "--incompatible_disable_legacy_flags_cc_toolchain_api on"
              + "https://docs.bazel.build/versions/master/skylark/backward-compatibility.html"
              + "#disable-legacy-c-toolchain-api for migration notes).");
    }
  }

  public void checkForLegacyLinkingApiAvailability() throws EvalException {
    if (cppOptions.disableLegacyLinkingApi || cppOptions.disableLegacyFlagsCcToolchainApi) {
      throw new EvalException(
          null,
          "Skylark APIs accessing linking flags has been removed. "
              + "Use the new API on cc_common (see "
              + "--incompatible_disable_legacy_flags_cc_toolchain_api on"
              + "https://docs.bazel.build/versions/master/skylark/backward-compatibility.html"
              + "#disable-legacy-c-toolchain-api for migration notes).");
    }
  }

  public static PathFragment computeDefaultSysroot(String builtInSysroot) {
    if (builtInSysroot.isEmpty()) {
      return null;
    }
    if (!PathFragment.isNormalized(builtInSysroot)) {
      throw new IllegalArgumentException(
          "The built-in sysroot '" + builtInSysroot + "' is not normalized.");
    }
    return PathFragment.create(builtInSysroot);
  }

  boolean enableCcToolchainConfigInfoFromSkylark() {
    return cppOptions.enableCcToolchainConfigInfoFromSkylark;
  }

  /**
   * Returns the value of the libc top-level directory (--grte_top) as specified on the command line
   */
  public Label getLibcTopLabel() {
    return cppOptions.libcTopLabel;
  }
}
