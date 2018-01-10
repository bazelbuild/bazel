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
import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.base.Verify;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMap.Builder;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ListMultimap;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.Options.MakeVariableSource;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.CompilationMode;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.config.PerLabelOptions;
import com.google.devtools.build.lib.analysis.config.transitions.PatchTransition;
import com.google.devtools.build.lib.analysis.skylark.SkylarkConfigurationField;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.OutputFile;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.rules.cpp.CppConfigurationLoader.CppConfigurationParameters;
import com.google.devtools.build.lib.rules.cpp.CrosstoolConfigurationLoader.CrosstoolFile;
import com.google.devtools.build.lib.rules.cpp.transitions.ContextCollectorOwnerTransition;
import com.google.devtools.build.lib.rules.cpp.transitions.DisableLipoTransition;
import com.google.devtools.build.lib.skyframe.serialization.InjectingObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.vfs.FileSystemProvider;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.LipoMode;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * This class represents the C/C++ parts of the {@link BuildConfiguration}, including the host
 * architecture, target architecture, compiler version, and a standard library version. It has
 * information about the tools locations and the flags required for compiling.
 */
@AutoCodec(dependency = FileSystemProvider.class)
@SkylarkModule(
  name = "cpp",
  doc = "A configuration fragment for C++.",
  category = SkylarkModuleCategory.CONFIGURATION_FRAGMENT
)
@Immutable
public final class CppConfiguration extends BuildConfiguration.Fragment {
  public static final InjectingObjectCodec<CppConfiguration, FileSystemProvider> CODEC =
      new CppConfiguration_AutoCodec();

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
  public static enum HeadersCheckingMode {
    /**
     * Legacy behavior: Silently allow any source header file in any of the directories of the
     * containing package to be included by sources in this rule and dependent rules.
     */
    LOOSE,
    /** Warn about undeclared headers. */
    WARN,
    /** Disallow undeclared headers. */
    STRICT
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
   * This macro will be passed as a command-line parameter (eg. -DBUILD_FDO_TYPE="LIPO").
   * For possible values see {@code CppModel.getFdoBuildStamp()}.
   */
  public static final String FDO_STAMP_MACRO = "BUILD_FDO_TYPE";

  private final Label crosstoolTop;
  private final CrosstoolFile crosstoolFile;
  // TODO(lberki): desiredCpu *should* be always the same as targetCpu, except that we don't check
  // that the CPU we get from the toolchain matches BuildConfiguration.Options.cpu . So we store
  // it here so that the output directory doesn't depend on the CToolchain. When we will eventually
  // verify that the two are the same, we can remove one of desiredCpu and targetCpu.
  private final String desiredCpu;
  private final boolean convertLipoToThinLto;
  private final PathFragment crosstoolTopPathFragment;

  private final Path fdoZip;

  // TODO(bazel-team): All these labels (except for ccCompilerRuleLabel) can be removed once the
  // transition to the cc_compiler rule is complete.
  private final Label ccToolchainLabel;
  private final Label stlLabel;

  // TODO(kmensah): This is temporary until all the Skylark functions that need this can be removed.
  private final PathFragment nonConfiguredSysroot;
  private final Label sysrootLabel;

  private final FlagList compilerFlags;
  private final FlagList cxxFlags;
  private final FlagList unfilteredCompilerFlags;
  private final ImmutableList<String> cOptions;

  private final FlagList fullyStaticLinkFlags;
  private final FlagList mostlyStaticLinkFlags;
  private final FlagList mostlyStaticSharedLinkFlags;
  private final FlagList dynamicLinkFlags;
  private final ImmutableList<String> copts;
  private final ImmutableList<String> cxxopts;

  private final ImmutableList<String> linkOptions;
  private final ImmutableList<String> ltoindexOptions;

  private final CppOptions cppOptions;
  private final CpuTransformer cpuTransformerEnum;

  // The dynamic mode for linking.
  private final boolean stripBinaries;
  private final CompilationMode compilationMode;
  private final boolean useLLVMCoverageMap;

  private final boolean shouldProvideMakeVariables;

  /**
   *  If true, the ConfiguredTarget is only used to get the necessary cross-referenced
   *  CppCompilationContexts, but registering build actions is disabled.
   */
  private final boolean lipoContextCollector;

  private final CppToolchainInfo cppToolchainInfo;

  static CppConfiguration create(CppConfigurationParameters params)
      throws InvalidConfigurationException {
    CrosstoolConfig.CToolchain toolchain = params.toolchain;
    CppOptions cppOptions = params.cppOptions;
    PathFragment crosstoolTopPathFragment =
        params.crosstoolTop.getPackageIdentifier().getPathUnderExecRoot();
    CppToolchainInfo cppToolchainInfo =
        CppToolchainInfo.create(toolchain, crosstoolTopPathFragment, params.crosstoolTop);

    ListMultimap<CompilationMode, String> cFlags = ArrayListMultimap.create();
    ListMultimap<CompilationMode, String> cxxFlags = ArrayListMultimap.create();
    for (CrosstoolConfig.CompilationModeFlags flags : toolchain.getCompilationModeFlagsList()) {
      // Remove this when CROSSTOOL files no longer contain 'coverage'.
      if (flags.getMode() == CrosstoolConfig.CompilationMode.COVERAGE) {
        continue;
      }
      CompilationMode realmode = CppToolchainInfo.importCompilationMode(flags.getMode());
      cFlags.putAll(realmode, flags.getCompilerFlagList());
      cxxFlags.putAll(realmode, flags.getCxxFlagList());
    }

    CompilationMode compilationMode = params.commonOptions.compilationMode;

    ListMultimap<LipoMode, String> lipoCFlags = ArrayListMultimap.create();
    ListMultimap<LipoMode, String> lipoCxxFlags = ArrayListMultimap.create();
    for (CrosstoolConfig.LipoModeFlags flags : toolchain.getLipoModeFlagsList()) {
      LipoMode realmode = flags.getMode();
      lipoCFlags.putAll(realmode, flags.getCompilerFlagList());
      lipoCxxFlags.putAll(realmode, flags.getCxxFlagList());
    }

    ImmutableList.Builder<String> coptsBuilder =
        ImmutableList.<String>builder()
            .addAll(toolchain.getCompilerFlagList())
            .addAll(cFlags.get(compilationMode))
            .addAll(lipoCFlags.get(cppOptions.getLipoMode()));
    if (cppOptions.experimentalOmitfp) {
      coptsBuilder.add("-fomit-frame-pointer");
      coptsBuilder.add("-fasynchronous-unwind-tables");
      coptsBuilder.add("-DNO_FRAME_POINTER");
    }

    ImmutableList.Builder<String> cxxOptsBuilder =
        ImmutableList.<String>builder()
            .addAll(toolchain.getCxxFlagList())
            .addAll(cxxFlags.get(compilationMode))
            .addAll(lipoCxxFlags.get(cppOptions.getLipoMode()));

    ImmutableList.Builder<String> linkoptsBuilder = ImmutableList.builder();
    linkoptsBuilder.addAll(cppOptions.linkoptList);
    if (cppOptions.experimentalOmitfp) {
      linkoptsBuilder.add("-Wl,--eh-frame-hdr");
    }

    return new CppConfiguration(
        params.crosstoolTop,
        params.crosstoolFile,
        Preconditions.checkNotNull(params.commonOptions.cpu),
        cppOptions.convertLipoToThinLto,
        crosstoolTopPathFragment,
        params.fdoZip,
        params.ccToolchainLabel,
        params.stlLabel,
        params.sysrootLabel == null
            ? cppToolchainInfo.getDefaultSysroot()
            : params.sysrootLabel.getPackageFragment(),
        params.sysrootLabel,
        new FlagList(
            coptsBuilder.build(),
            FlagList.convertOptionalOptions(toolchain.getOptionalCompilerFlagList()),
            ImmutableList.copyOf(cppOptions.coptList)),
        new FlagList(
            cxxOptsBuilder.build(),
            FlagList.convertOptionalOptions(toolchain.getOptionalCxxFlagList()),
            ImmutableList.copyOf(cppOptions.cxxoptList)),
        new FlagList(
            ImmutableList.copyOf(toolchain.getUnfilteredCxxFlagList()),
            FlagList.convertOptionalOptions(toolchain.getOptionalUnfilteredCxxFlagList()),
            ImmutableList.<String>of()),
        ImmutableList.copyOf(cppOptions.conlyoptList),
        new FlagList(
            cppToolchainInfo.configureLinkerOptions(
                compilationMode,
                cppOptions.getLipoMode(),
                LinkingMode.FULLY_STATIC,
                cppToolchainInfo.getLdExecutable()),
            FlagList.convertOptionalOptions(toolchain.getOptionalLinkerFlagList()),
            ImmutableList.<String>of()),
        new FlagList(
            cppToolchainInfo.configureLinkerOptions(
                compilationMode,
                cppOptions.getLipoMode(),
                LinkingMode.MOSTLY_STATIC,
                cppToolchainInfo.getLdExecutable()),
            FlagList.convertOptionalOptions(toolchain.getOptionalLinkerFlagList()),
            ImmutableList.<String>of()),
        new FlagList(
            cppToolchainInfo.configureLinkerOptions(
                compilationMode,
                cppOptions.getLipoMode(),
                LinkingMode.MOSTLY_STATIC_LIBRARIES,
                cppToolchainInfo.getLdExecutable()),
            FlagList.convertOptionalOptions(toolchain.getOptionalLinkerFlagList()),
            ImmutableList.<String>of()),
        new FlagList(
            cppToolchainInfo.configureLinkerOptions(
                compilationMode,
                cppOptions.getLipoMode(),
                LinkingMode.DYNAMIC,
                cppToolchainInfo.getLdExecutable()),
            FlagList.convertOptionalOptions(toolchain.getOptionalLinkerFlagList()),
            ImmutableList.<String>of()),
        ImmutableList.copyOf(cppOptions.coptList),
        ImmutableList.copyOf(cppOptions.cxxoptList),
        linkoptsBuilder.build(),
        ImmutableList.copyOf(cppOptions.ltoindexoptList),
        cppOptions,
        params.cpuTransformer,
        (cppOptions.stripBinaries == StripMode.ALWAYS
            || (cppOptions.stripBinaries == StripMode.SOMETIMES
                && compilationMode == CompilationMode.FASTBUILD)),
        compilationMode,
        params.commonOptions.useLLVMCoverageMapFormat,
        params.commonOptions.makeVariableSource == MakeVariableSource.CONFIGURATION,
        cppOptions.isLipoContextCollector(),
        cppToolchainInfo);
  }

  @AutoCodec.Constructor
  CppConfiguration(
      Label crosstoolTop,
      CrosstoolFile crosstoolFile,
      String desiredCpu,
      boolean convertLipoToThinLto,
      PathFragment crosstoolTopPathFragment,
      Path fdoZip,
      Label ccToolchainLabel,
      Label stlLabel,
      PathFragment nonConfiguredSysroot,
      Label sysrootLabel,
      FlagList compilerFlags,
      FlagList cxxFlags,
      FlagList unfilteredCompilerFlags,
      ImmutableList<String> cOptions,
      FlagList fullyStaticLinkFlags,
      FlagList mostlyStaticLinkFlags,
      FlagList mostlyStaticSharedLinkFlags,
      FlagList dynamicLinkFlags,
      ImmutableList<String> copts,
      ImmutableList<String> cxxopts,
      ImmutableList<String> linkOptions,
      ImmutableList<String> ltoindexOptions,
      CppOptions cppOptions,
      CpuTransformer cpuTransformerEnum,
      boolean stripBinaries,
      CompilationMode compilationMode,
      boolean useLLVMCoverageMap,
      boolean shouldProvideMakeVariables,
      boolean lipoContextCollector,
      CppToolchainInfo cppToolchainInfo) {
    this.crosstoolTop = crosstoolTop;
    this.crosstoolFile = crosstoolFile;
    this.desiredCpu = desiredCpu;
    this.convertLipoToThinLto = convertLipoToThinLto;
    this.crosstoolTopPathFragment = crosstoolTopPathFragment;
    this.fdoZip = fdoZip;
    this.ccToolchainLabel = ccToolchainLabel;
    this.stlLabel = stlLabel;
    this.nonConfiguredSysroot = nonConfiguredSysroot;
    this.sysrootLabel = sysrootLabel;
    this.compilerFlags = compilerFlags;
    this.cxxFlags = cxxFlags;
    this.unfilteredCompilerFlags = unfilteredCompilerFlags;
    this.cOptions = cOptions;
    this.fullyStaticLinkFlags = fullyStaticLinkFlags;
    this.mostlyStaticLinkFlags = mostlyStaticLinkFlags;
    this.mostlyStaticSharedLinkFlags = mostlyStaticSharedLinkFlags;
    this.dynamicLinkFlags = dynamicLinkFlags;
    this.copts = copts;
    this.cxxopts = cxxopts;
    this.linkOptions = linkOptions;
    this.ltoindexOptions = ltoindexOptions;
    this.cppOptions = cppOptions;
    this.cpuTransformerEnum = cpuTransformerEnum;
    this.stripBinaries = stripBinaries;
    this.compilationMode = compilationMode;
    this.useLLVMCoverageMap = useLLVMCoverageMap;
    this.shouldProvideMakeVariables = shouldProvideMakeVariables;
    this.lipoContextCollector = lipoContextCollector;
    this.cppToolchainInfo = cppToolchainInfo;
  }

  @VisibleForTesting
  static LinkingMode importLinkingMode(CrosstoolConfig.LinkingMode mode) {
    return LinkingMode.valueOf(mode.name());
  }

  /** Returns the {@link CppToolchainInfo} used by this configuration. */
  public CppToolchainInfo getCppToolchainInfo() {
    return cppToolchainInfo;
  }

  /**
   * Returns the toolchain identifier, which uniquely identifies the compiler
   * version, target libc version, target cpu, and LIPO linkage.
   */
  public String getToolchainIdentifier() {
    return cppToolchainInfo.getToolchainIdentifier();
  }

  /** Returns the contents of the CROSSTOOL for this configuration. */
  public CrosstoolFile getCrosstoolFile() {
    return crosstoolFile;
  }

  /** Returns the label of the CROSSTOOL for this configuration. */
  public Label getCrosstoolTop() {
    return crosstoolTop;
  }

  /** Returns the transformer that should be applied to cpu names in toolchain selection. */
  public Function<String, String> getCpuTransformer() {
    return cpuTransformerEnum.getTransformer();
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
  @SkylarkCallable(name = "compiler", structField = true, doc = "C++ compiler.")
  @Deprecated
  public String getCompiler() {
    return cppToolchainInfo.getCompiler();
  }

  /**
   * Returns the libc version string (e.g. "glibc-2.2.2").
   *
   * <p>Deprecated: Use {@link CcToolchainProvider#getTargetLibc()}
   */
  // TODO(b/68038647): Remove once make variables are no longer derived from CppConfiguration.
  @SkylarkCallable(name = "libc", structField = true, doc = "libc version string.")
  @Deprecated
  public String getTargetLibc() {
    return cppToolchainInfo.getTargetLibc();
  }

  /**
   * Returns the target architecture using blaze-specific constants (e.g. "piii").
   *
   * <p>Deprecated: Use {@link CcToolchainProvider#getTargetCpu()}
   */
  // TODO(b/68038647): Remove once skylark callers are migrated.
  @SkylarkCallable(name = "cpu", structField = true, doc = "Target CPU of the C++ toolchain.")
  @Deprecated
  public String getTargetCpu() {
    return cppToolchainInfo.getTargetCpu();
  }

  /**
   * Unused, for compatibility with things internal to Google.
   *
   * <p>Deprecated: Use platforms.
   */
  // TODO(b/64384912): Remove once c++ platforms are in use.
  @Deprecated
  public String getTargetOS() {
    return cppToolchainInfo.getTargetOS();
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

  /**
   * Returns the label of the <code>cc_compiler</code> rule for the C++ configuration.
   */
  @SkylarkConfigurationField(
      name = "cc_toolchain",
      doc = "The label of the target describing the C++ toolchain",
      defaultLabel = "//tools/defaults:crosstool",
      defaultInToolRepository = false
  )
  public Label getCcToolchainRuleLabel() {
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

  @SkylarkCallable(
    name = "built_in_include_directories",
    structField = true,
    doc =
        "Built-in system include paths for the toolchain compiler. All paths in this list"
            + " should be relative to the exec directory. They may be absolute if they are also"
            + " installed on the remote build nodes or for local compilation."
  )
  public ImmutableList<String> getBuiltInIncludeDirectoriesForSkylark()
      throws InvalidConfigurationException {
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
          CcToolchain.resolveIncludeDir(s, sysroot, crosstoolTopPathFragment));
    }
    return builtInIncludeDirectoriesBuilder.build();
  }

  /**
   * Returns the sysroot to be used. If the toolchain compiler does not support
   * different sysroots, or the sysroot is the same as the default sysroot, then
   * this method returns <code>null</code>.
   */
  @SkylarkCallable(name = "sysroot", structField = true,
      doc = "Returns the sysroot to be used. If the toolchain compiler does not support "
      + "different sysroots, or the sysroot is the same as the default sysroot, then "
      + "this method returns <code>None</code>.")
  public String getSysroot() {
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
   * <p>Deprecated: Use {@link CppHelper#getCompilerOptions}
   */
  // TODO(b/64384912): Migrate skylark callers and remove.
  @SkylarkCallable(
    name = "compiler_options",
    doc =
        "Returns the default options to use for compiling C, C++, and assembler. "
            + "This is just the options that should be used for all three languages. "
            + "There may be additional C-specific or C++-specific options that should be used, "
            + "in addition to the ones returned by this method"
  )
  @Deprecated
  public ImmutableList<String> getCompilerOptions(Iterable<String> features) {
    return compilerFlags.evaluate(features);
  }

  /**
   * Returns the list of additional C-specific options to use for compiling
   * C. These should be go on the command line after the common options
   * returned by {@link #getCompilerOptions}.
   */
  @SkylarkCallable(name = "c_options", structField = true,
      doc = "Returns the list of additional C-specific options to use for compiling C. "
      + "These should be go on the command line after the common options returned by "
      + "<code>compiler_options</code>")
  public ImmutableList<String> getCOptions() {
    return cOptions;
  }

  /**
   * Returns the list of additional C++-specific options to use for compiling C++. These should be
   * on the command line after the common options returned by {@link #getCompilerOptions}.
   *
   * <p>Deprecated: Use {@link CppHelper#getCxxOptions}
   */
  // TODO(b/64384912): Migrate skylark callers and remove.
  @SkylarkCallable(
    name = "cxx_options",
    doc =
        "Returns the list of additional C++-specific options to use for compiling C++. "
            + "These should be go on the command line after the common options returned by "
            + "<code>compiler_options</code>"
  )
  @Deprecated
  public ImmutableList<String> getCxxOptions(Iterable<String> features) {
    return cxxFlags.evaluate(features);
  }

  /**
   * Returns the default list of options which cannot be filtered by BUILD rules. These should be
   * appended to the command line after filtering.
   *
   * @deprecated since it uses nonconfigured sysroot. Use
   * {@link CcToolchainProvider#getUnfilteredCompilerOptionsWithSysroot(Iterable)} if you *really*
   * need to.
   */
  // TODO(b/65401585): Migrate existing uses to cc_toolchain and cleanup here.
  @Deprecated
  @SkylarkCallable(
    name = "unfiltered_compiler_options",
    doc =
        "Returns the default list of options which cannot be filtered by BUILD "
            + "rules. These should be appended to the command line after filtering."
  )
  public ImmutableList<String> getUnfilteredCompilerOptionsWithLegacySysroot(
      Iterable<String> features) {
    return getUnfilteredCompilerOptionsDoNotUse(features, nonConfiguredSysroot);
  }

  /**
   * @deprecated since it hardcodes --sysroot flag. Use
   * {@link com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration}
   * instead.
   */
  // TODO(b/65401585): Migrate existing uses to cc_toolchain and cleanup here.
  @Deprecated
  ImmutableList<String> getUnfilteredCompilerOptionsDoNotUse(
      Iterable<String> features, @Nullable PathFragment sysroot) {
    if (sysroot == null) {
      return unfilteredCompilerFlags.evaluate(features);
    }
    return ImmutableList.<String>builder()
        .add("--sysroot=" + sysroot)
        .addAll(unfilteredCompilerFlags.evaluate(features))
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
  @SkylarkCallable(
    name = "link_options",
    structField = true,
    doc =
        "Returns the set of command-line linker options, including any flags "
            + "inferred from the command-line options."
  )
  public ImmutableList<String> getLinkOptionsWithLegacySysroot() {
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
      return linkOptions;
    } else {
      return ImmutableList.<String>builder()
          .addAll(linkOptions)
          .add("--sysroot=" + sysroot)
          .build();
    }
  }

  public boolean hasStaticLinkOption() {
    return linkOptions.contains("-static");
  }

  public boolean hasSharedLinkOption() {
    return linkOptions.contains("-shared");
  }

  /** Returns the set of command-line LTO indexing options. */
  public ImmutableList<String> getLtoIndexOptions() {
    return ltoindexOptions;
  }

  /**
   * Returns the immutable list of linker options for fully statically linked outputs. Does not
   * include command-line options passed via --linkopt or --linkopts.
   *
   * @param features default settings affecting this link
   * @param sharedLib true if the output is a shared lib, false if it's an executable
   *     <p>Deprecated: Use {@link CppHelper#getFullyStaticLinkOptions(CppConfiguration,
   *     CcToolchainProvider, Iterable, Boolean)}
   */
  // TODO(b/64384912): Migrate skylark users to cc_common and remove.
  @SkylarkCallable(
    name = "fully_static_link_options",
    doc =
        "Returns the immutable list of linker options for fully statically linked "
            + "outputs. Does not include command-line options passed via --linkopt or "
            + "--linkopts."
  )
  @Deprecated
  public ImmutableList<String> getFullyStaticLinkOptions(
      Iterable<String> features, Boolean sharedLib) {
    if (sharedLib) {
      return getSharedLibraryLinkOptions(mostlyStaticLinkFlags, features);
    } else {
      return fullyStaticLinkFlags.evaluate(features);
    }
  }

  /**
   * Returns the immutable list of linker options for mostly statically linked outputs. Does not
   * include command-line options passed via --linkopt or --linkopts.
   *
   * @param features default settings affecting this link
   * @param sharedLib true if the output is a shared lib, false if it's an executable
   *     <p>Deprecated: Use {@link CppHelper#getMostlyStaticLinkOptions(CppConfiguration,
   *     CcToolchainProvider, Iterable, Boolean)}
   */
  // TODO(b/64384912): Migrate skylark users to cc_common and remove.
  @SkylarkCallable(
    name = "mostly_static_link_options",
    doc =
        "Returns the immutable list of linker options for mostly statically linked "
            + "outputs. Does not include command-line options passed via --linkopt or "
            + "--linkopts."
  )
  @Deprecated
  public ImmutableList<String> getMostlyStaticLinkOptions(
      Iterable<String> features, Boolean sharedLib) {
    if (sharedLib) {
      return getSharedLibraryLinkOptions(
          cppToolchainInfo.supportsEmbeddedRuntimes()
              ? mostlyStaticSharedLinkFlags
              : dynamicLinkFlags,
          features);
    } else {
      return mostlyStaticLinkFlags.evaluate(features);
    }
  }

  /**
   * Returns the immutable list of linker options for artifacts that are not fully or mostly
   * statically linked. Does not include command-line options passed via --linkopt or --linkopts.
   *
   * @param features default settings affecting this link
   * @param sharedLib true if the output is a shared lib, false if it's an executable
   *     <p>Deprecated: Use {@link CppHelper#getDynamicLinkOptions(CppConfiguration,
   *     CcToolchainProvider, Iterable, Boolean)}
   */
  // TODO(b/64384912): Migrate skylark users to cc_common and remove.
  @SkylarkCallable(
    name = "dynamic_link_options",
    doc =
        "Returns the immutable list of linker options for artifacts that are not "
            + "fully or mostly statically linked. Does not include command-line options "
            + "passed via --linkopt or --linkopts."
  )
  @Deprecated
  public ImmutableList<String> getDynamicLinkOptions(Iterable<String> features, Boolean sharedLib) {
    if (sharedLib) {
      return getSharedLibraryLinkOptions(dynamicLinkFlags, features);
    } else {
      return dynamicLinkFlags.evaluate(features);
    }
  }

  /**
   * Returns link options for the specified flag list, combined with universal options for all
   * shared libraries (regardless of link staticness).
   *
   * <p>Deprecated: Use {@link CcToolchainProvider#getSharedLibraryLinkOptions}
   */
  // TODO(b/64384912): Migrate skylark dependants and delete.
  private ImmutableList<String> getSharedLibraryLinkOptions(
      FlagList flags, Iterable<String> features) {
    return cppToolchainInfo.getSharedLibraryLinkOptions(flags, features);
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
  @SkylarkCallable(name = "ld_executable", structField = true, doc = "Path to the linker binary.")
  public String getLdExecutableForSkylark() {
    PathFragment ldExecutable = getToolPathFragment(CppConfiguration.Tool.LD);
    return ldExecutable != null ? ldExecutable.getPathString() : "";
  }

  /** Returns the value of the --dynamic_mode flag. */
  public DynamicMode getDynamicModeFlag() {
    return cppOptions.dynamicMode;
  }

  public boolean getLinkCompileOutputSeparately() {
    return cppOptions.linkCompileOutputSeparately;
  }

  /*
   * If true then the directory name for non-LIPO targets will have a '-lipodata' suffix in
   * AutoFDO mode.
   */
  public boolean getAutoFdoLipoData() {
    return cppOptions.getAutoFdoLipoData();
  }

  /**
   * Returns the STL label if given on the command line. {@code null}
   * otherwise.
   */
  public Label getStl() {
    return stlLabel;
  }

  /**
   * Returns the currently active LIPO compilation mode.
   */
  public LipoMode getLipoMode() {
    return cppOptions.getLipoMode();
  }

  /** Returns true if lipo should be converted to thinlto. */
  public boolean shouldConvertLipoToThinLto() {
    return convertLipoToThinLto;
  }

  public boolean isFdo() {
    return cppOptions.isFdo();
  }

  /** Deprecated: Use {@link CcToolchainProvider#isLLVMCompiler()} */
  // TODO(b/64384912): Remove in favor of CcToolchainProvider#isLLVMCompiler
  @Deprecated
  private final boolean isLLVMCompiler() {
    return cppToolchainInfo.isLLVMCompiler();
  }

  /**
   * Returns true if LLVM FDO Optimization should be applied for this configuration.
   *
   * <p>Deprecated: Use {@link CppConfiguration#isLLVMOptimizedFdo(boolean)}
   */
  // TODO(b/64384912): Remove in favor of overload with isLLVMCompiler.
  @Deprecated
  public boolean isLLVMOptimizedFdo() {
    return cppOptions.getFdoOptimize() != null
        && (CppFileTypes.LLVM_PROFILE.matches(cppOptions.getFdoOptimize())
            || CppFileTypes.LLVM_PROFILE_RAW.matches(cppOptions.getFdoOptimize())
            || (isLLVMCompiler()
                && cppOptions.getFdoOptimize().endsWith(".zip")));
  }

  /** Returns true if LLVM FDO Optimization should be applied for this configuration. */
  public boolean isLLVMOptimizedFdo(boolean isLLVMCompiler) {
    return cppOptions.getFdoOptimize() != null
        && (CppFileTypes.LLVM_PROFILE.matches(cppOptions.getFdoOptimize())
            || CppFileTypes.LLVM_PROFILE_RAW.matches(cppOptions.getFdoOptimize())
            || (isLLVMCompiler && cppOptions.getFdoOptimize().endsWith(".zip")));
  }

  /** Returns true if LIPO optimization is implied by the flags of this build. */
  public boolean lipoOptimizationIsActivated() {
    return cppOptions.isLipoOptimization();
  }

  /**
   * Returns true if LIPO optimization should be applied for this configuration.
   *
   * <p>Deprecated: Use {@link CppHelper#isLipoOptimization(CppConfiguration, CcToolchainProvider)}
   */
  // TODO(b/64384912): Remove usage in topLevelConfigurationHook and CppRuleClasses and delete.
  @Deprecated
  public boolean isLipoOptimization() {
    // The LIPO optimization bits are set in the LIPO context collector configuration, too.
    // If compiler is LLVM, then LIPO gets auto-converted to ThinLTO.
    return cppOptions.isLipoOptimization() && !isLLVMCompiler();
  }

  public boolean isLipoOptimizationOrInstrumentation() {
    return cppOptions.isLipoOptimizationOrInstrumentation();
  }

  /**
   * Returns true if it is AutoFDO LIPO build.
   */
  public boolean isAutoFdoLipo() {
    return cppOptions.getFdoOptimize() != null
        && CppFileTypes.GCC_AUTO_PROFILE.matches(cppOptions.getFdoOptimize())
        && getLipoMode() != LipoMode.OFF;
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
   * Returns the LIPO context for this configuration.
   *
   * <p>This only exists for configurations that apply LIPO in LIPO-optimized builds. It does
   * <b>not</b> exist for data configurations, which contain LIPO state but don't actually apply
   * LIPO. Nor does it exist for host configurations, which contain no LIPO state.
   */
  public Label getLipoContextLabel() {
    return cppOptions.getLipoContext();
  }

  /**
   * Returns the LIPO context for this build, even if LIPO isn't enabled in the current
   * configuration.
   *
   * <p>Unlike {@link #getLipoContextLabel}, this returns the LIPO context for the data
   * configuration.
   *
   * <p>Unless you have a clear reason to use this version (which basically involves
   * inspecting oher configurations' state), always use {@link #getLipoContextLabel}.
   */
  public Label getLipoContextForBuild() {
    return cppOptions.getLipoContextForBuild();
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

  public boolean forceIgnoreDashStatic() {
    return cppOptions.forceIgnoreDashStatic;
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
  @SkylarkCallable(
    name = "objcopy_executable",
    structField = true,
    doc = "Path to GNU binutils 'objcopy' binary."
  )
  public String getObjCopyExecutableForSkylark() {
    PathFragment objCopyExecutable = getToolPathFragment(Tool.OBJCOPY);
    return objCopyExecutable != null ? objCopyExecutable.getPathString() : "";
  }

  @SkylarkCallable(
    name = "compiler_executable",
    structField = true,
    doc = "Path to C/C++ compiler binary."
  )
  public String getCppExecutableForSkylark() {
    PathFragment cppExecutable = getToolPathFragment(Tool.GCC);
    return cppExecutable != null ? cppExecutable.getPathString() : "";
  }

  @SkylarkCallable(
    name = "preprocessor_executable",
    structField = true,
    doc = "Path to C/C++ preprocessor binary."
  )
  public String getCpreprocessorExecutableForSkylark() {
    PathFragment cpreprocessorExecutable = getToolPathFragment(Tool.CPP);
    return cpreprocessorExecutable != null ? cpreprocessorExecutable.getPathString() : "";
  }

  /**
   * Returns the path to the 'gcov-tool' executable that should be used
   * by this build. Relative paths are relative to the execution root.
   */
  public PathFragment getGcovToolExecutable() {
    return getToolPathFragment(CppConfiguration.Tool.GCOVTOOL);
  }

  @SkylarkCallable(
    name = "nm_executable",
    structField = true,
    doc = "Path to GNU binutils 'nm' binary."
  )
  public String getNmExecutableForSkylark() {
    PathFragment nmExecutable = getToolPathFragment(Tool.NM);
    return nmExecutable != null ? nmExecutable.getPathString() : "";
  }

  @SkylarkCallable(
    name = "objdump_executable",
    structField = true,
    doc = "Path to GNU binutils 'objdump' binary."
  )
  public String getObjdumpExecutableForSkylark() {
    PathFragment objdumpExecutable = getToolPathFragment(Tool.OBJDUMP);
    return objdumpExecutable != null ? objdumpExecutable.getPathString() : "";
  }

  @SkylarkCallable(
    name = "ar_executable",
    structField = true,
    doc = "Path to GNU binutils 'ar' binary."
  )
  public String getArExecutableForSkylark() {
    PathFragment arExecutable = getToolPathFragment(Tool.AR);
    return arExecutable != null ? arExecutable.getPathString() : "";
  }

  @SkylarkCallable(
    name = "strip_executable",
    structField = true,
    doc = "Path to GNU binutils 'strip' binary."
  )
  public String getStripExecutableForSkylark() {
    PathFragment stripExecutable = getToolPathFragment(Tool.STRIP);
    return stripExecutable != null ? stripExecutable.getPathString() : "";
  }

  /**
   * Returns the GNU System Name
   *
   */
  //TODO(b/70225490): Migrate skylark dependants to CcToolchainProvider and delete.
  @SkylarkCallable(name = "target_gnu_system_name", structField = true,
      doc = "The GNU System Name.")
  @Deprecated
  public String getTargetGnuSystemName() {
    return cppToolchainInfo.getTargetGnuSystemName();
  }

  /**
   * Returns whether the configuration's purpose is only to collect LIPO-related data.
   */
  public boolean isLipoContextCollector() {
    return lipoContextCollector;
  }

  /** Returns whether this configuration will use libunwind for stack unwinding. */
  public boolean isOmitfp() {
    return cppOptions.experimentalOmitfp;
  }

  /** Returns copts given at the Bazel command line. */
  public ImmutableList<String> getCopts() {
    return copts;
  }

  /** Returns copts for c++ given at the Bazel command line. */
  public ImmutableList<String> getCxxopts() {
    return cxxopts;
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

    if (cppOptions.getFdoInstrument() != null) {
      if (cppOptions.getFdoOptimize() != null) {
        reporter.handle(
            Event.error(
                "Cannot instrument and optimize for FDO at the same time. "
                    + "Remove one of the '--fdo_instrument' and '--fdo_optimize' options"));
      }
      if (!cppOptions.coptList.contains("-Wno-error")) {
        // This is effectively impossible. --fdo_instrument adds this value, and only invocation
        // policy could remove it.
        reporter.handle(Event.error("Cannot instrument FDO without --copt including -Wno-error."));
      }
    }

    if (cppOptions.getLipoMode() != LipoMode.OFF
        && isLLVMCompiler()
        && !cppOptions.convertLipoToThinLto) {
      reporter.handle(
          Event.error(
              "The LLVM compiler does not support LIPO. Use --convert_lipo_to_thinlto to "
                  + "automatically fall back to thinlto."));
    }
    if (cppOptions.lipoContextForBuild != null) {
      if (!cppOptions.linkoptList.contains("-Wl,--warn-unresolved-symbols")) {
        // This is effectively impossible. --lipo_context adds these values, and only invocation
        // policy could remove them.
        reporter.handle(
            Event.error(
                "The --lipo_context option cannot be used without -Wl,--warn-unresolved-symbols "
                    + "included as a linkoption"));
      }
      if (isLLVMCompiler()) {
        reporter.handle(
            Event.warn("LIPO options are not applicable with a LLVM compiler and will be "
                + "converted to ThinLTO"));
      } else if (cppOptions.getLipoMode() != LipoMode.BINARY
          || cppOptions.getFdoOptimize() == null) {
        reporter.handle(
            Event.warn(
                "The --lipo_context option can only be used together with --fdo_optimize="
                    + "<profile zip> and --lipo=binary. LIPO context will be ignored."));
      }
    } else {
      if (!isLLVMCompiler()
          && cppOptions.getLipoMode() == LipoMode.BINARY
          && cppOptions.getFdoOptimize() != null) {
        reporter.handle(
            Event.error(
                "The --lipo_context option must be specified when using "
                    + "--fdo_optimize=<profile zip> and --lipo=binary"));
      }
    }
    if (cppOptions.getLipoMode() == LipoMode.BINARY && compilationMode != CompilationMode.OPT) {
      reporter.handle(Event.error(
          "'--lipo=binary' can only be used with '--compilation_mode=opt' (or '-c opt')"));
    }

    // This is an assertion check vs. user error because users can't trigger this state.
    Verify.verify(
        !(buildOptions.get(BuildConfiguration.Options.class).isHost && cppOptions.isFdo()),
        "FDO/LIPO state should not propagate to the host configuration");
  }

  @Override
  public void addGlobalMakeVariables(Builder<String, String> globalMakeEnvBuilder) {
    if (!cppOptions.enableMakeVariables) {
      return;
    }

    if (!shouldProvideMakeVariables) {
      return;
    }
    globalMakeEnvBuilder.putAll(
        CcToolchainProvider.getCppBuildVariables(
            this::getToolPathFragment,
            getTargetLibc(),
            getCompiler(),
            desiredCpu,
            crosstoolTopPathFragment,
            cppToolchainInfo.getAbiGlibcVersion(),
            cppToolchainInfo.getAbi(),
            getAdditionalMakeVariables()));
  }

  @Override
  public String getOutputDirectoryName() {
    String lipoSuffix;
    if (getLipoMode() != LipoMode.OFF && !isAutoFdoLipo()) {
      lipoSuffix = "-lipo";
    } else if (getAutoFdoLipoData()) {
      lipoSuffix = "-lipodata";
    } else {
      lipoSuffix = "";
    }
    String toolchainPrefix = desiredCpu;
    if (!cppOptions.outputDirectoryTag.isEmpty()) {
      toolchainPrefix += "-" + cppOptions.outputDirectoryTag;
    }

    return toolchainPrefix + lipoSuffix;
  }

  @Override
  public String getPlatformName() {
    return getToolchainIdentifier();
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
    return ImmutableMap.<String, Object>of(
        "compiler", getCompiler());
  }

  public PathFragment getFdoInstrument() {
    return cppOptions.getFdoInstrument();
  }

  public Path getFdoZip() {
    return fdoZip;
  }

  /**
   * Return set of features enabled by the CppConfiguration, specifically the FDO and LIPO related
   * features enabled by options.
   */
  @Override
  public ImmutableSet<String> configurationEnabledFeatures(
      RuleContext ruleContext, ImmutableSet<String> disabledFeatures) {
    ImmutableSet.Builder<String> requestedFeatures = ImmutableSet.builder();
    if (cppOptions.getFdoInstrument() != null) {
      requestedFeatures.add(CppRuleClasses.FDO_INSTRUMENT);
    }

    boolean isFdo = fdoZip != null && compilationMode == CompilationMode.OPT;
    if (isFdo && !CppFileTypes.GCC_AUTO_PROFILE.matches(fdoZip)) {
      requestedFeatures.add(CppRuleClasses.FDO_OPTIMIZE);
    }
    if (isFdo && CppFileTypes.GCC_AUTO_PROFILE.matches(fdoZip)) {
      requestedFeatures.add(CppRuleClasses.AUTOFDO);
      // For LLVM, support implicit enabling of ThinLTO for AFDO unless it has been
      // explicitly disabled.
      if (isLLVMCompiler() && !disabledFeatures.contains(CppRuleClasses.THIN_LTO)) {
        requestedFeatures.add(CppRuleClasses.ENABLE_AFDO_THINLTO);
      }
    }
    if (isLipoOptimizationOrInstrumentation()) {
      // Map LIPO to ThinLTO for LLVM builds.
      if (isLLVMCompiler() && cppOptions.getFdoOptimize() != null) {
        requestedFeatures.add(CppRuleClasses.THIN_LTO);
      } else {
        requestedFeatures.add(CppRuleClasses.LIPO);
      }
    }
    if (ruleContext.getConfiguration().isCodeCoverageEnabled()) {
      requestedFeatures.add(CppRuleClasses.COVERAGE);
      if (useLLVMCoverageMap) {
        requestedFeatures.add(CppRuleClasses.LLVM_COVERAGE_MAP_FORMAT);
      } else {
        requestedFeatures.add(CppRuleClasses.GCC_COVERAGE_MAP_FORMAT);
      }
    }
    return requestedFeatures.build();
  }

  public static PathFragment computeDefaultSysroot(CToolchain toolchain) {
    PathFragment defaultSysroot =
        toolchain.getBuiltinSysroot().length() == 0
            ? null
            : PathFragment.create(toolchain.getBuiltinSysroot());
    if ((defaultSysroot != null) && !defaultSysroot.isNormalized()) {
      throw new IllegalArgumentException(
          "The built-in sysroot '" + defaultSysroot + "' is not normalized.");
    }
    return defaultSysroot;
  }

  @Override
  public PatchTransition getArtifactOwnerTransition() {
    return isLipoContextCollector() ? ContextCollectorOwnerTransition.INSTANCE : null;
  }

  @Nullable
  @Override
  public PatchTransition topLevelConfigurationHook(Target toTarget) {
    // Top-level output files that aren't outputs of the LIPO context should be built in
    // the data config. This is so their output path prefix doesn't have "-lipo" in it, which
    // is a confusing and unnecessary deviation from how they would normally look.
    if (toTarget instanceof OutputFile
        && isLipoOptimization()
        && !toTarget.getAssociatedRule().getLabel().equals(getLipoContextLabel())) {
      return DisableLipoTransition.INSTANCE;
    } else {
      return null;
    }
  }
}
