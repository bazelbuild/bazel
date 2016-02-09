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
import com.google.common.base.Predicate;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMap.Builder;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.ListMultimap;
import com.google.common.collect.Maps;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactFactory;
import com.google.devtools.build.lib.actions.PackageRootResolutionException;
import com.google.devtools.build.lib.actions.PackageRootResolver;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.ViewCreationFailedException;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.CompilationMode;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.config.PerLabelOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.rules.cpp.CppConfigurationLoader.CppConfigurationParameters;
import com.google.devtools.build.lib.rules.cpp.FdoSupport.FdoException;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.LinkingModeFlags;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.LipoMode;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.protobuf.TextFormat;
import com.google.protobuf.TextFormat.ParseException;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.zip.ZipException;

import javax.annotation.Nullable;

/**
 * This class represents the C/C++ parts of the {@link BuildConfiguration},
 * including the host architecture, target architecture, compiler version, and
 * a standard library version. It has information about the tools locations and
 * the flags required for compiling.
 */
@SkylarkModule(name = "cpp", doc = "A configuration fragment for C++")
@Immutable
public class CppConfiguration extends BuildConfiguration.Fragment {

  /**
   * String indicating a Mac system, for example when used in a crosstool configuration's host or
   * target system name.
   */
  public static final String MAC_SYSTEM_NAME = "x86_64-apple-macosx";

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
    DWP("dwp");

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
  public static enum StripMode {

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

  /** Storage for the libc label, if given. */
  public static class LibcTop implements Serializable {
    private final Label label;

    LibcTop(Label label) {
      Preconditions.checkArgument(label != null);
      this.label = label;
    }

    public Label getLabel() {
      return label;
    }

    public PathFragment getSysroot() {
      return label.getPackageFragment();
    }

    @Override
    public String toString() {
      return label.toString();
    }

    @Override
    public boolean equals(Object other) {
      if (this == other) {
        return true;
      } else if (other instanceof LibcTop) {
        return label.equals(((LibcTop) other).label);
      } else {
        return false;
      }
    }

    @Override
    public int hashCode() {
      return label.hashCode();
    }
  }

  /**
   * This macro will be passed as a command-line parameter (eg. -DBUILD_FDO_TYPE="LIPO").
   * For possible values see {@code CppModel.getFdoBuildStamp()}.
   */
  public static final String FDO_STAMP_MACRO = "BUILD_FDO_TYPE";

  /**
   * This file (found under the sysroot) may be unconditionally included in every C/C++ compilation.
   */
  private static final PathFragment BUILT_IN_INCLUDE_PATH_FRAGMENT =
      new PathFragment("include/stdc-predef.h");

  /**
   * Represents an optional flag that can be toggled using the package features mechanism.
   */
  @Immutable
  @VisibleForTesting
  static class OptionalFlag implements Serializable {
    private final String name;
    private final ImmutableList<String> flags;

    @VisibleForTesting
    OptionalFlag(String name, ImmutableList<String> flags) {
      this.name = name;
      this.flags = flags;
    }

    private ImmutableList<String> getFlags() {
      return flags;
    }

    private String getName() {
      return name;
    }
  }

  @Immutable
  @VisibleForTesting
  static class FlagList implements Serializable {
    private final ImmutableList<String> prefixFlags;
    private final ImmutableList<OptionalFlag> optionalFlags;
    private final ImmutableList<String> suffixFlags;

    @VisibleForTesting
    FlagList(ImmutableList<String> prefixFlags,
        ImmutableList<OptionalFlag> optionalFlags,
        ImmutableList<String> suffixFlags) {
      this.prefixFlags = prefixFlags;
      this.optionalFlags = optionalFlags;
      this.suffixFlags = suffixFlags;
    }

    @VisibleForTesting
    ImmutableList<String> evaluate(Iterable<String> features) {
      ImmutableSet<String> featureSet = ImmutableSet.copyOf(features);
      ImmutableList.Builder<String> result = ImmutableList.builder();
      result.addAll(prefixFlags);
      for (OptionalFlag optionalFlag : optionalFlags) {
        // The flag is added if the default is true and the flag is not specified,
        // or if the default is false and the flag is specified.
        if (featureSet.contains(optionalFlag.getName())) {
          result.addAll(optionalFlag.getFlags());
        }
      }

      result.addAll(suffixFlags);
      return result.build();
    }
  }

  private final Label crosstoolTop;
  private final String hostSystemName;
  private final String compiler;
  private final String targetCpu;
  private final String targetSystemName;
  private final String targetLibc;
  private final LipoMode lipoMode;
  private final PathFragment crosstoolTopPathFragment;

  private final String abi;
  private final String abiGlibcVersion;

  private final String toolchainIdentifier;

  private final CcToolchainFeatures toolchainFeatures;
  private final boolean supportsGoldLinker;
  private final boolean supportsThinArchives;
  private final boolean supportsStartEndLib;
  private final boolean supportsDynamicLinker;
  private final boolean supportsInterfaceSharedObjects;
  private final boolean supportsEmbeddedRuntimes;
  private final boolean supportsFission;

  // We encode three states with two booleans:
  // (1) (false false) -> no pic code
  // (2) (true false)  -> shared libraries as pic, but not binaries
  // (3) (true true)   -> both shared libraries and binaries as pic
  private final boolean toolchainNeedsPic;
  private final boolean usePicForBinaries;

  private final FdoSupport fdoSupport;

  // TODO(bazel-team): All these labels (except for ccCompilerRuleLabel) can be removed once the
  // transition to the cc_compiler rule is complete.
  private final Label libcLabel;
  private final Label staticRuntimeLibsLabel;
  private final Label dynamicRuntimeLibsLabel;
  private final Label ccToolchainLabel;

  private final PathFragment sysroot;
  private final PathFragment runtimeSysroot;
  private final List<PathFragment> builtInIncludeDirectories;
  private Artifact builtInIncludeFile;

  private final Map<String, PathFragment> toolPaths;
  private final PathFragment ldExecutable;

  // Only used during construction.
  private final ImmutableList<String> commonLinkOptions;
  private final ListMultimap<CompilationMode, String> linkOptionsFromCompilationMode;
  private final ListMultimap<LipoMode, String> linkOptionsFromLipoMode;
  private final ListMultimap<LinkingMode, String> linkOptionsFromLinkingMode;

  private final FlagList compilerFlags;
  private final FlagList cxxFlags;
  private final FlagList unfilteredCompilerFlags;
  private final ImmutableList<String> cOptions;

  private final FlagList fullyStaticLinkFlags;
  private final FlagList mostlyStaticLinkFlags;
  private final FlagList mostlyStaticSharedLinkFlags;
  private final FlagList dynamicLinkFlags;
  private final FlagList dynamicLibraryLinkFlags;
  private final ImmutableList<String> testOnlyLinkFlags;

  private final ImmutableList<String> linkOptions;

  private final ImmutableList<String> objcopyOptions;
  private final ImmutableList<String> ldOptions;
  private final ImmutableList<String> arOptions;
  private final ImmutableList<String> arThinArchivesOptions;

  private final ImmutableMap<String, String> additionalMakeVariables;

  private final CppOptions cppOptions;

  // The dynamic mode for linking.
  private final DynamicMode dynamicMode;
  private final boolean stripBinaries;
  private final ImmutableMap<String, String> commandLineDefines;
  private final String solibDirectory;
  private final CompilationMode compilationMode;
  private final Path execRoot;
  /**
   *  If true, the ConfiguredTarget is only used to get the necessary cross-referenced
   *  CppCompilationContexts, but registering build actions is disabled.
   */
  private final boolean lipoContextCollector;

  protected CppConfiguration(CppConfigurationParameters params)
      throws InvalidConfigurationException {
    CrosstoolConfig.CToolchain toolchain = params.toolchain;
    cppOptions = params.cppOptions;
    this.hostSystemName = toolchain.getHostSystemName();
    this.compiler = toolchain.getCompiler();
    this.targetCpu = toolchain.getTargetCpu();
    this.lipoMode = cppOptions.getLipoMode();
    this.targetSystemName = toolchain.getTargetSystemName();
    this.targetLibc = toolchain.getTargetLibc();
    this.crosstoolTop = params.crosstoolTop;
    this.ccToolchainLabel = params.ccToolchainLabel;
    this.compilationMode = params.commonOptions.compilationMode;
    this.lipoContextCollector = cppOptions.lipoCollector;
    this.execRoot = params.execRoot;

    this.crosstoolTopPathFragment = crosstoolTop.getPackageIdentifier().getPathFragment();

    try {
      this.staticRuntimeLibsLabel =
          crosstoolTop.getRelative(
              toolchain.hasStaticRuntimesFilegroup()
                  ? toolchain.getStaticRuntimesFilegroup()
                  : "static-runtime-libs-" + targetCpu);
      this.dynamicRuntimeLibsLabel =
          crosstoolTop.getRelative(
              toolchain.hasDynamicRuntimesFilegroup()
                  ? toolchain.getDynamicRuntimesFilegroup()
                  : "dynamic-runtime-libs-" + targetCpu);
    } catch (LabelSyntaxException e) {
      // All of the above label.getRelative() calls are valid labels, and the crosstool_top
      // was already checked earlier in the process.
      throw new AssertionError(e);
    }

    if (cppOptions.lipoMode == LipoMode.BINARY) {
      // TODO(bazel-team): implement dynamic linking with LIPO
      this.dynamicMode = DynamicMode.OFF;
    } else {
      switch (cppOptions.dynamicMode) {
        case DEFAULT:
          this.dynamicMode = DynamicMode.DEFAULT; break;
        case OFF: this.dynamicMode = DynamicMode.OFF; break;
        case FULLY: this.dynamicMode = DynamicMode.FULLY; break;
        default: throw new IllegalStateException("Invalid dynamicMode.");
      }
    }

    this.fdoSupport = new FdoSupport(
        cppOptions.fdoInstrument, params.fdoZip,
        cppOptions.lipoMode, execRoot);

    this.stripBinaries =
        (cppOptions.stripBinaries == StripMode.ALWAYS
            || (cppOptions.stripBinaries == StripMode.SOMETIMES
                && compilationMode == CompilationMode.FASTBUILD));

    CrosstoolConfigurationIdentifier crosstoolConfig =
        CrosstoolConfigurationIdentifier.fromToolchain(toolchain);
    Preconditions.checkState(crosstoolConfig.getCpu().equals(targetCpu));
    Preconditions.checkState(crosstoolConfig.getCompiler().equals(compiler));
    Preconditions.checkState(crosstoolConfig.getLibc().equals(targetLibc));

    this.solibDirectory = "_solib_" + targetCpu;

    this.toolchainIdentifier = toolchain.getToolchainIdentifier();

    toolchain = addLegacyFeatures(toolchain);
    this.toolchainFeatures = new CcToolchainFeatures(toolchain);
    this.supportsGoldLinker = toolchain.getSupportsGoldLinker();
    this.supportsThinArchives = toolchain.getSupportsThinArchives();
    this.supportsStartEndLib = toolchain.getSupportsStartEndLib();
    this.supportsInterfaceSharedObjects = toolchain.getSupportsInterfaceSharedObjects();
    this.supportsEmbeddedRuntimes = toolchain.getSupportsEmbeddedRuntimes();
    this.supportsFission = toolchain.getSupportsFission();
    this.toolchainNeedsPic = toolchain.getNeedsPic();
    this.usePicForBinaries =
        toolchain.getNeedsPic() && compilationMode != CompilationMode.OPT;

    this.toolPaths = Maps.newHashMap();
    for (CrosstoolConfig.ToolPath tool : toolchain.getToolPathList()) {
      PathFragment path = new PathFragment(tool.getPath());
      if (!path.isNormalized()) {
        throw new IllegalArgumentException("The include path '" + tool.getPath()
            + "' is not normalized.");
      }
      toolPaths.put(tool.getName(), crosstoolTopPathFragment.getRelative(path));
    }

    if (toolPaths.isEmpty()) {
      // If no paths are specified, we just use the names of the tools as the path.
      for (Tool tool : Tool.values()) {
        toolPaths.put(tool.getNamePart(),
            crosstoolTopPathFragment.getRelative(tool.getNamePart()));
      }
    } else {
      Iterable<Tool> neededTools = Iterables.filter(EnumSet.allOf(Tool.class),
          new Predicate<Tool>() {
            @Override
            public boolean apply(Tool tool) {
              if (tool == Tool.DWP) {
                // When fission is unsupported, don't check for the dwp tool.
                return supportsFission();
              } else if (tool == Tool.GCOVTOOL) {
                // gcov-tool is optional, don't check whether it's present
                return false;
              } else {
                return true;
              }
            }
          });
      for (Tool tool : neededTools) {
        if (!toolPaths.containsKey(tool.getNamePart())) {
          throw new IllegalArgumentException("Tool path for '" + tool.getNamePart()
              + "' is missing");
        }
      }
    }

    // We can't use an ImmutableMap.Builder here; we need the ability (at least
    // in tests) to add entries with keys that are already in the map, and only
    // HashMap supports this (by replacing the existing entry under the key).
    Map<String, String> commandLineDefinesBuilder = new HashMap<>();
    for (Map.Entry<String, String> define : cppOptions.commandLineDefinedVariables) {
      commandLineDefinesBuilder.put(define.getKey(), define.getValue());
    }
    commandLineDefines = ImmutableMap.copyOf(commandLineDefinesBuilder);

    ListMultimap<CompilationMode, String> cFlags = ArrayListMultimap.create();
    ListMultimap<CompilationMode, String> cxxFlags = ArrayListMultimap.create();
    linkOptionsFromCompilationMode = ArrayListMultimap.create();
    for (CrosstoolConfig.CompilationModeFlags flags : toolchain.getCompilationModeFlagsList()) {
      // Remove this when CROSSTOOL files no longer contain 'coverage'.
      if (flags.getMode() == CrosstoolConfig.CompilationMode.COVERAGE) {
        continue;
      }
      CompilationMode realmode = importCompilationMode(flags.getMode());
      cFlags.putAll(realmode, flags.getCompilerFlagList());
      cxxFlags.putAll(realmode, flags.getCxxFlagList());
      linkOptionsFromCompilationMode.putAll(realmode, flags.getLinkerFlagList());
    }

    ListMultimap<LipoMode, String> lipoCFlags = ArrayListMultimap.create();
    ListMultimap<LipoMode, String> lipoCxxFlags = ArrayListMultimap.create();
    linkOptionsFromLipoMode = ArrayListMultimap.create();
    for (CrosstoolConfig.LipoModeFlags flags : toolchain.getLipoModeFlagsList()) {
      LipoMode realmode = flags.getMode();
      lipoCFlags.putAll(realmode, flags.getCompilerFlagList());
      lipoCxxFlags.putAll(realmode, flags.getCxxFlagList());
      linkOptionsFromLipoMode.putAll(realmode, flags.getLinkerFlagList());
    }

    linkOptionsFromLinkingMode = ArrayListMultimap.create();

    // If a toolchain supports dynamic libraries at all, there must be at least one
    // of the following:
    // - a "DYNAMIC" section in linking_mode_flags (even if no flags are needed)
    // - a non-empty list in one of the dynamicLibraryLinkerFlag fields
    // If none of the above contain data, then the toolchain can't do dynamic linking.
    boolean haveDynamicMode = false;

    for (LinkingModeFlags flags : toolchain.getLinkingModeFlagsList()) {
      LinkingMode realmode = importLinkingMode(flags.getMode());
      if (realmode == LinkingMode.DYNAMIC) {
        haveDynamicMode = true;
      }
      linkOptionsFromLinkingMode.putAll(realmode, flags.getLinkerFlagList());
    }

    this.commonLinkOptions = ImmutableList.copyOf(toolchain.getLinkerFlagList());
    List<String> linkerFlagList = toolchain.getDynamicLibraryLinkerFlagList();
    List<CToolchain.OptionalFlag> optionalLinkerFlagList =
        toolchain.getOptionalDynamicLibraryLinkerFlagList();
    if (!linkerFlagList.isEmpty() || !optionalLinkerFlagList.isEmpty()) {
      haveDynamicMode = true;
    }
    this.supportsDynamicLinker = haveDynamicMode;
    dynamicLibraryLinkFlags = new FlagList(
        ImmutableList.copyOf(linkerFlagList),
        convertOptionalOptions(optionalLinkerFlagList),
        ImmutableList.<String>of());

    this.objcopyOptions = ImmutableList.copyOf(toolchain.getObjcopyEmbedFlagList());
    this.ldOptions = ImmutableList.copyOf(toolchain.getLdEmbedFlagList());
    this.arOptions = copyOrDefaultIfEmpty(toolchain.getArFlagList(), "rcsD");
    this.arThinArchivesOptions = copyOrDefaultIfEmpty(
        toolchain.getArThinArchivesFlagList(), "rcsDT");

    this.abi = toolchain.getAbiVersion();
    this.abiGlibcVersion = toolchain.getAbiLibcVersion();

    // The default value for optional string attributes is the empty string.
    PathFragment defaultSysroot = toolchain.getBuiltinSysroot().length() == 0
        ? null
        : new PathFragment(toolchain.getBuiltinSysroot());
    if ((defaultSysroot != null) && !defaultSysroot.isNormalized()) {
      throw new IllegalArgumentException("The built-in sysroot '" + defaultSysroot
          + "' is not normalized.");
    }

    if ((cppOptions.libcTop != null) && (defaultSysroot == null)) {
      throw new InvalidConfigurationException("The selected toolchain " + toolchainIdentifier
          + " does not support setting --grte_top.");
    }
    LibcTop libcTop = cppOptions.libcTop;
    if ((libcTop == null) && !toolchain.getDefaultGrteTop().isEmpty()) {
      try {
        libcTop = new CppOptions.LibcTopConverter().convert(toolchain.getDefaultGrteTop());
      } catch (OptionsParsingException e) {
        throw new InvalidConfigurationException(e.getMessage(), e);
      }
    }
    if ((libcTop != null) && (libcTop.getLabel() != null)) {
      libcLabel = libcTop.getLabel();
    } else {
      libcLabel = null;
    }

    ImmutableList.Builder<PathFragment> builtInIncludeDirectoriesBuilder
        = ImmutableList.builder();
    sysroot = libcTop == null ? defaultSysroot : libcTop.getSysroot();
    for (String s : toolchain.getCxxBuiltinIncludeDirectoryList()) {
      builtInIncludeDirectoriesBuilder.add(
          resolveIncludeDir(s, sysroot, crosstoolTopPathFragment));
    }
    builtInIncludeDirectories = builtInIncludeDirectoriesBuilder.build();

    // The runtime sysroot should really be set from --grte_top. However, currently libc has no
    // way to set the sysroot. The CROSSTOOL file does set the runtime sysroot, in the
    // builtin_sysroot field. This implies that you can not arbitrarily mix and match Crosstool
    // and libc versions, you must always choose compatible ones.
    runtimeSysroot = defaultSysroot;

    String sysrootFlag;
    if (sysroot != null) {
      sysrootFlag = "--sysroot=" + sysroot;
    } else {
      sysrootFlag = null;
    }

    ImmutableList.Builder<String> unfilteredCoptsBuilder = ImmutableList.builder();
    if (sysrootFlag != null) {
      unfilteredCoptsBuilder.add(sysrootFlag);
    }
    unfilteredCoptsBuilder.addAll(toolchain.getUnfilteredCxxFlagList());
    unfilteredCompilerFlags = new FlagList(
        unfilteredCoptsBuilder.build(),
        convertOptionalOptions(toolchain.getOptionalUnfilteredCxxFlagList()),
        ImmutableList.<String>of());

    ImmutableList.Builder<String> linkoptsBuilder = ImmutableList.builder();
    linkoptsBuilder.addAll(cppOptions.linkoptList);
    if (cppOptions.experimentalOmitfp) {
      linkoptsBuilder.add("-Wl,--eh-frame-hdr");
    }
    if (sysrootFlag != null) {
      linkoptsBuilder.add(sysrootFlag);
    }
    this.linkOptions = linkoptsBuilder.build();

    ImmutableList.Builder<String> coptsBuilder = ImmutableList.<String>builder()
        .addAll(toolchain.getCompilerFlagList())
        .addAll(cFlags.get(compilationMode))
        .addAll(lipoCFlags.get(cppOptions.getLipoMode()));
    if (cppOptions.experimentalOmitfp) {
      coptsBuilder.add("-fomit-frame-pointer");
      coptsBuilder.add("-fasynchronous-unwind-tables");
      coptsBuilder.add("-DNO_FRAME_POINTER");
    }
    this.compilerFlags = new FlagList(
        coptsBuilder.build(),
        convertOptionalOptions(toolchain.getOptionalCompilerFlagList()),
        ImmutableList.copyOf(cppOptions.coptList));

    this.cOptions = ImmutableList.copyOf(cppOptions.conlyoptList);

    ImmutableList.Builder<String> cxxOptsBuilder = ImmutableList.<String>builder()
        .addAll(toolchain.getCxxFlagList())
        .addAll(cxxFlags.get(compilationMode))
        .addAll(lipoCxxFlags.get(cppOptions.getLipoMode()));

    this.cxxFlags = new FlagList(
        cxxOptsBuilder.build(),
        convertOptionalOptions(toolchain.getOptionalCxxFlagList()),
        ImmutableList.copyOf(cppOptions.cxxoptList));

    this.ldExecutable = getToolPathFragment(CppConfiguration.Tool.LD);

    boolean stripBinaries =
        (cppOptions.stripBinaries == StripMode.ALWAYS)
            || ((cppOptions.stripBinaries == StripMode.SOMETIMES)
                && (compilationMode == CompilationMode.FASTBUILD));

    fullyStaticLinkFlags = new FlagList(
        configureLinkerOptions(compilationMode, lipoMode, LinkingMode.FULLY_STATIC,
                               ldExecutable, stripBinaries),
        convertOptionalOptions(toolchain.getOptionalLinkerFlagList()),
        ImmutableList.<String>of());
    mostlyStaticLinkFlags = new FlagList(
        configureLinkerOptions(compilationMode, lipoMode, LinkingMode.MOSTLY_STATIC,
                               ldExecutable, stripBinaries),
        convertOptionalOptions(toolchain.getOptionalLinkerFlagList()),
        ImmutableList.<String>of());
    mostlyStaticSharedLinkFlags = new FlagList(
        configureLinkerOptions(compilationMode, lipoMode,
                               LinkingMode.MOSTLY_STATIC_LIBRARIES, ldExecutable, stripBinaries),
        convertOptionalOptions(toolchain.getOptionalLinkerFlagList()),
        ImmutableList.<String>of());
    dynamicLinkFlags = new FlagList(
        configureLinkerOptions(compilationMode, lipoMode, LinkingMode.DYNAMIC,
                               ldExecutable, stripBinaries),
        convertOptionalOptions(toolchain.getOptionalLinkerFlagList()),
        ImmutableList.<String>of());
    testOnlyLinkFlags = ImmutableList.copyOf(toolchain.getTestOnlyLinkerFlagList());

    Map<String, String> makeVariablesBuilder = new HashMap<>();
    // The following are to be used to allow some build rules to avoid the limits on stack frame
    // sizes and variable-length arrays. Ensure that these are always set.
    makeVariablesBuilder.put("STACK_FRAME_UNLIMITED", "");
    makeVariablesBuilder.put("CC_FLAGS", "");
    for (CrosstoolConfig.MakeVariable variable : toolchain.getMakeVariableList()) {
      makeVariablesBuilder.put(variable.getName(), variable.getValue());
    }
    if (sysrootFlag != null) {
      String ccFlags = makeVariablesBuilder.get("CC_FLAGS");
      ccFlags = ccFlags.isEmpty() ? sysrootFlag : ccFlags + " " + sysrootFlag;
      makeVariablesBuilder.put("CC_FLAGS", ccFlags);
    }
    this.additionalMakeVariables = ImmutableMap.copyOf(makeVariablesBuilder);
  }

  private ImmutableList<OptionalFlag> convertOptionalOptions(
          List<CrosstoolConfig.CToolchain.OptionalFlag> optionalFlagList)
      throws IllegalArgumentException {
    ImmutableList.Builder<OptionalFlag> result = ImmutableList.builder();

    for (CrosstoolConfig.CToolchain.OptionalFlag crosstoolOptionalFlag : optionalFlagList) {
      String name = crosstoolOptionalFlag.getDefaultSettingName();
      result.add(new OptionalFlag(
          name,
          ImmutableList.copyOf(crosstoolOptionalFlag.getFlagList())));
    }

    return result.build();
  }

  // TODO(bazel-team): Remove this once bazel supports all crosstool flags through
  // feature configuration, and all crosstools have been converted.
  private CToolchain addLegacyFeatures(CToolchain toolchain) {
    CToolchain.Builder toolchainBuilder = CToolchain.newBuilder();
    ImmutableSet.Builder<String> featuresBuilder = ImmutableSet.builder();
    for (CToolchain.Feature feature : toolchain.getFeatureList()) {
      featuresBuilder.add(feature.getName());
    }
    Set<String> features = featuresBuilder.build();
    if (features.contains(CppRuleClasses.NO_LEGACY_FEATURES)) {
      // The toolchain requested to not get any legacy features enabled.
      return toolchain;
    }
    try {
      if (!features.contains("include_paths")) {
        TextFormat.merge(""
            + "feature {"
            + "  name: 'include_paths'"
            + "  flag_set {"
            + "    action: 'preprocess-assemble'"
            + "    action: 'c-compile'"
            + "    action: 'c++-compile'"
            + "    action: 'c++-header-parsing'"
            + "    action: 'c++-header-preprocessing'"
            + "    action: 'c++-module-compile'"
            + "    flag_group {"
            + "      flag: '-iquote'"
            + "      flag: '%{quote_include_paths}'"
            + "    }"
            + "    flag_group {"
            + "      flag: '-I%{include_paths}'"
            + "    }"
            + "    flag_group {"
            + "      flag: '-isystem'"
            + "      flag: '%{system_include_paths}'"
            + "    }"
            + "  }"
            + "}",
            toolchainBuilder);
      }
      if (!features.contains("fdo_instrument")) {
        TextFormat.merge(
            ""
                + "feature {"
                + "  name: 'fdo_instrument'"
                + "  provides: 'profile'"
                + "  flag_set {"
                + "    action: 'c-compile'"
                + "    action: 'c++-compile'"
                + "    action: 'c++-link'"
                + "    flag_group {"
                + "      flag: '-Xgcc-only=-fprofile-generate=%{fdo_instrument_path}'"
                + "      flag: '-Xclang-only=-fprofile-instr-generate=%{fdo_instrument_path}'"
                + "    }"
                + "    flag_group {"
                + "      flag: '-fno-data-sections'"
                + "    }"
                + "  }"
                + "}",
            toolchainBuilder);
      }
      if (!features.contains("fdo_optimize")) {
        TextFormat.merge(
            ""
                + "feature {"
                + "  name: 'fdo_optimize'"
                + "  provides: 'profile'"
                + "  flag_set {"
                + "    action: 'c-compile'"
                + "    action: 'c++-compile'"
                + "    expand_if_all_available: 'fdo_profile_path'"
                + "    flag_group {"
                + "      flag: '-Xgcc-only=-fprofile-use=%{fdo_profile_path}'"
                + "      flag: '-Xclang-only=-fprofile-instr-use=%{fdo_profile_path}'"
                + "      flag: '-Xclang-only=-Wno-profile-instr-unprofiled'"
                + "      flag: '-Xclang-only=-Wno-profile-instr-out-of-date'"
                + "      flag: '-fprofile-correction'"
                + "    }"
                + "  }"
                + "}",
            toolchainBuilder);
      }
      if (!features.contains("autofdo")) {
        TextFormat.merge(
            ""
                + "feature {"
                + "  name: 'autofdo'"
                + "  provides: 'profile'"
                + "  flag_set {"
                + "    action: 'c-compile'"
                + "    action: 'c++-compile'"
                + "    expand_if_all_available: 'fdo_profile_path'"
                + "    flag_group {"
                + "      flag: '-fauto-profile=%{fdo_profile_path}'"
                + "      flag: '-fprofile-correction'"
                + "    }"
                + "  }"
                + "}",
            toolchainBuilder);
      }
      if (!features.contains("lipo")) {
        TextFormat.merge(
            ""
                + "feature {"
                + "  name: 'lipo'"
                + "  requires { feature: 'autofdo' }"
                + "  requires { feature: 'fdo_optimize' }"
                + "  requires { feature: 'fdo_instrument' }"
                + "  flag_set {"
                + "    action: 'c-compile'"
                + "    action: 'c++-compile'"
                + "    flag_group {"
                + "      flag: '-fripa'"
                + "    }"
                + "  }"
                + "}",
            toolchainBuilder);
      }
      if (!features.contains("coverage")) {
        TextFormat.merge(
            ""
                + "feature {"
                + "  name: 'coverage'"
                + "  provides: 'profile'"
                + "  flag_set {"
                + "    action: 'preprocess-assemble'"
                + "    action: 'c-compile'"
                + "    action: 'c++-compile'"
                + "    action: 'c++-header-parsing'"
                + "    action: 'c++-header-preprocessing'"
                + "    action: 'c++-module-compile'"
                + "    expand_if_all_available: 'gcov_gcno_file'"
                + "    flag_group {"
                + "      flag: '-fprofile-arcs'"
                + "      flag: '-ftest-coverage'"
                + "    }"
                + "  }"
                + "  flag_set {"
                + "    action: 'c++-link'"
                + "    flag_group {"
                + "      flag: '-lgcov'"
                + "    }"
                + "  }"
                + "}",
            toolchainBuilder);
      }
    } catch (ParseException e) {
      // Can only happen if we change the proto definition without changing our configuration above.
      throw new RuntimeException(e);
    }
    toolchainBuilder.mergeFrom(toolchain);
    return toolchainBuilder.build();
  }

  private static ImmutableList<String> copyOrDefaultIfEmpty(List<String> list,
      String defaultValue) {
    return list.isEmpty() ? ImmutableList.of(defaultValue) : ImmutableList.copyOf(list);
  }

  @VisibleForTesting
  static CompilationMode importCompilationMode(CrosstoolConfig.CompilationMode mode) {
    return CompilationMode.valueOf(mode.name());
  }

  @VisibleForTesting
  static LinkingMode importLinkingMode(CrosstoolConfig.LinkingMode mode) {
    return LinkingMode.valueOf(mode.name());
  }

  private static final String SYSROOT_START = "%sysroot%/";
  private static final String WORKSPACE_START = "%workspace%/";
  private static final String CROSSTOOL_START = "%crosstool_top%/";
  private static final String PACKAGE_START = "%package(", PACKAGE_END = ")%";

  /**
   * Resolve the given include directory.
   *
   * <p>If it starts with %sysroot%/, that part is replaced with the actual sysroot.
   *
   * <p>If it starts with %workspace%/, that part is replaced with the empty string
   * (essentially making it relative to the build directory).
   *
   * <p>If it starts with %crosstool_top%/ or is any relative path, it is
   * interpreted relative to the crosstool top. The use of assumed-crosstool-relative
   * specifications is considered deprecated, and all such uses should eventually
   * be replaced by "%crosstool_top%/".
   *
   * <p>If it is of the form %package(@repository//my/package)%/folder, then it is
   * interpreted as the named folder in the appropriate package. All of the normal
   * package syntax is supported. The /folder part is optional.
   *
   * <p>It is illegal if it starts with a % and does not match any of the above
   * forms to avoid accidentally silently ignoring misspelled prefixes.
   *
   * <p>If it is absolute, it remains unchanged.
   */
  static PathFragment resolveIncludeDir(String s, PathFragment sysroot,
      PathFragment crosstoolTopPathFragment) throws InvalidConfigurationException {
    PathFragment pathPrefix;
    String pathString;
    int packageEndIndex = s.indexOf(PACKAGE_END);
    if (packageEndIndex != -1 && s.startsWith(PACKAGE_START)) {
      String packageString = s.substring(PACKAGE_START.length(), packageEndIndex);
      try {
        pathPrefix = PackageIdentifier.parse(packageString).getPathFragment();
      } catch (LabelSyntaxException e) {
        throw new InvalidConfigurationException("The package '" + packageString + "' is not valid");
      }
      int pathStartIndex = packageEndIndex + PACKAGE_END.length();
      if (pathStartIndex + 1 < s.length()) {
        if (s.charAt(pathStartIndex) != '/') {
          throw new InvalidConfigurationException(
              "The path in the package for '" + s + "' is not valid");
        }
        pathString = s.substring(pathStartIndex + 1, s.length());
      } else {
        pathString = "";
      }
    } else if (s.startsWith(SYSROOT_START)) {
      if (sysroot == null) {
        throw new InvalidConfigurationException("A %sysroot% prefix is only allowed if the "
            + "default_sysroot option is set");
      }
      pathPrefix = sysroot;
      pathString = s.substring(SYSROOT_START.length(), s.length());
    } else if (s.startsWith(WORKSPACE_START)) {
      pathPrefix = PathFragment.EMPTY_FRAGMENT;
      pathString = s.substring(WORKSPACE_START.length(), s.length());
    } else {
      pathPrefix = crosstoolTopPathFragment;
      if (s.startsWith(CROSSTOOL_START)) {
        pathString = s.substring(CROSSTOOL_START.length(), s.length());
      } else if (s.startsWith("%")) {
        throw new InvalidConfigurationException(
            "The include path '" + s + "' has an " + "unrecognized %prefix%");
      } else {
        pathString = s;
      }
    }

    PathFragment path = new PathFragment(pathString);
    if (!path.isNormalized()) {
      throw new InvalidConfigurationException("The include path '" + s + "' is not normalized.");
    }
    return pathPrefix.getRelative(path);
  }

  @VisibleForTesting
  ImmutableList<String> configureLinkerOptions(
      CompilationMode compilationMode, LipoMode lipoMode, LinkingMode linkingMode,
      PathFragment ldExecutable, boolean stripBinaries) {
    List<String> result = new ArrayList<>();
    result.addAll(commonLinkOptions);

    if (stripBinaries) {
      result.add("-Wl,-S");
    }

    result.addAll(linkOptionsFromCompilationMode.get(compilationMode));
    result.addAll(linkOptionsFromLipoMode.get(lipoMode));
    result.addAll(linkOptionsFromLinkingMode.get(linkingMode));
    return ImmutableList.copyOf(result);
  }

  /**
   * Returns the toolchain identifier, which uniquely identifies the compiler
   * version, target libc version, target cpu, and LIPO linkage.
   */
  public String getToolchainIdentifier() {
    return toolchainIdentifier;
  }

  /**
   * Returns the system name which is required by the toolchain to run.
   */
  public String getHostSystemName() {
    return hostSystemName;
  }

  @Override
  public String toString() {
    return toolchainIdentifier;
  }

  /**
   * Returns the compiler version string (e.g. "gcc-4.1.1").
   */
  @SkylarkCallable(name = "compiler", structField = true, doc = "C++ compiler.")
  public String getCompiler() {
    return compiler;
  }

  /**
   * Returns the libc version string (e.g. "glibc-2.2.2").
   */
  @SkylarkCallable(name = "libc", structField = true, doc = "libc version string.")
  public String getTargetLibc() {
    return targetLibc;
  }

  /**
   * Returns the target architecture using blaze-specific constants (e.g. "piii").
   */
  @SkylarkCallable(name = "cpu", structField = true, doc = "Target CPU of the C++ toolchain.")
  public String getTargetCpu() {
    return targetCpu;
  }

  /**
   * Returns the path fragment that is either absolute or relative to the
   * execution root that can be used to execute the given tool.
   *
   * <p>Note that you must not use this method to get the linker location, but
   * use {@link #getLdExecutable} instead!
   */
  public PathFragment getToolPathFragment(CppConfiguration.Tool tool) {
    return toolPaths.get(tool.getNamePart());
  }

  /**
   * Returns a label that forms a dependency to the files required for the
   * sysroot that is used.
   */
  public Label getLibcLabel() {
    return libcLabel;
  }

  /**
   * Returns a label that references the library files needed to statically
   * link the C++ runtime (i.e. libgcc.a, libgcc_eh.a, libstdc++.a) for the
   * target architecture.
   */
  public Label getStaticRuntimeLibsLabel() {
    return supportsEmbeddedRuntimes() ? staticRuntimeLibsLabel : null;
  }

  /**
   * Returns a label that references the library files needed to dynamically
   * link the C++ runtime (i.e. libgcc_s.so, libstdc++.so) for the target
   * architecture.
   */
  public Label getDynamicRuntimeLibsLabel() {
    return supportsEmbeddedRuntimes() ? dynamicRuntimeLibsLabel : null;
  }

  /**
   * Returns the label of the <code>cc_compiler</code> rule for the C++ configuration.
   */
  public Label getCcToolchainRuleLabel() {
    return ccToolchainLabel;
  }

  /**
   * Returns the abi we're using, which is a gcc version. E.g.: "gcc-3.4".
   * Note that in practice we might be using gcc-3.4 as ABI even when compiling
   * with gcc-4.1.0, because ABIs are backwards compatible.
   */
  // TODO(bazel-team): The javadoc should clarify how this is used in Blaze.
  public String getAbi() {
    return abi;
  }

  /**
   * Returns the glibc version used by the abi we're using.  This is a
   * glibc version number (e.g., "2.2.2").  Note that in practice we
   * might be using glibc 2.2.2 as ABI even when compiling with
   * gcc-4.2.2, gcc-4.3.1, or gcc-4.4.0 (which use glibc 2.3.6),
   * because ABIs are backwards compatible.
   */
  // TODO(bazel-team): The javadoc should clarify how this is used in Blaze.
  public String getAbiGlibcVersion() {
    return abiGlibcVersion;
  }

  /**
   * Returns the configured features of the toolchain. Rules should not call this directly, but
   * instead use {@code CcToolchainProvider.getFeatures}.
   */
  public CcToolchainFeatures getFeatures() {
    return toolchainFeatures;
  }

  /**
   * Returns the configured current compilation mode. Rules should not call this directly, but
   * instead use {@code CcToolchainProvider.getCompilationMode}.
   */
  public CompilationMode getCompilationMode() {
    return compilationMode;
  }

  /**
   * Returns whether the toolchain supports the gold linker.
   */
  public boolean supportsGoldLinker() {
    return supportsGoldLinker;
  }

  /**
   * Returns whether the toolchain supports thin archives.
   */
  public boolean supportsThinArchives() {
    return supportsThinArchives;
  }

  /**
   * Returns whether the toolchain supports the --start-lib/--end-lib options.
   */
  public boolean supportsStartEndLib() {
    return supportsStartEndLib;
  }

  /**
   * Returns whether the toolchain supports dynamic linking.
   */
  public boolean supportsDynamicLinker() {
    return supportsDynamicLinker;
  }

  /**
   * Returns whether build_interface_so can build interface shared objects for this toolchain.
   * Should be true if this toolchain generates ELF objects.
   */
  public boolean supportsInterfaceSharedObjects() {
    return supportsInterfaceSharedObjects;
  }

  /**
   * Returns whether the toolchain supports linking C/C++ runtime libraries
   * supplied inside the toolchain distribution.
   */
  public boolean supportsEmbeddedRuntimes() {
    return supportsEmbeddedRuntimes;
  }

  /**
   * Returns whether the toolchain supports EXEC_ORIGIN libraries resolution.
   */
  public boolean supportsExecOrigin() {
    // We're rolling out support for this in the same release that also supports embedded runtimes.
    return supportsEmbeddedRuntimes;
  }

  /**
   * Returns whether the toolchain supports "Fission" C++ builds, i.e. builds
   * where compilation partitions object code and debug symbols into separate
   * output files.
   */
  public boolean supportsFission() {
    return supportsFission;
  }

  /**
   * Returns whether shared libraries must be compiled with position
   * independent code on this platform.
   */
  public boolean toolchainNeedsPic() {
    return toolchainNeedsPic;
  }

  /**
   * Returns whether binaries must be compiled with position independent code.
   */
  public boolean usePicForBinaries() {
    return usePicForBinaries;
  }

  /**
   * Returns the type of archives being used.
   */
  public Link.ArchiveType archiveType() {
    if (useStartEndLib()) {
      return Link.ArchiveType.START_END_LIB;
    }
    if (useThinArchives()) {
      return Link.ArchiveType.THIN;
    }
    return Link.ArchiveType.FAT;
  }

  /**
   * Returns the ar flags to be used.
   */
  public ImmutableList<String> getArFlags(boolean thinArchives) {
    return thinArchives ? arThinArchivesOptions : arOptions;
  }

  /**
   * Returns the built-in list of system include paths for the toolchain
   * compiler. All paths in this list should be relative to the exec directory.
   * They may be absolute if they are also installed on the remote build nodes or
   * for local compilation.
   */
  public List<PathFragment> getBuiltInIncludeDirectories() {
    return builtInIncludeDirectories;
  }

  /**
   * Returns the built-in header automatically included by the toolchain compiler. All C++ files
   * may implicitly include this file. May be null if {@link #getSysroot} is null.
   */
  @Nullable
  public Artifact getBuiltInIncludeFile() {
    return builtInIncludeFile;
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
  public PathFragment getSysroot() {
    return sysroot;
  }

  /**
   * Returns the run time sysroot, which is where the dynamic linker
   * and system libraries are found at runtime.  This is usually an absolute path. If the
   * toolchain compiler does not support sysroots, then this method returns <code>null</code>.
   */
  public PathFragment getRuntimeSysroot() {
    return runtimeSysroot;
  }

  /**
   * Returns the default options to use for compiling C, C++, and assembler.
   * This is just the options that should be used for all three languages.
   * There may be additional C-specific or C++-specific options that should be used,
   * in addition to the ones returned by this method.
   */
  @SkylarkCallable(
    name = "compiler_options",
    doc =
        "Returns the default options to use for compiling C, C++, and assembler. "
            + "This is just the options that should be used for all three languages. "
            + "There may be additional C-specific or C++-specific options that should be used, "
            + "in addition to the ones returned by this method"
  )
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
   * Returns the list of additional C++-specific options to use for compiling
   * C++. These should be go on the command line after the common options
   * returned by {@link #getCompilerOptions}.
   */
  @SkylarkCallable(
    name = "cxx_options",
    doc =
        "Returns the list of additional C++-specific options to use for compiling C++. "
            + "These should be go on the command line after the common options returned by "
            + "<code>compiler_options</code>"
  )
  public ImmutableList<String> getCxxOptions(Iterable<String> features) {
    return cxxFlags.evaluate(features);
  }

  /**
   * Returns the default list of options which cannot be filtered by BUILD
   * rules. These should be appended to the command line after filtering.
   */
  @SkylarkCallable(
    name = "unfiltered_compiler_options",
    doc =
        "Returns the default list of options which cannot be filtered by BUILD "
            + "rules. These should be appended to the command line after filtering."
  )
  public ImmutableList<String> getUnfilteredCompilerOptions(Iterable<String> features) {
    return unfilteredCompilerFlags.evaluate(features);
  }

  /**
   * Returns the set of command-line linker options, including any flags
   * inferred from the command-line options.
   *
   * @see Link
   */
  // TODO(bazel-team): Clean up the linker options computation!
  @SkylarkCallable(name = "link_options", structField = true,
      doc = "Returns the set of command-line linker options, including any flags "
      + "inferred from the command-line options.")
  public ImmutableList<String> getLinkOptions() {
    return linkOptions;
  }

  /**
   * Returns the immutable list of linker options for fully statically linked
   * outputs. Does not include command-line options passed via --linkopt or
   * --linkopts.
   *
   * @param features default settings affecting this link
   * @param sharedLib true if the output is a shared lib, false if it's an executable
   */
  @SkylarkCallable(
    name = "fully_static_link_options",
    doc =
        "Returns the immutable list of linker options for fully statically linked "
            + "outputs. Does not include command-line options passed via --linkopt or "
            + "--linkopts."
  )
  public ImmutableList<String> getFullyStaticLinkOptions(Iterable<String> features,
      Boolean sharedLib) {
    if (sharedLib) {
      return getSharedLibraryLinkOptions(mostlyStaticLinkFlags, features);
    } else {
      return fullyStaticLinkFlags.evaluate(features);
    }
  }

  /**
   * Returns the immutable list of linker options for mostly statically linked
   * outputs. Does not include command-line options passed via --linkopt or
   * --linkopts.
   *
   * @param features default settings affecting this link
   * @param sharedLib true if the output is a shared lib, false if it's an executable
   */
  @SkylarkCallable(
    name = "mostly_static_link_options",
    doc =
        "Returns the immutable list of linker options for mostly statically linked "
            + "outputs. Does not include command-line options passed via --linkopt or "
            + "--linkopts."
  )
  public ImmutableList<String> getMostlyStaticLinkOptions(Iterable<String> features,
      Boolean sharedLib) {
    if (sharedLib) {
      return getSharedLibraryLinkOptions(
          supportsEmbeddedRuntimes ? mostlyStaticSharedLinkFlags : dynamicLinkFlags,
          features);
    } else {
      return mostlyStaticLinkFlags.evaluate(features);
    }
  }

  /**
   * Returns the immutable list of linker options for artifacts that are not
   * fully or mostly statically linked. Does not include command-line options
   * passed via --linkopt or --linkopts.
   *
   * @param features default settings affecting this link
   * @param sharedLib true if the output is a shared lib, false if it's an executable
   */
  @SkylarkCallable(
    name = "dynamic_link_options",
    doc =
        "Returns the immutable list of linker options for artifacts that are not "
            + "fully or mostly statically linked. Does not include command-line options "
            + "passed via --linkopt or --linkopts."
  )
  public ImmutableList<String> getDynamicLinkOptions(Iterable<String> features, Boolean sharedLib) {
    if (sharedLib) {
      return getSharedLibraryLinkOptions(dynamicLinkFlags, features);
    } else {
      return dynamicLinkFlags.evaluate(features);
    }
  }

  /**
   * Returns link options for the specified flag list, combined with universal options
   * for all shared libraries (regardless of link staticness).
   */
  private ImmutableList<String> getSharedLibraryLinkOptions(FlagList flags,
      Iterable<String> features) {
    return ImmutableList.<String>builder()
        .addAll(flags.evaluate(features))
        .addAll(dynamicLibraryLinkFlags.evaluate(features))
        .build();
  }

  /**
   * Returns test-only link options such that certain test-specific features can be configured
   * separately (e.g. lazy binding).
   */
  public ImmutableList<String> getTestOnlyLinkOptions() {
    return testOnlyLinkFlags;
  }


  /**
   * Returns the list of options to be used with 'objcopy' when converting
   * binary files to object files, or {@code null} if this operation is not
   * supported.
   */
  public ImmutableList<String> getObjCopyOptionsForEmbedding() {
    return objcopyOptions;
  }

  /**
   * Returns the list of options to be used with 'ld' when converting
   * binary files to object files, or {@code null} if this operation is not
   * supported.
   */
  public ImmutableList<String> getLdOptionsForEmbedding() {
    return ldOptions;
  }

  /**
   * Returns a map of additional make variables for use by {@link
   * BuildConfiguration}. These are to used to allow some build rules to
   * avoid the limits on stack frame sizes and variable-length arrays.
   *
   * <p>The returned map must contain an entry for {@code STACK_FRAME_UNLIMITED},
   * though the entry may be an empty string.
   */
  @VisibleForTesting
  public ImmutableMap<String, String> getAdditionalMakeVariables() {
    return additionalMakeVariables;
  }

  /**
   * Returns the execution path to the linker binary to use for this build.
   * Relative paths are relative to the execution root.
   */
  public PathFragment getLdExecutable() {
    return ldExecutable;
  }

  /**
   * Returns the dynamic linking mode (full, off, or default).
   */
  public DynamicMode getDynamicMode() {
    return dynamicMode;
  }

  /*
   * If true then the directory name for non-LIPO targets will have a '-lipodata' suffix in
   * AutoFDO mode.
   */
  public boolean getAutoFdoLipoData() {
    return cppOptions.autoFdoLipoData;
  }

  /**
   * Returns the STL label if given on the command line. {@code null}
   * otherwise.
   */
  public Label getStl() {
    return cppOptions.stl;
  }

  /*
   * Returns the command-line "Make" variable overrides.
   */
  @Override
  public ImmutableMap<String, String> getCommandLineDefines() {
    return commandLineDefines;
  }

  /**
   * Returns the command-line override value for the specified "Make" variable
   * for this configuration, or null if none.
   */
  public String getMakeVariableOverride(String var) {
    return commandLineDefines.get(var);
  }

  /**
   * Returns the currently active LIPO compilation mode.
   */
  public LipoMode getLipoMode() {
    return cppOptions.lipoMode;
  }

  public boolean isFdo() {
    return cppOptions.isFdo();
  }

  public boolean isLipoOptimization() {
    // The LIPO optimization bits are set in the LIPO context collector configuration, too.
    return cppOptions.isLipoOptimization() && !isLipoContextCollector();
  }

  public boolean isLipoOptimizationOrInstrumentation() {
    return cppOptions.isLipoOptimizationOrInstrumentation();
  }

  /**
   * Returns true if it is AutoFDO LIPO build.
   */
  public boolean isAutoFdoLipo() {
    return cppOptions.fdoOptimize != null && FdoSupport.isAutoFdo(cppOptions.fdoOptimize)
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

  public Label getLipoContextLabel() {
    return cppOptions.getLipoContextLabel();
  }

  /**
   * Returns the custom malloc library label.
   */
  public Label customMalloc() {
    return cppOptions.customMalloc;
  }

  /**
   * Returns the extra warnings enabled for C compilation.
   */
  public ImmutableList<String> getCWarns() {
    return ImmutableList.copyOf(cppOptions.cWarns);
  }

  /**
   * Returns true if mostly-static C++ binaries should be skipped.
   */
  public boolean skipStaticOutputs() {
    return cppOptions.skipStaticOutputs;
  }
  
  /**
   * Returns whether we are processing headers in dependencies of built C++ targets.
   */
  public boolean processHeadersInDependencies() {
    return cppOptions.processHeadersInDependencies;
  }

  /**
   * Returns true if Fission is specified for this build and supported by the crosstool.
   */
  public boolean useFission() {
    return cppOptions.fissionModes.contains(compilationMode) && supportsFission();
  }

  /**
   * Returns true if Fission is enabled for this build and the user requested automatic building
   * of .dwp files for C++ test targets.
   */
  public boolean shouldBuildTestDwp() {
    return useFission() && cppOptions.buildTestDwp;
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

  public boolean useStartEndLib() {
    return cppOptions.useStartEndLib && supportsStartEndLib();
  }

  public boolean useThinArchives() {
    return cppOptions.useThinArchives && supportsThinArchives();
  }

  /**
   * Returns true if interface shared objects should be used.
   */
  public boolean useInterfaceSharedObjects() {
    return supportsInterfaceSharedObjects() && cppOptions.useInterfaceSharedObjects;
  }

  public boolean forceIgnoreDashStatic() {
    return cppOptions.forceIgnoreDashStatic;
  }

  /**
   * Returns true if shared libraries must be compiled with position independent code
   * on this platform or in this configuration.
   */
  public boolean needsPic() {
    return forcePic() || toolchainNeedsPic();
  }

  /**
   * Returns true iff we should use ".pic.o" files when linking executables.
   */
  public boolean usePicObjectsForBinaries() {
    return forcePic() || usePicForBinaries();
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

  public boolean useIsystemForIncludes() {
    return cppOptions.useIsystemForIncludes;
  }

  public LibcTop getLibcTop() {
    return cppOptions.libcTop;
  }

  public boolean getUseInterfaceSharedObjects() {
    return cppOptions.useInterfaceSharedObjects;
  }

  /**
   * Returns the FDO support object.
   */
  public FdoSupport getFdoSupport() {
    return fdoSupport;
  }

  /**
   * Return the name of the directory (relative to the bin directory) that
   * holds mangled links to shared libraries. This name is always set to
   * the '{@code _solib_<cpu_archictecture_name>}.
   */
  public String getSolibDirectory() {
    return solibDirectory;
  }

  /**
   * Returns the path to the GNU binutils 'objcopy' binary to use for this
   * build. (Corresponds to $(OBJCOPY) in make-dbg.) Relative paths are
   * relative to the execution root.
   */
  @SkylarkCallable(name = "objcopy_executable", structField = true,
      doc = "Path to GNU binutils 'objcopy' binary")
  public PathFragment getObjCopyExecutable() {
    return getToolPathFragment(CppConfiguration.Tool.OBJCOPY);
  }

  /**
   * Returns the path to the GNU binutils 'gcc' binary that should be used
   * by this build.  This binary should support compilation of both C (*.c)
   * and C++ (*.cc) files. Relative paths are relative to the execution root.
   */
  @SkylarkCallable(name = "compiler_executable", structField = true,
      doc = "Path to C/C++ compiler binary")
  public PathFragment getCppExecutable() {
    return getToolPathFragment(CppConfiguration.Tool.GCC);
  }

  /**
   * Returns the path to the GNU binutils 'g++' binary that should be used
   * by this build.  This binary should support linking of both C (*.c)
   * and C++ (*.cc) files. Relative paths are relative to the execution root.
   */
  public PathFragment getCppLinkExecutable() {
    return getToolPathFragment(CppConfiguration.Tool.GCC);
  }

  /**
   * Returns the path to the GNU binutils 'cpp' binary that should be used
   * by this build. Relative paths are relative to the execution root.
   */
  public PathFragment getCpreprocessorExecutable() {
    return getToolPathFragment(CppConfiguration.Tool.CPP);
  }

  /**
   * Returns the path to the GNU binutils 'gcov' binary that should be used
   * by this build to analyze C++ coverage data. Relative paths are relative to
   * the execution root.
   */
  public PathFragment getGcovExecutable() {
    return getToolPathFragment(CppConfiguration.Tool.GCOV);
  }

  /**
   * Returns the path to the 'gcov-tool' executable that should be used
   * by this build. Relative paths are relative to the execution root.
   */
  public PathFragment getGcovToolExecutable() {
    return getToolPathFragment(CppConfiguration.Tool.GCOVTOOL);
  }

  /**
   * Returns the path to the GNU binutils 'nm' executable that should be used
   * by this build. Used only for testing. Relative paths are relative to the
   * execution root.
   */
  @SkylarkCallable(name = "nm_executable", structField = true,
      doc = "Path to GNU binutils 'nm' binary")
  public PathFragment getNmExecutable() {
    return getToolPathFragment(CppConfiguration.Tool.NM);
  }

  /**
   * Returns the path to the GNU binutils 'objdump' executable that should be
   * used by this build. Used only for testing. Relative paths are relative to
   * the execution root.
   */
  @SkylarkCallable(name = "objdump_executable", structField = true,
      doc = "Path to GNU binutils 'objdump' binary")
  public PathFragment getObjdumpExecutable() {
    return getToolPathFragment(CppConfiguration.Tool.OBJDUMP);
  }

  /**
   * Returns the path to the GNU binutils 'ar' binary to use for this build.
   * Relative paths are relative to the execution root.
   */
  @SkylarkCallable(name = "ar_executable", structField = true,
      doc = "Path to GNU binutils 'ar' binary")
  public PathFragment getArExecutable() {
    return getToolPathFragment(CppConfiguration.Tool.AR);
  }

  /**
   * Returns the path to the GNU binutils 'strip' executable that should be used
   * by this build. Relative paths are relative to the execution root.
   */
  @SkylarkCallable(name = "strip_executable", structField = true,
      doc = "Path to GNU binutils 'strip' binary")
  public PathFragment getStripExecutable() {
    return getToolPathFragment(CppConfiguration.Tool.STRIP);
  }

  /**
   * Returns the path to the GNU binutils 'dwp' binary that should be used by this
   * build to combine debug info output from individual C++ compilations (i.e. .dwo
   * files) into aggregate target-level debug packages. Relative paths are relative to the
   * execution root. See https://gcc.gnu.org/wiki/DebugFission .
   */
  public PathFragment getDwpExecutable() {
    return getToolPathFragment(CppConfiguration.Tool.DWP);
  }

  /**
   * Returns the GNU System Name
   */
  @SkylarkCallable(name = "target_gnu_system_name", structField = true,
      doc = "The GNU System Name.")
  public String getTargetGnuSystemName() {
    return targetSystemName;
  }

  /**
   * Returns the architecture component of the GNU System Name
   */
  public String getGnuSystemArch() {
    if (targetSystemName.indexOf('-') == -1) {
      return targetSystemName;
    }
    return targetSystemName.substring(0, targetSystemName.indexOf('-'));
  }

  /**
   * Returns whether the configuration's purpose is only to collect LIPO-related data.
   */
  public boolean isLipoContextCollector() {
    return lipoContextCollector;
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
        reporter.handle(Event.warn("Stripping enabled, but '--copt=-g' (or --per_file_copt=...@-g) "
            + "specified. Debug information will be generated and then stripped away. This is "
            + "probably not what you want! Use '-c dbg' for debug mode, or use '--strip=never' "
            + "to disable stripping"));
      }
    }

    if (cppOptions.fdoInstrument != null && cppOptions.fdoOptimize != null) {
      reporter.handle(Event.error("Cannot instrument and optimize for FDO at the same time. "
          + "Remove one of the '--fdo_instrument' and '--fdo_optimize' options"));
    }

    if (cppOptions.lipoContext != null) {
      if (cppOptions.lipoMode != LipoMode.BINARY || cppOptions.fdoOptimize == null) {
        reporter.handle(Event.warn("The --lipo_context option can only be used together with "
            + "--fdo_optimize=<profile zip> and --lipo=binary. LIPO context will be ignored."));
      }
    } else {
      if (cppOptions.lipoMode == LipoMode.BINARY && cppOptions.fdoOptimize != null) {
        reporter.handle(Event.error("The --lipo_context option must be specified when using "
            + "--fdo_optimize=<profile zip> and --lipo=binary"));
      }
    }
    if (cppOptions.lipoMode == LipoMode.BINARY && compilationMode != CompilationMode.OPT) {
      reporter.handle(Event.error(
          "'--lipo=binary' can only be used with '--compilation_mode=opt' (or '-c opt')"));
    }

    if (cppOptions.fissionModes.contains(compilationMode) && !supportsFission()) {
      reporter.handle(
          Event.warn(
              "Fission is not supported by this crosstool. Please use a supporting "
                  + "crosstool to enable fission"));
    }
    if (cppOptions.buildTestDwp && !useFission()) {
      reporter.handle(Event.warn("Test dwp file requested, but Fission is not enabled. To "
          + "generate a dwp for the test executable, use '--fission=yes' with a toolchain "
          + "that supports Fission and build statically."));
    }
  }

  @Override
  public void addGlobalMakeVariables(Builder<String, String> globalMakeEnvBuilder) {
    // hardcoded CC->gcc setting for unit tests
    globalMakeEnvBuilder.put("CC", getCppExecutable().getPathString());

    // Make variables provided by crosstool/gcc compiler suite.
    globalMakeEnvBuilder.put("AR", getArExecutable().getPathString());
    globalMakeEnvBuilder.put("NM", getNmExecutable().getPathString());
    globalMakeEnvBuilder.put("OBJCOPY", getObjCopyExecutable().getPathString());
    globalMakeEnvBuilder.put("STRIP", getStripExecutable().getPathString());

    PathFragment gcovtool = getGcovToolExecutable();
    if (gcovtool != null) {
      // gcov-tool is optional in Crosstool
      globalMakeEnvBuilder.put("GCOVTOOL", gcovtool.getPathString());
    }

    if (getTargetLibc().startsWith("glibc-")) {
      globalMakeEnvBuilder.put("GLIBC_VERSION",
          getTargetLibc().substring("glibc-".length()));
    } else {
      globalMakeEnvBuilder.put("GLIBC_VERSION", getTargetLibc());
    }

    globalMakeEnvBuilder.put("C_COMPILER", getCompiler());
    globalMakeEnvBuilder.put("TARGET_CPU", getTargetCpu());

    // Deprecated variables

    // TODO(bazel-team): delete all of these.
    globalMakeEnvBuilder.put("CROSSTOOLTOP", crosstoolTopPathFragment.getPathString());

    globalMakeEnvBuilder.putAll(getAdditionalMakeVariables());

    globalMakeEnvBuilder.put("ABI_GLIBC_VERSION", getAbiGlibcVersion());
    globalMakeEnvBuilder.put("ABI", abi);
  }

  @Override
  public void addImplicitLabels(Multimap<String, Label> implicitLabels) {
    if (getLibcLabel() != null) {
      implicitLabels.put("crosstool", getLibcLabel());
    }

    implicitLabels.put("crosstool", crosstoolTop);
  }

  @Override
  public void prepareHook(Path execRoot, ArtifactFactory artifactFactory,
      PackageRootResolver resolver) throws ViewCreationFailedException {
    // TODO(bazel-team): Remove the "relative" guard. sysroot should always be relative, and this
    // should be enforced in the creation of CppConfiguration.
    if (getSysroot() != null && !getSysroot().isAbsolute()) {
      Root sysrootRoot;
      try {
        sysrootRoot = Iterables.getOnlyElement(
          resolver.findPackageRootsForFiles(
              // See doc of findPackageRootsForFiles for why we need a getChild here.
              ImmutableList.of(getSysroot().getChild("dummy_child"))).entrySet()).getValue();
      } catch (PackageRootResolutionException prre) {
        throw new ViewCreationFailedException("Failed to determine sysroot", prre);
      }

      PathFragment sysrootExecPath = sysroot.getRelative(BUILT_IN_INCLUDE_PATH_FRAGMENT);
      if (sysrootRoot.getPath().getRelative(sysrootExecPath).exists()) {
        builtInIncludeFile = Preconditions.checkNotNull(
            artifactFactory.getSourceArtifact(sysrootExecPath, sysrootRoot),
            "%s %s", sysrootRoot, sysroot);
      }
    }
    try {
      getFdoSupport().prepareToBuild(execRoot, artifactFactory, resolver);
    } catch (ZipException e) {
      throw new ViewCreationFailedException("Error reading provided FDO zip file", e);
    } catch (FdoException | IOException | PackageRootResolutionException e) {
      throw new ViewCreationFailedException("Error while initializing FDO support", e);
    }
  }

  @Override
  public void declareSkyframeDependencies(Environment env) {
    getFdoSupport().declareSkyframeDependencies(env, execRoot);
  }

  @Override
  public void addRoots(List<Root> roots) {
    // Fdo root can only exist for the target configuration.
    FdoSupport fdoSupport = getFdoSupport();
    if (fdoSupport.getFdoRoot() != null) {
      roots.add(fdoSupport.getFdoRoot());
    }
  }

  @Override
  public Map<String, String> getCoverageEnvironment() {
    ImmutableMap.Builder<String, String> env = ImmutableMap.builder();
    env.put("COVERAGE_GCOV_PATH", getGcovExecutable().getPathString());
    PathFragment fdoInstrument = getFdoSupport().getFdoInstrument();
    if (fdoInstrument != null) {
      env.put("FDO_DIR", fdoInstrument.getPathString());
    }
    return env.build();
  }

  @Override
  public ImmutableList<Label> getGcovLabels() {
    // TODO(bazel-team): Using a gcov-specific crosstool filegroup here could reduce the number of
    // inputs significantly. We'd also need to add logic in tools/coverage/collect_coverage.sh to
    // drop crosstool dependency if metadataFiles does not contain *.gcno artifacts.
    return ImmutableList.of(crosstoolTop);
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
    return toolchainIdentifier + lipoSuffix;
  }

  @Override
  public String getPlatformName() {
    return getToolchainIdentifier();
  }

  @Override
  public boolean supportsIncrementalBuild() {
    return !isLipoOptimization();
  }

  @Override
  public boolean performsStaticLink() {
    return getLinkOptions().contains("-static");
  }

  /**
   * Returns true if we should share identical native libraries between different targets.
   */
  public boolean shareNativeDeps() {
    return cppOptions.shareNativeDeps;
  }

  @Override
  public void prepareForExecutionPhase() throws IOException {
    // _fdo has a prefix of "_", but it should nevertheless be deleted. Detailed description
    // of the structure of the symlinks / directories can be found at FdoSupport.extractFdoZip().
    // We actually create a directory named "blaze-fdo" under the exec root, the previous version
    // of which is deleted in FdoSupport.prepareToBuildExec(). We cannot do that just before the
    // execution phase because that needs to happen before the analysis phase (in order to create
    // the artifacts corresponding to the .gcda files).
    Path tempPath = execRoot.getRelative("_fdo");
    if (tempPath.exists()) {
      FileSystemUtils.deleteTree(tempPath);
    }
  }

  @Override
  public Map<String, Object> lateBoundOptionDefaults() {
    // --cpu and --compiler initially default to null because their *actual* defaults aren't known
    // until they're read from the CROSSTOOL. Feed the CROSSTOOL defaults in here.
    return ImmutableMap.<String, Object>of(
        "cpu", getTargetCpu(),
        "compiler", getCompiler()
    );
  }

  /**
   * Return set of features enabled by the CppConfiguration, specifically
   * the FDO and LIPO related features enabled by options.
   */
  @Override
  public ImmutableSet<String> configurationEnabledFeatures(RuleContext ruleContext) {
    ImmutableSet.Builder<String> requestedFeatures = ImmutableSet.builder();
    FdoSupport fdoSupport = getFdoSupport();
    if (fdoSupport.getFdoInstrument() != null) {
      requestedFeatures.add(CppRuleClasses.FDO_INSTRUMENT);
    }
    if (fdoSupport.getFdoOptimizeProfile() != null
        && !fdoSupport.isAutoFdoEnabled()) {
      requestedFeatures.add(CppRuleClasses.FDO_OPTIMIZE);
    }
    if (fdoSupport.isAutoFdoEnabled()) {
      requestedFeatures.add(CppRuleClasses.AUTOFDO);
    }
    if (isLipoOptimizationOrInstrumentation()) {
      requestedFeatures.add(CppRuleClasses.LIPO);
    }
    if (ruleContext.getConfiguration().isCodeCoverageEnabled()) {
      requestedFeatures.add(CppRuleClasses.COVERAGE);
    }
    return requestedFeatures.build();
  }
}
