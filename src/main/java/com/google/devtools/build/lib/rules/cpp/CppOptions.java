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

import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.LabelConverter;
import com.google.devtools.build.lib.analysis.config.CompilationMode;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.PerLabelOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration.DynamicMode;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration.StripMode;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.OptionsUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.LipoMode;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.EnumConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * Command-line options for C++.
 */
public class CppOptions extends FragmentOptions {

  /**
   * Converts a comma-separated list of compilation mode settings to a properly typed List.
   */
  public static class FissionOptionConverter implements Converter<List<CompilationMode>> {
    @Override
    public List<CompilationMode> convert(String input) throws OptionsParsingException {
      ImmutableSet.Builder<CompilationMode> modes = ImmutableSet.builder();
      if (input.equals("yes")) { // Special case: enable all modes.
        modes.add(CompilationMode.values());
      } else if (!input.equals("no")) { // "no" is another special case that disables all modes.
        CompilationMode.Converter modeConverter = new CompilationMode.Converter();
        for (String mode : Splitter.on(',').split(input)) {
          modes.add(modeConverter.convert(mode));
        }
      }
      return modes.build().asList();
    }

    @Override
    public String getTypeDescription() {
      return "a set of compilation modes";
    }
  }

  /**
   * Converter for {@link DynamicMode}
   */
  public static class DynamicModeConverter extends EnumConverter<DynamicMode> {
    public DynamicModeConverter() {
      super(DynamicMode.class, "dynamic mode");
    }
  }

  /**
   * Converter for the --strip option.
   */
  public static class StripModeConverter extends EnumConverter<StripMode> {
    public StripModeConverter() {
      super(StripMode.class, "strip mode");
    }
  }

  private static final String LIBC_RELATIVE_LABEL = ":everything";

  /**
   * Converts a String, which is a package label into a label that can be used for a LibcTop object.
   */
  public static class LibcTopLabelConverter implements Converter<Label> {
    @Override
    public Label convert(String input) throws OptionsParsingException {
      if (input.equals("default")) {
        // This is needed for defining config_setting() values, the syntactic form
        // of which must be a String, to match absence of a --grte_top option.
        // "--grte_top=default" works on the command-line too,
        // but that's an inconsequential side-effect, not the intended purpose.
        return null;
      }
      if (!input.startsWith("//")) {
        throw new OptionsParsingException("Not a label");
      }
      try {
        return Label.parseAbsolute(input).getRelative(LIBC_RELATIVE_LABEL);
      } catch (LabelSyntaxException e) {
        throw new OptionsParsingException(e.getMessage());
      }
    }

    @Override
    public String getTypeDescription() {
      return "a label";
    }
  }

  /**
   * Converter for the --lipo option.
   */
  public static class LipoModeConverter extends EnumConverter<LipoMode> {
    public LipoModeConverter() {
      super(LipoMode.class, "LIPO mode");
    }
  }

  @Option(
    name = "crosstool_top",
    defaultValue = "@bazel_tools//tools/cpp:toolchain",
    category = "version",
    converter = LabelConverter.class,
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "The label of the crosstool package to be used for compiling C++ code."
  )
  public Label crosstoolTop;

  @Option(
    name = "compiler",
    defaultValue = "null",
    category = "version",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "The C++ compiler to use for compiling the target."
  )
  public String cppCompiler;

  @Option(
    name = "glibc",
    defaultValue = "null",
    category = "version",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "The version of glibc the target should be linked against. "
            + "By default, a suitable version is chosen based on --cpu."
  )
  public String glibc;

  // O intrepid reaper of unused options: Be warned that the [no]start_end_lib
  // option, however tempting to remove, has a use case. Look in our telemetry data.
  @Option(
    name = "start_end_lib",
    defaultValue = "true",
    category = "strategy", // but also adds edges to the action graph
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "Use the --start-lib/--end-lib ld options if supported by the toolchain."
  )
  public boolean useStartEndLib;

  @Option(
    name = "interface_shared_objects",
    defaultValue = "true",
    category = "strategy", // but also adds edges to the action graph
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "Use interface shared objects if supported by the toolchain. "
            + "All ELF toolchains currently support this setting."
  )
  public boolean useInterfaceSharedObjects;

  @Option(
    name = "fission",
    defaultValue = "no",
    converter = FissionOptionConverter.class,
    category = "semantics",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "Specifies which compilation modes use fission for C++ compilations and links. "
            + " May be any combination of {'fastbuild', 'dbg', 'opt'} or the special values 'yes' "
            + " to enable all modes and 'no' to disable all modes."
  )
  public List<CompilationMode> fissionModes;

  @Option(
    name = "build_test_dwp",
    defaultValue = "false",
    category = "semantics",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "If enabled, when building C++ tests statically and with fission the .dwp file "
            + " for the test binary will be automatically built as well."
  )
  public boolean buildTestDwp;

  @Option(
    name = "dynamic_mode",
    defaultValue = "default",
    converter = DynamicModeConverter.class,
    category = "semantics",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "Determines whether C++ binaries will be linked dynamically.  'default' means "
            + "blaze will choose whether to link dynamically.  'fully' means all libraries "
            + "will be linked dynamically. 'off' means that all libraries will be linked "
            + "in mostly static mode."
  )
  public DynamicMode dynamicMode;

  @Option(
    name = "experimental_link_compile_output_separately",
    defaultValue = "false",
    category = "semantics",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "This flag is experimental and may go away at any time.  "
            + "If true, dynamically linked binary targets will build and link their own srcs as "
            + "a dynamic library instead of directly linking it in."
  )
  public boolean linkCompileOutputSeparately;

  @Option(
    name = "force_pic",
    defaultValue = "false",
    category = "semantics",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "If enabled, all C++ compilations produce position-independent code (\"-fPIC\"),"
            + " links prefer PIC pre-built libraries over non-PIC libraries, and links produce"
            + " position-independent executables (\"-pie\")."
  )
  public boolean forcePic;

  @Option(
    name = "force_ignore_dash_static",
    defaultValue = "false",
    category = "semantics",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "If set, '-static' options in the linkopts of cc_* rules will be ignored."
  )
  public boolean forceIgnoreDashStatic;

  @Option(
    name = "experimental_skip_static_outputs",
    defaultValue = "false",
    category = "semantics",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "This flag is experimental and may go away at any time.  "
            + "If true, linker output for mostly-static C++ executables is a tiny amount of "
            + "dummy dependency information, and NOT a usable binary.  Kludge, but can reduce "
            + "network and disk I/O load (and thus, continuous build cycle times) by a lot.  "
            + "NOTE: use of this flag REQUIRES --distinct_host_configuration."
  )
  public boolean skipStaticOutputs;

  // TODO(djasper): Remove once it has been removed from the global blazerc.
  @Option(
    name = "send_transitive_header_module_srcs",
    defaultValue = "true",
    category = "semantics",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "Obsolete. Don't use."
  )
  public boolean sendTransitiveHeaderModuleSrcs;

  @Option(
    name = "process_headers_in_dependencies",
    defaultValue = "false",
    category = "semantics",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "When building a target //a:a, process headers in all targets that //a:a depends "
            + "on (if header processing is enabled for the toolchain)."
  )
  public boolean processHeadersInDependencies;

  @Option(
    name = "copt",
    allowMultiple = true,
    defaultValue = "",
    category = "flags",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "Additional options to pass to gcc."
  )
  public List<String> coptList;

  @Option(
    name = "cxxopt",
    defaultValue = "",
    category = "flags",
    allowMultiple = true,
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "Additional option to pass to gcc when compiling C++ source files."
  )
  public List<String> cxxoptList;

  @Option(
    name = "conlyopt",
    allowMultiple = true,
    defaultValue = "",
    category = "flags",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "Additional option to pass to gcc when compiling C source files."
  )
  public List<String> conlyoptList;

  @Option(
    name = "linkopt",
    defaultValue = "",
    category = "flags",
    allowMultiple = true,
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "Additional option to pass to gcc when linking."
  )
  public List<String> linkoptList;

  @Option(
    name = "ltoindexopt",
    defaultValue = "",
    category = "flags",
    allowMultiple = true,
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "Additional option to pass to the LTO indexing step (under --features=thin_lto)."
  )
  public List<String> ltoindexoptList;

  @Option(
    name = "stripopt",
    allowMultiple = true,
    defaultValue = "",
    category = "flags",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "Additional options to pass to strip when generating a '<name>.stripped' binary."
  )
  public List<String> stripoptList;

  @Option(
    name = "custom_malloc",
    defaultValue = "null",
    category = "semantics",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "Specifies a custom malloc implementation. This setting overrides malloc "
            + "attributes in build rules.",
    converter = LabelConverter.class
  )
  public Label customMalloc;

  @Option(
    name = "legacy_whole_archive",
    defaultValue = "true",
    category = "semantics",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "When on, use --whole-archive for cc_binary rules that have "
            + "linkshared=1 and either linkstatic=1 or '-static' in linkopts. "
            + "This is for backwards compatibility only. "
            + "A better alternative is to use alwayslink=1 where required."
  )
  public boolean legacyWholeArchive;

  @Option(
    name = "strip",
    defaultValue = "sometimes",
    category = "flags",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "Specifies whether to strip binaries and shared libraries "
            + " (using \"-Wl,--strip-debug\").  The default value of 'sometimes'"
            + " means strip iff --compilation_mode=fastbuild.",
    converter = StripModeConverter.class
  )
  public StripMode stripBinaries;

  @Option(
    name = "fdo_instrument",
    defaultValue = "null",
    converter = OptionsUtils.PathFragmentConverter.class,
    category = "flags",
    implicitRequirements = {"--copt=-Wno-error"},
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "Generate binaries with FDO instrumentation. Specify the relative "
            + "directory name for the .gcda files at runtime with GCC compiler. "
            + "With Clang/LLVM compiler, it also accepts the directory name under"
            + "which the raw profile file(s) will be dumped at runtime."
  )
  /**
   * Never read FDO/LIPO options directly. This is because {@link #lipoConfigurationState}
   * determines whether these options are actually "active" for this configuration. Instead, use the
   * equivalent getter method, which takes that into account.
   */
  public PathFragment fdoInstrumentForBuild;

  /**
   * Returns the --fdo_instrument value if FDO is specified and active for this configuration,
   * the default value otherwise.
   */
  public PathFragment getFdoInstrument() {
    return enableLipoSettings() ? fdoInstrumentForBuild : null;
  }

  @Option(
    name = "fdo_optimize",
    defaultValue = "null",
    category = "flags",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "Use FDO profile information to optimize compilation. Specify the name "
            + "of the zip file containing the .gcda file tree or an afdo file containing "
            + "an auto profile. This flag also accepts files specified as labels, for "
            + "example //foo/bar:file.afdo. Such labels must refer to input files; you may "
            + "need to add an exports_files directive to the corresponding package to make "
            + "the file visible to Blaze. It also accepts a raw or an indexed LLVM profile file."
  )
  /**
   * Never read FDO/LIPO options directly. This is because {@link #lipoConfigurationState}
   * determines whether these options are actually "active" for this configuration. Instead, use the
   * equivalent getter method, which takes that into account.
   */
  public String fdoOptimizeForBuild;

  /**
   * Returns the --fdo_optimize value if FDO is specified and active for this configuration,
   * the default value otherwise.
   */
  public String getFdoOptimize() {
    return enableLipoSettings() ? fdoOptimizeForBuild : null;
  }

  @Option(
    name = "autofdo_lipo_data",
    defaultValue = "false",
    category = "flags",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "If true then the directory name for non-LIPO targets will have a "
            + "'-lipodata' suffix in AutoFDO mode."
  )
  /**
   * Never read FDO/LIPO options directly. This is because {@link #lipoConfigurationState}
   * determines whether these options are actually "active" for this configuration. Instead, use the
   * equivalent getter method, which takes that into account.
   */
  public boolean autoFdoLipoDataForBuild;

  /**
   * Returns the --autofdo_lipo_data value for this configuration. This is false except for data
   * configurations under LIPO builds.
   */
  public boolean getAutoFdoLipoData() {
    return enableLipoSettings()
        ? autoFdoLipoDataForBuild
        : lipoModeForBuild != LipoMode.OFF && fdoOptimizeForBuild != null && FdoSupport.isAutoFdo(
            fdoOptimizeForBuild);
  }

  @Option(
    name = "lipo",
    defaultValue = "off",
    converter = LipoModeConverter.class,
    category = "flags",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "Enable LIPO optimization (lightweight inter-procedural optimization, The allowed "
            + "values for  this option are 'off' and 'binary', which enables LIPO. This option "
            + "only has an effect when FDO is also enabled. Currently LIPO is only supported "
            + "when building a single cc_binary rule."
  )
  /**
   * Never read FDO/LIPO options directly. This is because {@link #lipoConfigurationState}
   * determines whether these options are actually "active" for this configuration. Instead, use the
   * equivalent getter method, which takes that into account.
   */
  public LipoMode lipoModeForBuild;

  /**
   * Returns the --lipo value if LIPO is specified and active for this configuration,
   * the default value otherwise.
   */
  public LipoMode getLipoMode() {
    return enableLipoSettings() ? lipoModeForBuild : LipoMode.OFF;
  }

  @Option(
    name = "lipo_context",
    defaultValue = "null",
    category = "flags",
    converter = LabelConverter.class,
    implicitRequirements = {"--linkopt=-Wl,--warn-unresolved-symbols"},
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "Specifies the binary from which the LIPO profile information comes."
  )
  /**
   * Never read FDO/LIPO options directly. This is because {@link #lipoConfigurationState}
   * determines whether these options are actually "active" for this configuration. Instead, use the
   * equivalent getter method, which takes that into account.
   */
  public Label lipoContextForBuild;

  @Option(
    name = "experimental_toolchain_id_in_output_directory",
    defaultValue = "true",
    category = "semantics",
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    effectTags = OptionEffectTag.AFFECTS_OUTPUTS,
    help = "Whether to embed the name of the C++ toolchain in the name of the output directory"
  )
  public boolean toolchainIdInOutputDirectory;

  /**
   * Returns the --lipo_context value if LIPO is specified and active for this configuration,
   * null otherwise.
   */
  @Nullable
  public Label getLipoContext() {
    return isLipoOptimization() ? lipoContextForBuild : null;
  }

  /**
   * Returns the LIPO context for this build, even if LIPO isn't enabled in the current
   * configuration.
   */
  public Label getLipoContextForBuild() {
    return lipoContextForBuild;
  }

  /**
   * Internal state determining how to apply LIPO settings under this configuration.
   */
  public enum LipoConfigurationState {
    /** Don't LIPO-optimize targets under this configuration. */
    IGNORE_LIPO,
    /** LIPO-optimize targets under this configuration if this is a LIPO build. */
    APPLY_LIPO,
    /**
     * Evaluate targets in this configuration in "LIPO context collector" mode. See
     * {@link FdoSupport} for details.
      */
    LIPO_CONTEXT_COLLECTOR,
  }

  /**
   * Converter for {@link LipoConfigurationState}.
   */
  public static class LipoConfigurationStateConverter
      extends EnumConverter<LipoConfigurationState> {
    public LipoConfigurationStateConverter() {
      super(LipoConfigurationState.class, "LIPO configuration state");
    }
  }

  @Option(
    name = "lipo configuration state",
    defaultValue = "apply_lipo",
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    effectTags = {OptionEffectTag.UNKNOWN},
    metadataTags = {OptionMetadataTag.INTERNAL},
    converter = LipoConfigurationStateConverter.class
  )
  public LipoConfigurationState lipoConfigurationState;

  /**
   * Returns true if targets under this configuration should use the build's LIPO settings.
   *
   * <p>Even when we switch off LIPO (e.g. by switching to a data configuration), we still need to
   * remember the LIPO settings in case we need to switch back (e.g. if we build a genrule with a
   * data dependency on the LIPO context).
   *
   * <p>We achieve this by maintaining a "configuration state" flag that flips on / off when we
   * want to enable / disable LIPO respectively. This means we need to be careful to distinguish
   * between the existence of LIPO settings and LIPO actually applying to the configuration. So when
   * buiding a target, it's not enough to see if {@link #lipoContextForBuild} or
   * {@link #lipoModeForBuild} are set. We also need to check this flag.
   *
   * <p>This class exposes appropriate convenience methods to make these checks convenient and easy.
   * Use them and read the documentation carefully.
   */
  private boolean enableLipoSettings() {
    return lipoConfigurationState != LipoConfigurationState.IGNORE_LIPO;
  }

  @Option(
    name = "experimental_stl",
    converter = LabelConverter.class,
    defaultValue = "null",
    category = "version",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "If set, use this label instead of the default STL implementation. "
            + "This option is EXPERIMENTAL and may go away in a future release."
  )
  public Label stl;

  @Option(
    name = "save_temps",
    defaultValue = "false",
    category = "what",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "If set, temporary outputs from gcc will be saved.  "
            + "These include .s files (assembler code), .i files (preprocessed C) and "
            + ".ii files (preprocessed C++)."
  )
  public boolean saveTemps;

  @Option(
    name = "per_file_copt",
    allowMultiple = true,
    converter = PerLabelOptions.PerLabelOptionsConverter.class,
    defaultValue = "",
    category = "semantics",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "Additional options to selectively pass to gcc when compiling certain files. "
            + "This option can be passed multiple times. "
            + "Syntax: regex_filter@option_1,option_2,...,option_n. Where regex_filter stands "
            + "for a list of include and exclude regular expression patterns (Also see "
            + "--instrumentation_filter). option_1 to option_n stand for "
            + "arbitrary command line options. If an option contains a comma it has to be "
            + "quoted with a backslash. Options can contain @. Only the first @ is used to "
            + "split the string. Example: "
            + "--per_file_copt=//foo/.*\\.cc,-//foo/bar\\.cc@-O0 adds the -O0 "
            + "command line option to the gcc command line of all cc files in //foo/ "
            + "except bar.cc."
  )
  public List<PerLabelOptions> perFileCopts;

  @Option(
    name = "host_crosstool_top",
    defaultValue = "null",
    converter = LabelConverter.class,
    category = "semantics",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "By default, the --crosstool_top and --compiler options are also used "
            + "for the host configuration. If this flag is provided, Blaze uses the default libc "
            + "and compiler for the given crosstool_top."
  )
  public Label hostCrosstoolTop;

  @Option(
    name = "host_copt",
    allowMultiple = true,
    defaultValue = "",
    category = "flags",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "Additional options to pass to gcc for host tools."
  )
  public List<String> hostCoptList;

  @Option(
    name = "host_cxxopt",
    allowMultiple = true,
    defaultValue = "",
    category = "flags",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "Additional options to pass to gcc for host tools."
  )
  public List<String> hostCxxoptList;

  @Option(
    name = "grte_top",
    defaultValue = "null", // The default value is chosen by the toolchain.
    category = "version",
    converter = LibcTopLabelConverter.class,
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "A label to a checked-in libc library. The default value is selected by the crosstool "
            + "toolchain, and you almost never need to override it."
  )
  public Label libcTopLabel;

  @Option(
    name = "host_grte_top",
    defaultValue = "null", // The default value is chosen by the toolchain.
    category = "version",
    converter = LibcTopLabelConverter.class,
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "If specified, this setting overrides the libc top-level directory (--grte_top) "
            + "for the host configuration."
  )
  public Label hostLibcTopLabel;

  @Option(
    name = "output_symbol_counts",
    defaultValue = "false",
    category = "flags",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "If enabled, for every C++ binary linked with gold, the number of defined symbols "
            + "and the number of used symbols per input file is stored in a .sc file."
  )
  public boolean symbolCounts;

  @Option(
    name = "experimental_inmemory_dotd_files",
    defaultValue = "false",
    category = "experimental",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "If enabled, C++ .d files will be passed through in memory directly from the remote "
            + "build nodes instead of being written to disk."
  )
  public boolean inmemoryDotdFiles;

  @Option(
    name = "experimental_skip_unused_modules",
    defaultValue = "false",
    category = "experimental",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "Deprecated. No effect."
  )
  public boolean skipUnusedModules;

  @Option(
    name = "experimental_prune_more_modules",
    defaultValue = "false",
    category = "experimental",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "Deprecated. No effect."
  )
  public boolean pruneMoreModules;

  @Option(
    name = "prune_cpp_modules",
    defaultValue = "true",
    category = "strategy",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "If enabled, use the results of input discovery to reduce the number of used modules."
  )
  public boolean pruneCppModules;

  @Option(
    name = "parse_headers_verifies_modules",
    defaultValue = "false",
    category = "strategy",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "If enabled, the parse_headers feature verifies that a header module can be built for the "
            + "target in question instead of doing a separate compile of the header."
  )
  public boolean parseHeadersVerifiesModules;

  @Option(
    name = "experimental_omitfp",
    defaultValue = "false",
    category = "semantics",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "If true, use libunwind for stack unwinding, and compile with "
            + "-fomit-frame-pointer and -fasynchronous-unwind-tables."
  )
  public boolean experimentalOmitfp;

  @Option(
    name = "share_native_deps",
    defaultValue = "true",
    category = "strategy",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "If true, native libraries that contain identical functionality "
            + "will be shared among different targets"
  )
  public boolean shareNativeDeps;

  @Option(
    name = "strict_system_includes",
    defaultValue = "false",
    category = "strategy",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "If true, headers found through system include paths (-isystem) are also required to be "
            + "declared."
  )
  public boolean strictSystemIncludes;

  @Override
  public FragmentOptions getHost(boolean fallback) {
    CppOptions host = (CppOptions) getDefault();

    // The crosstool options are partially copied from the target configuration.
    if (!fallback) {
      if (hostCrosstoolTop == null) {
        host.cppCompiler = cppCompiler;
        host.crosstoolTop = crosstoolTop;
        host.glibc = glibc;
      } else {
        host.crosstoolTop = hostCrosstoolTop;
      }
    }

    // the name of the output directory for the host configuration is forced to be "host" in
    // BuildConfiguration.Options#getHost(), but let's be prudent here in case someone ends up
    // removing that
    host.toolchainIdInOutputDirectory = toolchainIdInOutputDirectory;

    // hostLibcTop doesn't default to the target's libcTop.
    // Only an explicit command-line option will change it.
    // The default is whatever the host's crosstool (which might have been specified
    // by --host_crosstool_top, or --crosstool_top as a fallback) says it should be.
    host.libcTopLabel = hostLibcTopLabel;

    // -g0 is the default, but allowMultiple options cannot have default values so we just pass
    // -g0 first and let the user options override it.
    ImmutableList.Builder<String> coptListBuilder = ImmutableList.builder();
    ImmutableList.Builder<String> cxxoptListBuilder = ImmutableList.builder();
    // Don't add -g0 if the host platform is Windows.
    // Note that host platform is not necessarily the platform bazel is running on (foundry)
    if (OS.getCurrent() != OS.WINDOWS) {
      coptListBuilder.add("-g0");
      cxxoptListBuilder.add("-g0");
    }
    host.coptList = coptListBuilder.addAll(hostCoptList).build();
    host.cxxoptList = cxxoptListBuilder.addAll(hostCxxoptList).build();

    host.useStartEndLib = useStartEndLib;
    host.stripBinaries = StripMode.ALWAYS;
    host.fdoOptimizeForBuild = null;
    host.lipoModeForBuild = LipoMode.OFF;
    host.inmemoryDotdFiles = inmemoryDotdFiles;

    return host;
  }

  @Override
  public Map<String, Set<Label>> getDefaultsLabels(BuildConfiguration.Options commonOptions) {
    Set<Label> crosstoolLabels = new LinkedHashSet<>();
    crosstoolLabels.add(crosstoolTop);
    if (hostCrosstoolTop != null) {
      crosstoolLabels.add(hostCrosstoolTop);
    }

    if (libcTopLabel != null) {
      Label libcLabel = libcTopLabel;
      if (libcLabel != null) {
        crosstoolLabels.add(libcLabel);
      }
    }

    return ImmutableMap.of("CROSSTOOL", crosstoolLabels, "COVERAGE", ImmutableSet.<Label>of());
  }

  /**
   * Returns true if targets under this configuration should apply FDO.
   */
  public boolean isFdo() {
    return getFdoOptimize() != null || getFdoInstrument() != null;
  }

  /**
   * Returns true if this configuration has LIPO optimization settings (even if they're
   * not necessarily active).
   */
  private boolean hasLipoOptimizationState() {
    return lipoModeForBuild == LipoMode.BINARY && fdoOptimizeForBuild != null
        && lipoContextForBuild != null;
  }

  /**
   * Returns true if targets under this configuration should LIPO-optimize.
   */
  public boolean isLipoOptimization() {
    return hasLipoOptimizationState() && enableLipoSettings() && !isLipoContextCollector();
  }

  /**
   * Returns true if this is a data configuration for a LIPO-optimizing build.
   *
   * <p>This means LIPO is not applied for this configuration, but LIPO might be reenabled further
   * down the dependency tree.
   */
  public boolean isDataConfigurationForLipoOptimization() {
    return hasLipoOptimizationState() && !enableLipoSettings();
  }

  /**
   * Returns true if targets under this configuration should LIPO-optimize or LIPO-instrument.
   */
  public boolean isLipoOptimizationOrInstrumentation() {
    return getLipoMode() == LipoMode.BINARY
        && ((getFdoOptimize() != null && getLipoContext() != null) || getFdoInstrument() != null)
        && !isLipoContextCollector();
  }

  /**
   * Returns true if this is the LIPO context collector configuration.
   */
  public boolean isLipoContextCollector() {
    return lipoConfigurationState == LipoConfigurationState.LIPO_CONTEXT_COLLECTOR;
  }

  @Override
  public boolean enableActions() {
    return !isLipoContextCollector();
  }
}
