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
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.LabelConverter;
import com.google.devtools.build.lib.analysis.config.CompilationMode;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.PerLabelOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration.DynamicMode;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration.StripMode;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.EnumConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;
import java.util.Set;

/** Command-line options for C++. */
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
        return Label.parseAbsolute(input, ImmutableMap.of())
            .getRelativeWithRemapping(LIBC_RELATIVE_LABEL, ImmutableMap.of());
      } catch (LabelSyntaxException e) {
        throw new OptionsParsingException(e.getMessage());
      }
    }

    @Override
    public String getTypeDescription() {
      return "a label";
    }
  }

  @Option(
    name = "crosstool_top",
    defaultValue = "@bazel_tools//tools/cpp:toolchain",
    converter = LabelConverter.class,
    documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
    effectTags = {
      OptionEffectTag.LOADING_AND_ANALYSIS,
      OptionEffectTag.CHANGES_INPUTS,
      OptionEffectTag.AFFECTS_OUTPUTS
    },
    help = "The label of the crosstool package to be used for compiling C++ code."
  )
  public Label crosstoolTop;

  @Option(
    name = "compiler",
    defaultValue = "null",
    documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
    effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS, OptionEffectTag.EXECUTION},
    help = "The C++ compiler to use for compiling the target."
  )
  public String cppCompiler;

  // This is different from --platform_suffix in that that one is designed to facilitate the
  // migration to toolchains and this one is designed to eliminate the C++ toolchain identifier
  // from the output directory path.
  @Option(
    name = "cc_output_directory_tag",
    defaultValue = "",
    documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
    effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
    help = "Specifies a suffix to be added to the configuration directory."
  )
  public String outputDirectoryTag;

  @Option(
      name = "minimum_os_version",
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS, OptionEffectTag.AFFECTS_OUTPUTS},
      help = "The minimum OS version which your compilation targets.")
  public String minimumOsVersion;

  // O intrepid reaper of unused options: Be warned that the [no]start_end_lib
  // option, however tempting to remove, has a use case. Look in our telemetry data.
  @Option(
    name = "start_end_lib",
    defaultValue = "true",
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    effectTags = {
      OptionEffectTag.LOADING_AND_ANALYSIS,
      OptionEffectTag.CHANGES_INPUTS,
      OptionEffectTag.AFFECTS_OUTPUTS
    },
    metadataTags = {OptionMetadataTag.HIDDEN},
    help = "Use the --start-lib/--end-lib ld options if supported by the toolchain."
  )
  public boolean useStartEndLib;

  @Option(
    name = "interface_shared_objects",
    defaultValue = "true",
    documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
    effectTags = {
      OptionEffectTag.LOADING_AND_ANALYSIS,
      OptionEffectTag.AFFECTS_OUTPUTS,
      OptionEffectTag.AFFECTS_OUTPUTS
    },
    help =
        "Use interface shared objects if supported by the toolchain. "
            + "All ELF toolchains currently support this setting."
  )
  public boolean useInterfaceSharedObjects;

  @Option(
    name = "fission",
    defaultValue = "no",
    converter = FissionOptionConverter.class,
    documentationCategory = OptionDocumentationCategory.OUTPUT_SELECTION,
    effectTags = {
      OptionEffectTag.LOADING_AND_ANALYSIS,
      OptionEffectTag.ACTION_COMMAND_LINES,
      OptionEffectTag.AFFECTS_OUTPUTS
    },
    help =
        "Specifies which compilation modes use fission for C++ compilations and links. "
            + " May be any combination of {'fastbuild', 'dbg', 'opt'} or the special values 'yes' "
            + " to enable all modes and 'no' to disable all modes."
  )
  public List<CompilationMode> fissionModes;

  @Option(
    name = "build_test_dwp",
    defaultValue = "false",
    documentationCategory = OptionDocumentationCategory.OUTPUT_SELECTION,
    effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS, OptionEffectTag.AFFECTS_OUTPUTS},
    help =
        "If enabled, when building C++ tests statically and with fission the .dwp file "
            + " for the test binary will be automatically built as well."
  )
  public boolean buildTestDwp;

  @Option(
    name = "dynamic_mode",
    defaultValue = "default",
    converter = DynamicModeConverter.class,
    documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
    effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS, OptionEffectTag.AFFECTS_OUTPUTS},
    help =
        "Determines whether C++ binaries will be linked dynamically.  'default' means "
            + "Bazel will choose whether to link dynamically.  'fully' means all libraries "
            + "will be linked dynamically. 'off' means that all libraries will be linked "
            + "in mostly static mode."
  )
  public DynamicMode dynamicMode;

  @Option(
    name = "experimental_drop_fully_static_linking_mode",
    defaultValue = "false",
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS, OptionEffectTag.AFFECTS_OUTPUTS},
    help =
        "If enabled, Bazel will not scan linkopts for -static. Rules have to define their fully"
            + " static linking mode through 'fully_static_link' feature."
  )
  public boolean dropFullyStaticLinkingMode;

  @Option(
    name = "experimental_link_compile_output_separately",
    defaultValue = "false",
    documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
    effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS, OptionEffectTag.AFFECTS_OUTPUTS},
    metadataTags = {OptionMetadataTag.EXPERIMENTAL},
    help =
        "This flag is experimental and may go away at any time.  "
            + "If true, dynamically linked binary targets will build and link their own srcs as "
            + "a dynamic library instead of directly linking it in."
  )
  public boolean linkCompileOutputSeparately;

  @Option(
    name = "force_pic",
    defaultValue = "false",
    documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
    effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS, OptionEffectTag.AFFECTS_OUTPUTS},
    help =
        "If enabled, all C++ compilations produce position-independent code (\"-fPIC\"),"
            + " links prefer PIC pre-built libraries over non-PIC libraries, and links produce"
            + " position-independent executables (\"-pie\")."
  )
  public boolean forcePic;

  @Option(
    name = "force_ignore_dash_static",
    defaultValue = "false",
    documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
    effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS, OptionEffectTag.AFFECTS_OUTPUTS},
    help = "If set, '-static' options in the linkopts of cc_* rules will be ignored."
  )
  public boolean forceIgnoreDashStatic;

  @Option(
    name = "process_headers_in_dependencies",
    defaultValue = "false",
    documentationCategory = OptionDocumentationCategory.BUILD_TIME_OPTIMIZATION,
    effectTags = {OptionEffectTag.EXECUTION},
    help =
        "When building a target //a:a, process headers in all targets that //a:a depends "
            + "on (if header processing is enabled for the toolchain)."
  )
  public boolean processHeadersInDependencies;

  @Option(
    name = "copt",
    allowMultiple = true,
    defaultValue = "",
    documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
    effectTags = {OptionEffectTag.ACTION_COMMAND_LINES, OptionEffectTag.AFFECTS_OUTPUTS},
    help = "Additional options to pass to gcc."
  )
  public List<String> coptList;

  @Option(
    name = "cxxopt",
    defaultValue = "",
    allowMultiple = true,
    documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
    effectTags = {OptionEffectTag.ACTION_COMMAND_LINES, OptionEffectTag.AFFECTS_OUTPUTS},
    help = "Additional option to pass to gcc when compiling C++ source files."
  )
  public List<String> cxxoptList;

  @Option(
    name = "conlyopt",
    allowMultiple = true,
    defaultValue = "",
    documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
    effectTags = {OptionEffectTag.ACTION_COMMAND_LINES, OptionEffectTag.AFFECTS_OUTPUTS},
    help = "Additional option to pass to gcc when compiling C source files."
  )
  public List<String> conlyoptList;

  @Option(
    name = "linkopt",
    defaultValue = "",
    allowMultiple = true,
    documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
    effectTags = {OptionEffectTag.ACTION_COMMAND_LINES, OptionEffectTag.AFFECTS_OUTPUTS},
    help = "Additional option to pass to gcc when linking."
  )
  public List<String> linkoptList;

  @Option(
    name = "ltoindexopt",
    defaultValue = "",
    allowMultiple = true,
    documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
    effectTags = {OptionEffectTag.ACTION_COMMAND_LINES, OptionEffectTag.AFFECTS_OUTPUTS},
    help = "Additional option to pass to the LTO indexing step (under --features=thin_lto)."
  )
  public List<String> ltoindexoptList;

  @Option(
    name = "ltobackendopt",
    defaultValue = "",
    allowMultiple = true,
    documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
    effectTags = {OptionEffectTag.ACTION_COMMAND_LINES, OptionEffectTag.AFFECTS_OUTPUTS},
    help = "Additional option to pass to the LTO backend step (under --features=thin_lto)."
  )
  public List<String> ltobackendoptList;

  @Option(
    name = "stripopt",
    allowMultiple = true,
    defaultValue = "",
    documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
    effectTags = {OptionEffectTag.ACTION_COMMAND_LINES, OptionEffectTag.AFFECTS_OUTPUTS},
    help = "Additional options to pass to strip when generating a '<name>.stripped' binary."
  )
  public List<String> stripoptList;

  @Option(
    name = "custom_malloc",
    defaultValue = "null",
    documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
    effectTags = {OptionEffectTag.CHANGES_INPUTS, OptionEffectTag.AFFECTS_OUTPUTS},
    help =
        "Specifies a custom malloc implementation. This setting overrides malloc "
            + "attributes in build rules.",
    converter = LabelConverter.class
  )
  public Label customMalloc;

  @Option(
      name = "experimental_shortened_obj_file_path",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.ACTION_COMMAND_LINES, OptionEffectTag.AFFECTS_OUTPUTS},
      help =
          "When off, object files are generated at _objs/<target_name>/<source_package_path>/"
              + "<source_base_name>.o, otherwise they are shortened to _objs/<target_name>/"
              + "<source_base_name>.o. If there are multiple source files with the same base name, "
              + "to avoid conflict, the object file path is _objs/<target_name>/<N>"
              + "/<source_base_name>.o, where N = the source file's order among all source files "
              + "with the same base name, N starts with 0.")
  public boolean shortenObjFilePath;

  @Option(
    name = "legacy_whole_archive",
    defaultValue = "true",
    documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
    effectTags = {OptionEffectTag.ACTION_COMMAND_LINES, OptionEffectTag.AFFECTS_OUTPUTS},
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
    documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
    effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
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
    implicitRequirements = {"--copt=-Wno-error"},
    documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
    effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
    help =
        "Generate binaries with FDO instrumentation. Specify the relative "
            + "directory name for the .gcda files at runtime with GCC compiler. "
            + "With Clang/LLVM compiler, it also accepts the directory name under"
            + "which the raw profile file(s) will be dumped at runtime."
  )
  public String fdoInstrumentForBuild;

  @Option(
    name = "fdo_optimize",
    allowMultiple = true,
    defaultValue = "null",
    documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
    effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
    help =
        "Use FDO profile information to optimize compilation. Specify the name "
            + "of the zip file containing the .gcda file tree, an afdo file containing "
            + "an auto profile or an xfdo file containing a default cross binary profile. "
            + "If the multiple profiles passed through the option include xfdo file and "
            + "other types of profiles, the last profile other than xfdo file will prevail. "
            + "If the multiple profiles include only xfdo files or don't include any xfdo file, "
            + "the last profile will prevail. This flag also accepts files specified as labels, "
            + "for example //foo/bar:file.afdo. Such labels must refer to input files; you may "
            + "need to add an exports_files directive to the corresponding package to make "
            + "the file visible to Bazel. It also accepts a raw or an indexed LLVM profile file. "
            + "This flag will be superseded by fdo_profile rule."
  )
  public List<String> fdoProfiles;

  /**
   * Returns the --fdo_optimize value if FDO is specified and active for this configuration, the
   * default value otherwise.
   */
  public String getFdoOptimize() {
    if (fdoProfiles == null) {
      return null;
    }

    // Return the last profile in the list that is not a crossbinary profile.
    String lastXBinaryProfile = null;
    ListIterator<String> iter = fdoProfiles.listIterator(fdoProfiles.size());
    while (iter.hasPrevious()) {
      String profile = iter.previous();
      if (CppFileTypes.XBINARY_PROFILE.matches(profile)) {
        lastXBinaryProfile = profile;
        continue;
      }
      return profile;
    }

    // If crossbinary profile is the only kind of profile in the list, return the last one.
    return lastXBinaryProfile;
  }

  @Option(
    name = "fdo_prefetch_hints",
    defaultValue = "null",
    converter = LabelConverter.class,
    category = "flags",
    documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
    effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
    help = "Use cache prefetch hints."
  )
  public Label fdoPrefetchHintsLabel;

  /**
   * Returns the --fdo_prefetch_hints value.
   */
  public Label getFdoPrefetchHintsLabel() {
    return fdoPrefetchHintsLabel;
  }

  @Option(
    name = "fdo_profile",
    defaultValue = "null",
    category = "flags",
    converter = LabelConverter.class,
    documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
    effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
    help = "The fdo_profile representing the profile to be used for optimization."
  )
  public Label fdoProfileLabel;

  @Option(
      name = "enable_fdo_profile_absolute_path",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      help = "If set, use of fdo_absolute_profile_path will raise an error.")
  public boolean enableFdoProfileAbsolutePath;

  @Option(
    name = "experimental_stl",
    converter = LabelConverter.class,
    defaultValue = "null",
    documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
    effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.ACTION_COMMAND_LINES},
    metadataTags = {OptionMetadataTag.EXPERIMENTAL},
    help =
        "If set, use this label instead of the default STL implementation. "
            + "This option is EXPERIMENTAL and may go away in a future release."
  )
  public Label stl;

  @Option(
    name = "save_temps",
    defaultValue = "false",
    documentationCategory = OptionDocumentationCategory.OUTPUT_SELECTION,
    effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
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
    documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
    effectTags = {OptionEffectTag.ACTION_COMMAND_LINES, OptionEffectTag.AFFECTS_OUTPUTS},
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
    name = "per_file_ltobackendopt",
    allowMultiple = true,
    converter = PerLabelOptions.PerLabelOptionsConverter.class,
    defaultValue = "",
    documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
    effectTags = {OptionEffectTag.ACTION_COMMAND_LINES, OptionEffectTag.AFFECTS_OUTPUTS},
    help =
        "Additional options to selectively pass to LTO backend (under --features=thin_lto) when "
            + "compiling certain backend objects. This option can be passed multiple times. "
            + "Syntax: regex_filter@option_1,option_2,...,option_n. Where regex_filter stands "
            + "for a list of include and exclude regular expression patterns. "
            + "option_1 to option_n stand for arbitrary command line options. "
            + "If an option contains a comma it has to be quoted with a backslash. "
            + "Options can contain @. Only the first @ is used to split the string. Example: "
            + "--per_file_ltobackendopt=//foo/.*\\.o,-//foo/bar\\.o@-O0 adds the -O0 "
            + "command line option to the LTO backend command line of all o files in //foo/ "
            + "except bar.o."
  )
  public List<PerLabelOptions> perFileLtoBackendOpts;

  @Option(
    name = "host_crosstool_top",
    defaultValue = "null",
    converter = LabelConverter.class,
    documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
    effectTags = {
      OptionEffectTag.LOADING_AND_ANALYSIS,
      OptionEffectTag.CHANGES_INPUTS,
      OptionEffectTag.AFFECTS_OUTPUTS
    },
    help =
        "By default, the --crosstool_top and --compiler options are also used "
            + "for the host configuration. If this flag is provided, Bazel uses the default libc "
            + "and compiler for the given crosstool_top."
  )
  public Label hostCrosstoolTop;

  @Option(
    name = "host_copt",
    allowMultiple = true,
    defaultValue = "",
    documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
    effectTags = {OptionEffectTag.ACTION_COMMAND_LINES, OptionEffectTag.AFFECTS_OUTPUTS},
    help = "Additional options to pass to gcc for host tools."
  )
  public List<String> hostCoptList;

  @Option(
    name = "host_cxxopt",
    allowMultiple = true,
    defaultValue = "",
    documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
    effectTags = {OptionEffectTag.ACTION_COMMAND_LINES, OptionEffectTag.AFFECTS_OUTPUTS},
    help = "Additional options to pass to gcc for host tools."
  )
  public List<String> hostCxxoptList;

  @Option(
    name = "host_conlyopt",
    allowMultiple = true,
    defaultValue = "",
    documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
    effectTags = {OptionEffectTag.ACTION_COMMAND_LINES, OptionEffectTag.AFFECTS_OUTPUTS},
    help = "Additional option to pass to gcc when compiling C source files for host tools."
  )
  public List<String> hostConlyoptList;

  @Option(
    name = "host_linkopt",
    defaultValue = "",
    allowMultiple = true,
    documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
    effectTags = {OptionEffectTag.ACTION_COMMAND_LINES, OptionEffectTag.AFFECTS_OUTPUTS},
    help = "Additional option to pass to gcc when linking host tools."
  )
  public List<String> hostLinkoptList;

  @Option(
    name = "grte_top",
    defaultValue = "null", // The default value is chosen by the toolchain.
    converter = LibcTopLabelConverter.class,
    documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
    effectTags = {OptionEffectTag.ACTION_COMMAND_LINES, OptionEffectTag.AFFECTS_OUTPUTS},
    help =
        "A label to a checked-in libc library. The default value is selected by the crosstool "
            + "toolchain, and you almost never need to override it."
  )
  public Label libcTopLabel;

  @Option(
    name = "host_grte_top",
    defaultValue = "null", // The default value is chosen by the toolchain.
    converter = LibcTopLabelConverter.class,
    documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
    effectTags = {OptionEffectTag.ACTION_COMMAND_LINES, OptionEffectTag.AFFECTS_OUTPUTS},
    help =
        "If specified, this setting overrides the libc top-level directory (--grte_top) "
            + "for the host configuration."
  )
  public Label hostLibcTopLabel;

  @Option(
    name = "output_symbol_counts",
    defaultValue = "false",
    documentationCategory = OptionDocumentationCategory.OUTPUT_SELECTION,
    effectTags = {OptionEffectTag.ACTION_COMMAND_LINES, OptionEffectTag.AFFECTS_OUTPUTS},
    help =
        "If enabled, for every C++ binary linked with gold, the number of defined symbols "
            + "and the number of used symbols per input file is stored in a .sc file."
  )
  public boolean symbolCounts;

  @Option(
    name = "experimental_inmemory_dotd_files",
    defaultValue = "false",
    documentationCategory = OptionDocumentationCategory.BUILD_TIME_OPTIMIZATION,
    effectTags = {
      OptionEffectTag.LOADING_AND_ANALYSIS,
      OptionEffectTag.EXECUTION,
      OptionEffectTag.AFFECTS_OUTPUTS
    },
    metadataTags = {OptionMetadataTag.EXPERIMENTAL},
    help =
        "If enabled, C++ .d files will be passed through in memory directly from the remote "
            + "build nodes instead of being written to disk."
  )
  public boolean inmemoryDotdFiles;

  @Option(
    name = "prune_cpp_modules",
    defaultValue = "true",
    documentationCategory = OptionDocumentationCategory.BUILD_TIME_OPTIMIZATION,
    effectTags = {
      OptionEffectTag.LOADING_AND_ANALYSIS,
      OptionEffectTag.EXECUTION,
      OptionEffectTag.CHANGES_INPUTS
    },
    help = "If enabled, use the results of input discovery to reduce the number of used modules."
  )
  public boolean pruneCppModules;

  @Option(
    name = "experimental_prune_cpp_input_discovery",
    defaultValue = "false",
    documentationCategory = OptionDocumentationCategory.BUILD_TIME_OPTIMIZATION,
    effectTags = {OptionEffectTag.EXECUTION, OptionEffectTag.CHANGES_INPUTS},
    help = "If enabled, stop C++ input discovery at modular headers."
  )
  public boolean pruneCppInputDiscovery;


  @Option(
    name = "parse_headers_verifies_modules",
    defaultValue = "false",
    documentationCategory = OptionDocumentationCategory.BUILD_TIME_OPTIMIZATION,
    effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS, OptionEffectTag.CHANGES_INPUTS},
    help =
        "If enabled, the parse_headers feature verifies that a header module can be built for the "
            + "target in question instead of doing a separate compile of the header."
  )
  public boolean parseHeadersVerifiesModules;

  @Option(
    name = "experimental_omitfp",
    defaultValue = "false",
    documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
    effectTags = {OptionEffectTag.ACTION_COMMAND_LINES, OptionEffectTag.AFFECTS_OUTPUTS},
    metadataTags = {OptionMetadataTag.EXPERIMENTAL},
    help =
        "If true, use libunwind for stack unwinding, and compile with "
            + "-fomit-frame-pointer and -fasynchronous-unwind-tables."
  )
  public boolean experimentalOmitfp;

  @Option(
    name = "share_native_deps",
    defaultValue = "true",
    documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
    effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS, OptionEffectTag.AFFECTS_OUTPUTS},
    help =
        "If true, native libraries that contain identical functionality "
            + "will be shared among different targets"
  )
  public boolean shareNativeDeps;

  @Option(
    name = "strict_system_includes",
    defaultValue = "false",
    documentationCategory = OptionDocumentationCategory.INPUT_STRICTNESS,
    effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS, OptionEffectTag.EAGERNESS_TO_EXIT},
    help =
        "If true, headers found through system include paths (-isystem) are also required to be "
            + "declared."
  )
  public boolean strictSystemIncludes;

 @Option(
      name = "experimental_enable_cc_configuration_make_variables",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.UNKNOWN},
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help = "If enabled, the C++ configuration fragment supplies Make variables. This option "
          + "is used in the migration to remove them in favor of requiring an explicit "
          + "dependency on the C++ toolchain for rules that use them."
  )
  public boolean enableMakeVariables;

  @Option(
    name = "experimental_use_llvm_covmap",
    defaultValue = "false",
    documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
    effectTags = {
      OptionEffectTag.CHANGES_INPUTS,
      OptionEffectTag.AFFECTS_OUTPUTS,
      OptionEffectTag.LOADING_AND_ANALYSIS
    },
    metadataTags = {OptionMetadataTag.EXPERIMENTAL},
    help =
        "If specified, Bazel will generate llvm-cov coverage map information rather than "
            + "gcov when collect_code_coverage is enabled."
  )
  public boolean useLLVMCoverageMapFormat;

  @Option(
      name = "experimental_expand_linkopts_labels",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      help = "If true, entries in linkopts that are not preceded by - or $ will be expanded.")
  public boolean expandLinkoptsLabels;

  @Option(
      name = "incompatible_enable_legacy_cpp_toolchain_skylark_api",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      metadataTags = {
          OptionMetadataTag.INCOMPATIBLE_CHANGE,
          OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES},
      help = "Flag for disabling access to the C++ toolchain API through the ctx.fragments.cpp .")
  public boolean enableLegacyToolchainSkylarkApi;

  @Option(
      name = "experimental_enable_cc_skylark_api",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      help =
          "If true, the C++ Skylark API can be used. Don't enable this flag yet, we will be making "
              + "breaking changes.")
  public boolean enableCcSkylarkApi;

  @Override
  public FragmentOptions getHost() {
    CppOptions host = (CppOptions) getDefault();

    // The crosstool options are partially copied from the target configuration.
    if (hostCrosstoolTop == null) {
      host.cppCompiler = cppCompiler;
      host.crosstoolTop = crosstoolTop;
    } else {
      host.crosstoolTop = hostCrosstoolTop;
    }

    host.enableMakeVariables = enableMakeVariables;

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
    host.conlyoptList = ImmutableList.copyOf(hostConlyoptList);
    host.linkoptList = ImmutableList.copyOf(hostLinkoptList);

    host.useStartEndLib = useStartEndLib;
    host.stripBinaries = StripMode.ALWAYS;
    host.fdoProfiles = null;
    host.fdoProfileLabel = null;
    host.inmemoryDotdFiles = inmemoryDotdFiles;
    host.pruneCppInputDiscovery = pruneCppInputDiscovery;

    return host;
  }

  @Override
  public Map<String, Set<Label>> getDefaultsLabels() {
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

    return ImmutableMap.of("CROSSTOOL", crosstoolLabels);
  }

  /**
   * Returns true if targets under this configuration should apply FDO.
   */
  public boolean isFdo() {
    return getFdoOptimize() != null || fdoInstrumentForBuild != null || fdoProfileLabel != null;
  }
}
