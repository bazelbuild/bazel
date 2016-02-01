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
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.Constants;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.DefaultLabelConverter;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.LabelConverter;
import com.google.devtools.build.lib.analysis.config.CompilationMode;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.PerLabelOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration.LibcTop;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration.StripMode;
import com.google.devtools.build.lib.util.OptionsUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.LipoMode;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.Converters;
import com.google.devtools.common.options.EnumConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionsParsingException;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Command-line options for C++.
 */
public class CppOptions extends FragmentOptions {
  /** Custom converter for {@code --crosstool_top}. */
  public static class CrosstoolTopConverter extends DefaultLabelConverter {
    public CrosstoolTopConverter() {
      super(Constants.TOOLS_REPOSITORY + "//tools/cpp:toolchain");
    }
  }

  /**
   * Converter for --cwarn flag
   */
  public static class GccWarnConverter implements Converter<String> {
    @Override
    public String convert(String input) throws OptionsParsingException {
      if (input.startsWith("no-") || input.startsWith("-W")) {
        throw new OptionsParsingException("Not a valid gcc warning to enable");
      }
      return input;
    }

    @Override
    public String getTypeDescription() {
      return "A gcc warning to enable";
    }
  }

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
   * The same as DynamicMode, but on command-line we also allow AUTO.
   */
  public enum DynamicModeFlag { OFF, DEFAULT, FULLY, AUTO }

  /**
   * Converter for DynamicModeFlag
   */
  public static class DynamicModeConverter extends EnumConverter<DynamicModeFlag> {
    public DynamicModeConverter() {
      super(DynamicModeFlag.class, "dynamic mode");
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
   * Converts a String, which is an absolute path or label into a LibcTop
   * object.
   */
  public static class LibcTopConverter implements Converter<LibcTop> {
    @Override
    public LibcTop convert(String input) throws OptionsParsingException {
      if (!input.startsWith("//")) {
        throw new OptionsParsingException("Not a label");
      }
      try {
        Label label = Label.parseAbsolute(input).getRelative(LIBC_RELATIVE_LABEL);
        return new LibcTop(label);
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

  @Option(name = "lipo input collector",
      defaultValue = "false",
      category = "undocumented",
      help = "Internal flag, only used to create configurations with the LIPO-collector flag set.")
  public boolean lipoCollector;

  @Option(name = "crosstool_top",
          defaultValue = "",
          category = "version",
          converter = CrosstoolTopConverter.class,
          help = "The label of the crosstool package to be used for compiling C++ code.")
  public Label crosstoolTop;

  @Option(name = "compiler",
          defaultValue = "null",
          category = "version",
          help = "The C++ compiler to use for compiling the target.")
  public String cppCompiler;

  @Option(name = "glibc",
          defaultValue = "null",
          category = "version",
          help = "The version of glibc the target should be linked against. "
                 + "By default, a suitable version is chosen based on --cpu.")
  public String glibc;

  @Option(name = "thin_archives",
          defaultValue = "false",
          category = "strategy",  // but also adds edges to the action graph
          help = "Pass the 'T' flag to ar if supported by the toolchain. "
                 + "All supported toolchains support this setting.")
  public boolean useThinArchives;

  // O intrepid reaper of unused options: Be warned that the [no]start_end_lib
  // option, however tempting to remove, has a use case. Look in our telemetry data.
  @Option(name = "start_end_lib",
          defaultValue = "true",
          category = "strategy",  // but also adds edges to the action graph
          help = "Use the --start-lib/--end-lib ld options if supported by the toolchain.")
  public boolean useStartEndLib;

  @Option(name = "interface_shared_objects",
      defaultValue = "true",
      category = "strategy", // but also adds edges to the action graph
      help = "Use interface shared objects if supported by the toolchain. " +
             "All ELF toolchains currently support this setting.")
  public boolean useInterfaceSharedObjects;

  @Option(name = "fission",
          defaultValue = "no",
          converter = FissionOptionConverter.class,
          category = "semantics",
          help = "Specifies which compilation modes use fission for C++ compilations and links. "
          + " May be any combination of {'fastbuild', 'dbg', 'opt'} or the special values 'yes' "
          + " to enable all modes and 'no' to disable all modes.")
  public List<CompilationMode> fissionModes;

  @Option(name = "dynamic_mode",
          defaultValue = "default",
          converter = DynamicModeConverter.class,
          category = "semantics",
          help = "Determines whether C++ binaries will be linked dynamically.  'default' means "
            + "blaze will choose whether to link dynamically.  'fully' means all libraries "
            + "will be linked dynamically. 'off' means that all libraries will be linked "
            + "in mostly static mode.")
  public DynamicModeFlag dynamicMode;

  @Option(name = "force_pic",
          defaultValue = "false",
          category = "semantics",
          help = "If enabled, all C++ compilations produce position-independent code (\"-fPIC\"),"
            + " links prefer PIC pre-built libraries over non-PIC libraries, and links produce"
            + " position-independent executables (\"-pie\").")
  public boolean forcePic;

  @Option(name = "force_ignore_dash_static",
          defaultValue = "false",
          category = "semantics",
          help = "If set, '-static' options in the linkopts of cc_* rules will be ignored.")
  public boolean forceIgnoreDashStatic;

  @Option(name = "experimental_skip_static_outputs",
          defaultValue = "false",
          category = "semantics",
          help = "This flag is experimental and may go away at any time.  "
            + "If true, linker output for mostly-static C++ executables is a tiny amount of "
            + "dummy dependency information, and NOT a usable binary.  Kludge, but can reduce "
            + "network and disk I/O load (and thus, continuous build cycle times) by a lot.  "
            + "NOTE: use of this flag REQUIRES --distinct_host_configuration.")
  public boolean skipStaticOutputs;

  @Option(name = "copt",
          allowMultiple = true,
          defaultValue = "",
          category = "flags",
          help = "Additional options to pass to gcc.")
  public List<String> coptList;

  @Option(name = "cwarn",
          converter = GccWarnConverter.class,
          defaultValue = "",
          category = "flags",
          allowMultiple = true,
          help = "Additional warnings to enable when compiling C or C++ source files.")
  public List<String> cWarns;

  @Option(name = "cxxopt",
          defaultValue = "",
          category = "flags",
          allowMultiple = true,
          help = "Additional option to pass to gcc when compiling C++ source files.")
  public List<String> cxxoptList;

  @Option(name = "conlyopt",
          allowMultiple = true,
          defaultValue = "",
          category = "flags",
          help = "Additional option to pass to gcc when compiling C source files.")
  public List<String> conlyoptList;

  @Option(name = "linkopt",
          defaultValue = "",
          category = "flags",
          allowMultiple = true,
          help = "Additional option to pass to gcc when linking.")
  public List<String> linkoptList;

  @Option(name = "stripopt",
          allowMultiple = true,
          defaultValue = "",
          category = "flags",
          help = "Additional options to pass to strip when generating a '<name>.stripped' binary.")
  public List<String> stripoptList;

  @Option(name = "custom_malloc",
          defaultValue = "null",
          category = "semantics",
          help = "Specifies a custom malloc implementation. This setting overrides malloc " +
                 "attributes in build rules.",
          converter = LabelConverter.class)
  public Label customMalloc;

  @Option(name = "legacy_whole_archive",
          defaultValue = "true",
          category = "semantics",
          help = "When on, use --whole-archive for cc_binary rules that have "
            + "linkshared=1 and either linkstatic=1 or '-static' in linkopts. "
            + "This is for backwards compatibility only. "
            + "A better alternative is to use alwayslink=1 where required.")
  public boolean legacyWholeArchive;

  @Option(name = "strip",
      defaultValue = "sometimes",
      category = "flags",
      help = "Specifies whether to strip binaries and shared libraries "
          + " (using \"-Wl,--strip-debug\").  The default value of 'sometimes'"
          + " means strip iff --compilation_mode=fastbuild.",
      converter = StripModeConverter.class)
  public StripMode stripBinaries;

  @Option(name = "fdo_instrument",
          defaultValue = "null",
          converter = OptionsUtils.PathFragmentConverter.class,
          category = "flags",
          implicitRequirements = {"--copt=-Wno-error"},
          help = "Generate binaries with FDO instrumentation. Specify the relative " +
                 "directory name for the .gcda files at runtime. It also accepts " +
                 "an LLVM profile output file path.")
  public PathFragment fdoInstrument;

  @Option(name = "fdo_optimize",
          defaultValue = "null",
          category = "flags",
          help = "Use FDO profile information to optimize compilation. Specify the name " +
                 "of the zip file containing the .gcda file tree or an afdo file containing " +
                 "an auto profile. This flag also accepts files specified as labels, for " +
                 "example //foo/bar:file.afdo. Such labels must refer to input files; you may " +
                 "need to add an exports_files directive to the corresponding package to make " +
                 "the file visible to Blaze. It also accepts an indexed LLVM profile file.")
  public String fdoOptimize;

  @Option(name = "autofdo_lipo_data",
          defaultValue = "false",
          category = "flags",
          help = "If true then the directory name for non-LIPO targets will have a " +
                 "'-lipodata' suffix in AutoFDO mode.")
  public boolean autoFdoLipoData;

  @Option(name = "lipo",
      defaultValue = "off",
      converter = LipoModeConverter.class,
      category = "flags",
      help = "Enable LIPO optimization (lightweight inter-procedural optimization, The allowed "
          + "values for  this option are 'off' and 'binary', which enables LIPO. This option only "
          + "has an effect when FDO is also enabled. Currently LIPO is only supported when "
          + "building a single cc_binary rule.")
  public LipoMode lipoMode;

  @Option(name = "lipo_context",
      defaultValue = "null",
      category = "flags",
      converter = LabelConverter.class,
      implicitRequirements = {"--linkopt=-Wl,--warn-unresolved-symbols"},
      help = "Specifies the binary from which the LIPO profile information comes.")
  public Label lipoContext;

  @Option(name = "experimental_stl",
      converter = LabelConverter.class,
      defaultValue = "null",
      category = "version",
      help = "If set, use this label instead of the default STL implementation. "
          + "This option is EXPERIMENTAL and may go away in a future release.")
  public Label stl;

  @Option(name = "save_temps",
      defaultValue = "false",
      category = "what",
      help = "If set, temporary outputs from gcc will be saved.  "
          + "These include .s files (assembler code), .i files (preprocessed C) and "
          + ".ii files (preprocessed C++).")
  public boolean saveTemps;

  @Option(name = "per_file_copt",
      allowMultiple = true,
      converter = PerLabelOptions.PerLabelOptionsConverter.class,
      defaultValue = "",
      category = "semantics",
      help = "Additional options to selectively pass to gcc when compiling certain files. "
            + "This option can be passed multiple times. "
            + "Syntax: regex_filter@option_1,option_2,...,option_n. Where regex_filter stands "
            + "for a list of include and exclude regular expression patterns (Also see "
            + "--instrumentation_filter). option_1 to option_n stand for "
            + "arbitrary command line options. If an option contains a comma it has to be "
            + "quoted with a backslash. Options can contain @. Only the first @ is used to "
            + "split the string. Example: "
            + "--per_file_copt=//foo/.*\\.cc,-//foo/bar\\.cc@-O0 adds the -O0 "
            + "command line option to the gcc command line of all cc files in //foo/ "
            + "except bar.cc.")
  public List<PerLabelOptions> perFileCopts;

  @Option(name = "host_crosstool_top",
      defaultValue = "null",
      converter = LabelConverter.class,
      category = "semantics",
      help = "By default, the --crosstool_top and --compiler options are also used " +
          "for the host configuration. If this flag is provided, Blaze uses the default libc " +
          "and compiler for the given crosstool_top.")
  public Label hostCrosstoolTop;

  @Option(name = "host_copt",
      allowMultiple = true,
      defaultValue = "",
      category = "flags",
      help = "Additional options to pass to gcc for host tools.")
  public List<String> hostCoptList;

  @Option(name = "define",
      converter = Converters.AssignmentConverter.class,
      defaultValue = "",
      category = "semantics",
      allowMultiple = true,
      help = "Each --define option specifies an assignment for a build variable.")
  public List<Map.Entry<String, String>> commandLineDefinedVariables;

  @Option(name = "grte_top",
      defaultValue = "null", // The default value is chosen by the toolchain.
      category = "version",
      converter = LibcTopConverter.class,
      help = "A label to a checked-in libc library. The default value is selected by the crosstool "
          + "toolchain, and you almost never need to override it.")
  public LibcTop libcTop;

  @Option(name = "host_grte_top",
      defaultValue = "null", // The default value is chosen by the toolchain.
      category = "version",
      converter = LibcTopConverter.class,
      help = "If specified, this setting overrides the libc top-level directory (--grte_top) "
          + "for the host configuration.")
  public LibcTop hostLibcTop;

  @Option(name = "output_symbol_counts",
      defaultValue = "false",
      category = "flags",
      help = "If enabled, every C++ binary linked with gold will store the number of used "
          + "symbols per object file in a .sc file.")
  public boolean symbolCounts;

  @Option(name = "experimental_inmemory_dotd_files",
      defaultValue = "false",
      category = "experimental",
      help = "If enabled, C++ .d files will be passed through in memory directly from the remote "
          + "build nodes instead of being written to disk.")
  public boolean inmemoryDotdFiles;

  @Option(name = "use_isystem_for_includes",
      defaultValue = "true",
      category = "undocumented",
      help = "Instruct C and C++ compilations to treat 'includes' paths as system header " +
             "paths, by translating it into -isystem instead of -I.")
  public boolean useIsystemForIncludes;

  @Option(name = "experimental_omitfp",
      defaultValue = "false",
      category = "semantics",
      help = "If true, use libunwind for stack unwinding, and compile with " +
      "-fomit-frame-pointer and -fasynchronous-unwind-tables.")
  public boolean experimentalOmitfp;

  @Option(name = "share_native_deps",
      defaultValue = "true",
      category = "strategy",
      help = "If true, native libraries that contain identical functionality "
          + "will be shared among different targets")
  public boolean shareNativeDeps;

  @Override
  public FragmentOptions getHost(boolean fallback) {
    CppOptions host = (CppOptions) getDefault();

    host.commandLineDefinedVariables = commandLineDefinedVariables;

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

    // hostLibcTop doesn't default to the target's libcTop.
    // Only an explicit command-line option will change it.
    // The default is whatever the host's crosstool (which might have been specified
    // by --host_crosstool_top, or --crosstool_top as a fallback) says it should be.
    host.libcTop = hostLibcTop;

    // -g0 is the default, but allowMultiple options cannot have default values so we just pass
    // -g0 first and let the user options override it.
    host.coptList = ImmutableList.<String>builder().add("-g0").addAll(hostCoptList).build();

    host.useThinArchives = useThinArchives;
    host.useStartEndLib = useStartEndLib;
    host.stripBinaries = StripMode.ALWAYS;
    host.fdoOptimize = null;
    host.lipoMode = LipoMode.OFF;
    host.inmemoryDotdFiles = inmemoryDotdFiles;

    return host;
  }

  @Override
  public void addAllLabels(Multimap<String, Label> labelMap) {
    labelMap.put("crosstool", crosstoolTop);
    if (hostCrosstoolTop != null) {
      labelMap.put("crosstool", hostCrosstoolTop);
    }

    if (libcTop != null) {
      Label libcLabel = libcTop.getLabel();
      if (libcLabel != null) {
        labelMap.put("crosstool", libcLabel);
      }
    }
    addOptionalLabel(labelMap, "fdo", fdoOptimize);

    if (stl != null) {
      labelMap.put("STL", stl);
    }

    if (customMalloc != null) {
      labelMap.put("custom_malloc", customMalloc);
    }

    if (getLipoContextLabel() != null) {
      labelMap.put("lipo", getLipoContextLabel());
    }
  }

  @Override
  public Map<String, Set<Label>> getDefaultsLabels(BuildConfiguration.Options commonOptions) {
    Set<Label> crosstoolLabels = new LinkedHashSet<>();
    crosstoolLabels.add(crosstoolTop);
    if (hostCrosstoolTop != null) {
      crosstoolLabels.add(hostCrosstoolTop);
    }

    if (libcTop != null) {
      Label libcLabel = libcTop.getLabel();
      if (libcLabel != null) {
        crosstoolLabels.add(libcLabel);
      }
    }

    return ImmutableMap.of(
        "CROSSTOOL", crosstoolLabels,
        "COVERAGE", ImmutableSet.<Label>of());
  }

  public boolean isFdo() {
    return fdoOptimize != null || fdoInstrument != null;
  }

  public boolean isLipoOptimization() {
    return lipoMode == LipoMode.BINARY && fdoOptimize != null && lipoContext != null;
  }

  public boolean isLipoOptimizationOrInstrumentation() {
    return lipoMode == LipoMode.BINARY &&
        ((fdoOptimize != null && lipoContext != null) || fdoInstrument != null);
  }

  public Label getLipoContextLabel() {
    return (lipoMode == LipoMode.BINARY && fdoOptimize != null)
        ? lipoContext : null;
  }

  public LipoMode getLipoMode() {
    return lipoMode;
  }
}
