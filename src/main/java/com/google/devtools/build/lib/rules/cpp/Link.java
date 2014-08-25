// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.common.base.Function;
import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicates;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.rules.cpp.LinkerInputs.LibraryToLink;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.devtools.build.lib.util.ShellEscaper;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.config.BuildConfiguration;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Utility types and methods for generating command lines for the linker, given
 * a CppLinkAction or LinkConfiguration.
 *
 * <p>The linker commands, e.g. "ar", may not be functional, i.e.
 * they may mutate the output file rather than overwriting it.
 * To avoid this, we need to delete the output file before invoking the
 * command.  But that is not done by this class; deleting the output
 * file is the responsibility of the classes derived from LinkStrategy.
 */
public abstract class Link {

  // TODO(bazel-team): Move to LinkerInputs class.
  private static final Function<LibraryToLink, Artifact> LIBRARY_TO_NON_SOLIB =
      new Function<LibraryToLink, Artifact>() {
        @Override
        public Artifact apply(LibraryToLink input) {
          return input.getOriginalLibraryArtifact();
        }
      };

  static Iterable<Artifact> toNonSolibArtifacts(Iterable<LibraryToLink> libraries) {
    return Iterables.transform(libraries, LIBRARY_TO_NON_SOLIB);
  }

  /**
   * Returns the linker input artifacts from a collection of {@link LinkerInput} objects.
   */
  public static Iterable<Artifact> toLibraryArtifacts(Iterable<? extends LinkerInput> artifacts) {
    return Iterables.transform(artifacts, new Function<LinkerInput, Artifact>() {
      @Override
      public Artifact apply(LinkerInput input) {
        return input.getArtifact();
      }
    });
  }


  private Link() {} // uninstantiable

  /** The set of valid linker input files.  */
  public static final FileTypeSet VALID_LINKER_INPUTS = FileTypeSet.of(
      CppFileTypes.ARCHIVE, CppFileTypes.PIC_ARCHIVE,
      CppFileTypes.ALWAYS_LINK_LIBRARY, CppFileTypes.ALWAYS_LINK_PIC_LIBRARY,
      CppFileTypes.OBJECT_FILE, CppFileTypes.PIC_OBJECT_FILE,
      CppFileTypes.SHARED_LIBRARY, CppFileTypes.VERSIONED_SHARED_LIBRARY,
      CppFileTypes.INTERFACE_SHARED_LIBRARY);

  /**
   * These file are supposed to be added using {@code addLibrary()} calls to {@link CppLinkAction}
   * but will never be expanded to their constituent {@code .o} files. {@link CppLinkAction} checks
   * that these files are never added as non-libraries.
   */
  public static final FileTypeSet SHARED_LIBRARY_FILETYPES = FileTypeSet.of(
      CppFileTypes.SHARED_LIBRARY,
      CppFileTypes.VERSIONED_SHARED_LIBRARY,
      CppFileTypes.INTERFACE_SHARED_LIBRARY);

  /** These need special handling when --thin_archive is true. {@link CppLinkAction} checks that
   * these files are never added as non-libraries.
   */
  public static final FileTypeSet ARCHIVE_LIBRARY_FILETYPES = FileTypeSet.of(
      CppFileTypes.ARCHIVE,
      CppFileTypes.PIC_ARCHIVE,
      CppFileTypes.ALWAYS_LINK_LIBRARY,
      CppFileTypes.ALWAYS_LINK_PIC_LIBRARY);

  public static final FileTypeSet ARCHIVE_FILETYPES = FileTypeSet.of(
      CppFileTypes.ARCHIVE,
      CppFileTypes.PIC_ARCHIVE);

  public static final FileTypeSet LINK_LIBRARY_FILETYPES = FileTypeSet.of(
      CppFileTypes.ALWAYS_LINK_LIBRARY,
      CppFileTypes.ALWAYS_LINK_PIC_LIBRARY);


  /** The set of object files */
  public static final FileTypeSet OBJECT_FILETYPES = FileTypeSet.of(
      CppFileTypes.OBJECT_FILE,
      CppFileTypes.PIC_OBJECT_FILE);

  /**
   * Prefix that is prepended to command line entries that refer to the output
   * of cc_fake_binary compile actions. This is a bad hack to signal to the code
   * in {@code CppLinkAction#executeFake(Executor, FileOutErr)} that it needs
   * special handling.
   */
  public static final String FAKE_OBJECT_PREFIX = "fake:";

  /**
   * Types of ELF files that can be created by the linker (.a, .so, .lo,
   * executable).
   */
  public enum LinkTargetType {
    /** A normal static archive. */
    STATIC_LIBRARY(".a", true),

    /** A static archive with .pic.o object files (compiled with -fPIC). */
    PIC_STATIC_LIBRARY(".pic.a", true),

    /** An interface dynamic library. */
    INTERFACE_DYNAMIC_LIBRARY(".ifso", false),

    /** A dynamic library. */
    DYNAMIC_LIBRARY(".so", false),

    /** A static archive without removal of unused object files. */
    ALWAYS_LINK_STATIC_LIBRARY(".lo", true),

    /** A PIC static archive without removal of unused object files. */
    ALWAYS_LINK_PIC_STATIC_LIBRARY(".pic.lo", true),

    /** An executable binary. */
    EXECUTABLE("", false);

    private final String extension;
    private final boolean staticLibraryLink;

    private LinkTargetType(String extension, boolean staticLibraryLink) {
      this.extension = extension;
      this.staticLibraryLink = staticLibraryLink;
    }

    public String getExtension() {
      return extension;
    }

    public boolean isStaticLibraryLink() {
      return staticLibraryLink;
    }
  }

  /**
   * The degree of "staticness" of symbol resolution during linking.
   */
  public enum LinkStaticness {
    FULLY_STATIC,       // Static binding of all symbols.
    MOSTLY_STATIC,      // Use dynamic binding only for symbols from glibc.
    DYNAMIC,            // Use dynamic binding wherever possible.
  }

  /**
   * Types of archive.
   */
  public enum ArchiveType {
    FAT,            // Regular archive that includes its members.
    THIN,           // Thin archive that just points to its members.
    START_END_LIB   // A --start-lib ... --end-lib group in the command line.
  }

  private static boolean useStartEndLib(LinkerInput linkerInput, ArchiveType archiveType) {
    // TODO(bazel-team): Figure out if PicArchives are actually used. For it to be used, both
    // linkingStatically and linkShared must me true, we must be in opt mode and cpu has to be k8.
    return archiveType == ArchiveType.START_END_LIB
        && ARCHIVE_FILETYPES.matches(linkerInput.getArtifact().getFilename())
        && linkerInput.containsObjectFiles();
  }

  /**
   * Expands the archives in a collection of artifacts. If deps is true we include all
   * dependencies. If it is false, only what should be passed to the link command.
   */
  private static Iterable<LinkerInput> filterMembersForLink(Iterable<LibraryToLink> inputs,
      final boolean globalNeedWholeArchive, final ArchiveType archiveType, final boolean deps) {
    ImmutableList.Builder<LinkerInput> builder = ImmutableList.builder();

    for (LibraryToLink inputLibrary : inputs) {
      Artifact input = inputLibrary.getArtifact();
      String name = input.getFilename();

      // True if the linker might use the members of this file.
      boolean needMembersForLink = archiveType != ArchiveType.FAT &&
          ARCHIVE_LIBRARY_FILETYPES.matches(name) && inputLibrary.containsObjectFiles();
      // True if we will pass the members instead of the original archive.
      boolean passMembersToLinkCmd = needMembersForLink &&
          (globalNeedWholeArchive || LINK_LIBRARY_FILETYPES.matches(name));

      if (passMembersToLinkCmd || (needMembersForLink && deps)) {
        builder.addAll(LinkerInputs.simpleLinkerInputs(inputLibrary.getObjectFiles()));
      }

      if (!passMembersToLinkCmd && (!deps || !useStartEndLib(inputLibrary, archiveType))) {
        builder.add(inputLibrary);
      }
    }

    return builder.build();
  }

  /**
   * Replace always used archives with its members. This is used to build the linker cmd line.
   */
  public static Iterable<LinkerInput> mergeInputsCmdLine(Iterable<LibraryToLink> inputs,
      boolean globalNeedWholeArchive, ArchiveType archiveType) {
    return filterMembersForLink(inputs, globalNeedWholeArchive, archiveType, false);
  }

  /**
   * Add in any object files which are implicitly named as inputs by the linker.
   */
  public static Iterable<LinkerInput> mergeInputsDependencies(Iterable<LibraryToLink> inputs,
      boolean globalNeedWholeArchive, ArchiveType archiveType) {
    return filterMembersForLink(inputs, globalNeedWholeArchive, archiveType, true);
  }

  /**
   * Returns a new, mutable list of command and arguments (argv) to be passed to
   * the linker subprocess.
   *
   * @param linkConfig configuration on what to build and how to build it
   * @return the list of arguments to pass to exec.
   */
  static List<String> getArgv(LinkConfiguration linkConfig) {
    return finalizeWithLinkstampCommands(linkConfig, getRawLinkArgv(linkConfig));
  }

  /**
   * Returns a raw link command for the given link invocation, including both command and
   * arguments (argv). After any further usage-specific processing, this can be passed to
   * {@link #finalizeWithLinkstampCommands} to give the final command line.
   *
   * @param linkConfig configuration on what to build and how to build it
   *     this action's last execution.
   * @return raw link command line.
   */
  public static List<String> getRawLinkArgv(LinkConfiguration linkConfig) {
    CppConfiguration cppConfig = linkConfig.getConfiguration().getFragment(CppConfiguration.class);

    LinkTargetType targetType = linkConfig.getLinkTargetType();
    List<String> argv = new ArrayList<>();
    switch (targetType) {
      case EXECUTABLE:
        addCppArgv(linkConfig, argv);
        break;

      case DYNAMIC_LIBRARY:
        if (linkConfig.getInterfaceOutput() != null) {
          BuildConfiguration config = linkConfig.getConfiguration();
          argv.add(config.getShExecutable().getPathString());
          argv.add("-c");
          argv.add("build_iface_so=\"$0\"; impl=\"$1\"; iface=\"$2\"; cmd=\"$3\"; shift 3; " +
              "\"$cmd\" \"$@\" && \"$build_iface_so\" \"$impl\" \"$iface\"");
          argv.add(linkConfig.buildInterfaceSo().getExecPathString());
          argv.add(linkConfig.getOutput().getExecPathString());
          argv.add(linkConfig.getInterfaceOutput().getExecPathString());
        }
        addCppArgv(linkConfig, argv);
        // -pie is not compatible with -shared and should be
        // removed when the latter is part of the link command. Should we need to further
        // distinguish between shared libraries and executables, we could add additional
        // command line / CROSSTOOL flags that distinguish them. But as long as this is
        // the only relevant use case we're just special-casing it here.
        Iterables.removeIf(argv, Predicates.equalTo("-pie"));
        break;

      case STATIC_LIBRARY:
      case PIC_STATIC_LIBRARY:
      case ALWAYS_LINK_STATIC_LIBRARY:
      case ALWAYS_LINK_PIC_STATIC_LIBRARY:
        // The static library link command follows this template:
        // ar <cmd> <output_archive> <input_files...>
        argv.add(cppConfig.getArExecutable().getPathString());
        argv.addAll(cppConfig.getArFlags(cppConfig.archiveType() == Link.ArchiveType.THIN));
        argv.add(linkConfig.getOutput().getExecPathString());
        addInputFileLinkOptions(argv, linkConfig,
            /*needWholeArchive=*/false, /*includeLinkopts=*/false);
        break;

      default:
        throw new IllegalArgumentException();
    }

    // Fission mode: debug info is in .dwo files instead of .o files. Inform the linker of this.
    if ((targetType == LinkTargetType.EXECUTABLE || targetType == LinkTargetType.DYNAMIC_LIBRARY)
        && cppConfig.useFission()) {
      argv.add("-Wl,--gdb-index");
    }

    return argv;
  }

  /**
   * Takes a raw link command line and gives the final link command that will
   * also first compile any linkstamps necessary. Elements of rawLinkArgv are
   * shell-escaped.
   * @param linkConfig link configuration
   * @param rawLinkArgv raw link command line
   *
   * @return final link command line suitable for execution
   */
  public static List<String> finalizeWithLinkstampCommands(
      LinkConfiguration linkConfig, List<String> rawLinkArgv) {
    return addLinkstampingToCommand(getLinkstampCompileCommands(linkConfig, ""),
        rawLinkArgv, true);
  }

  /**
   * Takes a raw link command line and gives the final link command that will
   * also first compile any linkstamps necessary. Elements of rawLinkArgv are
   * not shell-escaped.
   * @param linkConfig link configuration
   * @param rawLinkArgv raw link command line
   * @param outputPrefix prefix to add before the linkstamp outputs' exec paths
   *
   * @return final link command line suitable for execution
   */
  static List<String> finalizeAlreadyEscapedWithLinkstampCommands(
      LinkConfiguration linkConfig, List<String> rawLinkArgv, String outputPrefix) {
    return addLinkstampingToCommand(getLinkstampCompileCommands(linkConfig, outputPrefix),
        rawLinkArgv, false);
  }

  /**
   * Adds linkstamp compilation to the (otherwise) fully specified link
   * command if {@link LinkConfiguration#getLinkstamps} is non-empty.
   *
   * <p>Linkstamps were historically compiled implicitly as part of the link
   * command, but implicit compilation doesn't guarantee consistent outputs.
   * For example, the command "gcc input.o input.o foo/linkstamp.cc -o myapp"
   * causes gcc to implicitly run "gcc foo/linkstamp.cc -o /tmp/ccEtJHDB.o",
   * for some internally decided output path /tmp/ccEtJHDB.o, then add that path
   * to the linker's command line options. The name of this path can change
   * even between equivalently specified gcc invocations.
   *
   * <p>So now we explicitly compile these files in their own command
   * invocations before running the link command, thus giving us direct
   * control over the naming of their outputs. This method adds those extra
   * steps as necessary.
   * @param linkstampCommands individual linkstamp compilation commands
   * @param linkCommand the complete list of link command arguments (after
   *        .params file compacting) for an invocation
   * @param escapeArgs if true, linkCommand arguments are shell escaped. if
   *        false, arguments are returned as-is
   *
   * @return The original argument list if no linkstamps compilation commands
   *         are given, otherwise an expanded list that adds the linkstamp
   *         compilation commands and funnels their outputs into the link step.
   *         Note that these outputs only need to persist for the duration of
   *         the link step.
   */
  private static List<String> addLinkstampingToCommand(
      List<String> linkstampCommands,
      List<String> linkCommand,
      boolean escapeArgs) {
    if (linkstampCommands.isEmpty()) {
      return linkCommand;
    } else {
      List<String> batchCommand = Lists.newArrayListWithCapacity(3);
      batchCommand.add("/bin/bash");
      batchCommand.add("-c");
      batchCommand.add(
          Joiner.on(" && ").join(linkstampCommands) + " && " +
          (escapeArgs
              ? ShellEscaper.escapeJoinAll(linkCommand)
              : Joiner.on(" ").join(linkCommand)));
      return ImmutableList.copyOf(batchCommand);
    }
  }

  /**
   * Computes, for each C++ source file in
   * {@link LinkConfiguration#getLinkstamps}, the command necessary to compile
   * that file such that the output is correctly fed into the link command.
   *
   * <p>As these options (as well as all others) are taken into account when
   * computing the action key, they do not directly contain volatile build
   * information to avoid unnecessary relinking. Instead this information is
   * passed as an additional header generated by
   * {@link com.google.devtools.build.lib.rules.cpp.WriteBuildInfoHeaderAction}.
   *
   * @param linkConfig a link configuration
   * @param outputPrefix prefix to add before the linkstamp outputs' exec paths
   * @return a list of shell-escaped compiler commmands, one for each entry
   *         in {@link LinkConfiguration#getLinkstamps}
   */
  public static List<String> getLinkstampCompileCommands(LinkConfiguration linkConfig,
      String outputPrefix) {
    Map<Artifact, Artifact> linkstamps = linkConfig.getLinkstamps();
    if (linkstamps.isEmpty()) {
      return ImmutableList.of();
    }

    CppConfiguration cppConfig = linkConfig.getConfiguration().getFragment(CppConfiguration.class);
    String compilerCommand = cppConfig.getCppExecutable().getPathString();
    List<String> commands = Lists.newArrayListWithCapacity(linkstamps.size());

    for (Map.Entry<Artifact, Artifact> linkstamp : linkstamps.entrySet()) {
      List<String> optionList = new ArrayList<>();

      // Defines related to the build info are read from generated headers.
      for (Artifact header : linkConfig.getBuildInfoHeaderArtifacts()) {
        optionList.add("-include");
        optionList.add(header.getExecPathString());
      }

      // G3_VERSION_INFO and G3_TARGET_NAME are C string literals that normally
      // contain the label of the target being linked.  However, they are set
      // differently when using shared native deps. In that case, a single .so file
      // is shared by multiple targets, and its contents cannot depend on which
      // target(s) were specified on the command line.  So in that case we have
      // to use the (obscure) name of the .so file instead, or more precisely
      // the path of the .so file relative to the workspace root.
      String targetLabel = isSharedNativeLibrary(linkConfig)
          ? linkConfig.getOutput().getExecPathString()
          : Label.print(linkConfig.getOwner().getLabel());
      optionList.add("-DG3_VERSION_INFO=\"" + targetLabel + "\"");
      optionList.add("-DG3_TARGET_NAME=\"" + targetLabel + "\"");

      // G3_BUILD_TARGET is a C string literal containing the output of this
      // link.  (An undocumented and untested invariant is that G3_BUILD_TARGET is the location of
      // the executable, either absolutely, or relative to the directory part of BUILD_INFO.)
      optionList.add("-DG3_BUILD_TARGET=\"" + linkConfig.getOutput().getExecPathString() + "\"");

      optionList.add("-DGPLATFORM=\"" + cppConfig + "\"");

      // Needed to find headers included from linkstamps.
      optionList.add("-I.");

      // Add sysroot.
      PathFragment sysroot = cppConfig.getSysroot();
      if (sysroot != null) {
        optionList.add("--sysroot=" + sysroot.getPathString());
      }

      // Add toolchain compiler options.
      optionList.addAll(cppConfig.getCompilerOptions(linkConfig.getFeatures()));
      optionList.addAll(cppConfig.getCOptions());
      optionList.addAll(cppConfig.getUnfilteredCompilerOptions(linkConfig.getFeatures()));

      // For dynamic libraries, produce position independent code.
      if (linkConfig.getLinkTargetType() == LinkTargetType.DYNAMIC_LIBRARY &&
          cppConfig.toolchainNeedsPic()) {
        optionList.add("-fPIC");
      }

      // Stamp FDO builds with FDO subtype string
      String fdoBuildStamp = CppHelper.getFdoBuildStamp(cppConfig);
      if (fdoBuildStamp != null) {
        optionList.add("-D" + CppConfiguration.FDO_STAMP_MACRO + "=\"" + fdoBuildStamp + "\"");
      }

      // Add the compilation target.
      optionList.add("-c");
      optionList.add(linkstamp.getKey().getExecPathString());

      // Assemble the final command, exempting outputPrefix from shell escaping.
      commands.add(compilerCommand + " "
          + ShellEscaper.escapeJoinAll(optionList)
          + " -o "
          + outputPrefix
          + ShellEscaper.escapeString(linkstamp.getValue().getExecPathString()));
    }

    return commands;
  }

  static boolean needWholeArchive(LinkStaticness staticness,
      LinkTargetType type, Collection<String> linkopts, boolean isNativeDeps,
      CppConfiguration cppConfig) {
    boolean fullyStatic = (staticness == LinkStaticness.FULLY_STATIC);
    boolean mostlyStatic = (staticness == LinkStaticness.MOSTLY_STATIC);
    boolean sharedLinkopts =
        type == LinkTargetType.DYNAMIC_LIBRARY ||
        linkopts.contains("-shared") ||
        cppConfig.getLinkOptions().contains("-shared");
    return (isNativeDeps || cppConfig.legacyWholeArchive()) &&
        (fullyStatic || mostlyStatic) &&
        sharedLinkopts;
  }

  /**
   * Determine the arguments to pass to the C++ compiler when linking.
   * Add them to the {@code argv} parameter.
   */
  private static void addCppArgv(LinkConfiguration linkConfig, List<String> argv) {
    BuildConfiguration config = linkConfig.getConfiguration();
    CppConfiguration cppConfig = config.getFragment(CppConfiguration.class);

    argv.add(cppConfig.getCppExecutable().getPathString());

    // When using gold to link an executable, output the number of used and unused symbols.
    if (linkConfig.getSymbolCountsOutput() != null) {
      argv.add("-Wl,--print-symbol-counts=" +
          linkConfig.getSymbolCountsOutput().getExecPathString());
    }

    if (linkConfig.getLinkTargetType() == LinkTargetType.DYNAMIC_LIBRARY) {
      argv.add("-shared");
    }

    // Add the outputs of any associated linkstamp compilations.
    for (Artifact linkstampOutput : linkConfig.getLinkstamps().values()) {
      argv.add(linkstampOutput.getExecPathString());
    }

    boolean fullyStatic = (linkConfig.getLinkStaticness() == LinkStaticness.FULLY_STATIC);
    boolean mostlyStatic = (linkConfig.getLinkStaticness() == LinkStaticness.MOSTLY_STATIC);
    boolean sharedLinkopts =
        linkConfig.getLinkTargetType() == LinkTargetType.DYNAMIC_LIBRARY ||
        linkConfig.getLinkopts().contains("-shared") ||
        cppConfig.getLinkOptions().contains("-shared");
    boolean needWholeArchive = needWholeArchive(linkConfig.getLinkStaticness(),
          linkConfig.getLinkTargetType(), linkConfig.getLinkopts(), linkConfig.isNativeDeps(),
          cppConfig);

    if (linkConfig.getOutput() != null) {
      argv.add("-o");
      String execpath = linkConfig.getOutput().getExecPathString();
      if (mostlyStatic && linkConfig.getLinkTargetType() == LinkTargetType.EXECUTABLE &&
          cppConfig.skipStaticOutputs()) {
        // Linked binary goes to /dev/null; bogus dependency info in its place.
        Collections.addAll(argv, "/dev/null", "-MMD", "-MF", execpath);  // thanks Ambrose
      } else {
        argv.add(execpath);
      }
    }

    addInputFileLinkOptions(argv, linkConfig, needWholeArchive, /*includeLinkopts=*/true);

    // Extra toolchain link options based on the output's link staticness.
    Collection<String> features = linkConfig.getFeatures();
    if (fullyStatic) {
      argv.addAll(cppConfig.getFullyStaticLinkOptions(features, sharedLinkopts));
    } else if (mostlyStatic) {
      argv.addAll(cppConfig.getMostlyStaticLinkOptions(features, sharedLinkopts));
    } else {
      argv.addAll(cppConfig.getDynamicLinkOptions(features, sharedLinkopts));
    }

    if (config.isCodeCoverageEnabled()) {
      // Note we apply the same logic independently in GoCompilationHelper (using "--coverage").
      // Keep this in mind if this ever gets moved out to CROSSTOOL or a centralized place
      // for both languages becomes available.
      argv.add("-lgcov");
    }

    if (linkConfig.getLinkTargetType() == LinkTargetType.EXECUTABLE && cppConfig.forcePic()) {
      argv.add("-pie");
    }

    argv.addAll(cppConfig.getLinkOptions());
    argv.addAll(cppConfig.getFdoSupport().getLinkOptions());
  }

  private static boolean isDynamicLibrary(LinkerInput linkInput) {
    Artifact libraryArtifact = linkInput.getArtifact();
    String name = libraryArtifact.getFilename();
    return SHARED_LIBRARY_FILETYPES.matches(name) && name.startsWith("lib");
  }

  private static boolean isSharedNativeLibrary(LinkConfiguration linkConfig) {
    return linkConfig.isNativeDeps() &&
        linkConfig.getConfiguration().getFragment(CppConfiguration.class).shareNativeDeps();
  }

  /**
   * When linking a shared library fully or mostly static then we need to link in
   * *all* dependent files, not just what the shared library needs for its own
   * code. This is done by wrapping all objects/libraries with
   * -Wl,-whole-archive and -Wl,-no-whole-archive. For this case the
   * globalNeedWholeArchive parameter must be set to true.  Otherwise only
   * library objects (.lo) need to be wrapped with -Wl,-whole-archive and
   * -Wl,-no-whole-archive.
   */
  private static void addInputFileLinkOptions(List<String> argv,
                                              LinkConfiguration linkConfig,
                                              boolean globalNeedWholeArchive,
                                              boolean includeLinkopts) {
    // The Apple ld doesn't support -whole-archive/-no-whole-archive. It
    // does have -all_load/-noall_load, but -all_load is a global setting
    // that affects all subsequent files, and -noall_load is simply ignored.
    // TODO(bazel-team): Not sure what the implications of this are, other than
    // bloated binaries.
    CppConfiguration cppConfig = linkConfig.getConfiguration().getFragment(CppConfiguration.class);
    boolean macosx = cppConfig.getTargetLibc().equals("macosx");
    if (globalNeedWholeArchive) {
      argv.add(macosx ? "-Wl,-all_load" : "-Wl,-whole-archive");
    }

    // Used to collect -L and -Wl,-rpath options, ensuring that each used only once.
    Set<String> libOpts = new LinkedHashSet<>();

    // List of command line parameters to link input files (either directly or using -l).
    List<String> linkerInputs = new ArrayList<>();

    // List of command line parameters that need to be placed *outside* of
    // --whole-archive ... --no-whole-archive.
    List<String> noWholeArchiveInputs = new ArrayList<>();

    PathFragment solibDir = linkConfig.getConfiguration().getBinDirectory().getExecPath()
        .getRelative(cppConfig.getSolibDirectory());
    PathFragment runtimeSolibDir = linkConfig.getRuntimeSolibDir();
    String runtimeSolibName = runtimeSolibDir != null ? runtimeSolibDir.getBaseName() : null;
    boolean runtimeRpath = runtimeSolibDir != null &&
        (linkConfig.getLinkTargetType() == LinkTargetType.DYNAMIC_LIBRARY ||
        (linkConfig.getLinkTargetType() == LinkTargetType.EXECUTABLE &&
         linkConfig.getLinkStaticness() == LinkStaticness.DYNAMIC));

    String rpathRoot = null;

    if (linkConfig.getOutput() != null) {
      String origin = linkConfig.useExecOrigin() ? "$EXEC_ORIGIN/" : "$ORIGIN/";
      rpathRoot = "-Wl,-rpath," + origin;
      if (runtimeRpath) {
        argv.add("-Wl,-rpath," + origin + runtimeSolibName + "/");
      }

      // Calculate the correct relative value for the "-rpath" link option (which sets
      // the search path for finding shared libraries).
      if (isSharedNativeLibrary(linkConfig)) {
        // For shared native libraries, special symlinking is applied to ensure C++
        // runtimes are available under $ORIGIN/_solib_[arch]. So we set the RPATH to find
        // them.
        //
        // Note that we have to do this because $ORIGIN points to different paths for
        // different targets. In other words, blaze-bin/p1/p2/p3/a_shareddeps.so and
        // blaze-bin/p4/b_shareddeps.so have different path depths. The first could
        // reference a standard blaze-bin/_solib_[arch] via $ORIGIN/../../../_solib[arch],
        // and the second could use $ORIGIN/../_solib_[arch]. But since this is a shared
        // artifact, both are symlinks to the same place, so
        // there's no *one* RPATH setting that fits all targets involved in the sharing.
        rpathRoot += ":" + origin + cppConfig.getSolibDirectory() + "/";
        if (runtimeRpath) {
          argv.add("-Wl,-rpath," + origin + "../" + runtimeSolibName + "/");
        }
      } else {
        // For all other links, calculate the relative path from the output file to _solib_[arch]
        // (the directory where all shared libraries are stored, which resides under the blaze-bin
        // directory. In other words, given blaze-bin/my/package/binary, rpathRoot would be
        // "../../_solib_[arch]".
        if (runtimeRpath) {
          argv.add("-Wl,-rpath," + origin + Strings.repeat("../",
              linkConfig.getOutput().getRootRelativePath().segmentCount() - 1) +
              runtimeSolibName + "/");
        }

        rpathRoot += Strings.repeat("../",
            linkConfig.getOutput().getRootRelativePath().segmentCount() - 1) +
            cppConfig.getSolibDirectory() + "/";

        if (linkConfig.isNativeDeps()) {
          // We also retain the $ORIGIN/ path to solibs that are in _solib_<arch>, as opposed to
          // the package directory)
          if (runtimeRpath) {
            argv.add("-Wl,-rpath," + origin + "../" + runtimeSolibName + "/");
          }
          rpathRoot += ":" + origin;
        }
      }
    }

    boolean includeSolibDir = false;
    for (LinkerInput input : linkConfig.getLinkerInputs()) {
      if (isDynamicLibrary(input)) {
        PathFragment libDir = input.getArtifact().getExecPath().getParentDirectory();
        Preconditions.checkState(
            libDir.startsWith(solibDir),
            "Artifact '%s' is not under directory '%s'.", input.getArtifact(), solibDir);
        if (libDir.equals(solibDir)) {
          includeSolibDir = true;
        }
        addDynamicInputLinkOptions(linkConfig, input, linkerInputs, libOpts, solibDir, rpathRoot);
      } else {
        addStaticInputLinkOptions(linkConfig, input, linkerInputs);
      }
    }

    for (LinkerInput input : linkConfig.getRuntimeInputs()) {
      List<String> optionsList = globalNeedWholeArchive
          ? noWholeArchiveInputs
          : linkerInputs;

      if (isDynamicLibrary(input)) {
        PathFragment libDir = input.getArtifact().getExecPath().getParentDirectory();
        Preconditions.checkState(
            libDir.startsWith(solibDir) ||
            (runtimeSolibDir != null && libDir.equals(runtimeSolibDir)),
            "Artifact '%s' is not under directory '%s'.", input.getArtifact(), solibDir);
        if (libDir.equals(solibDir) ||
            (runtimeSolibDir != null && libDir.equals(runtimeSolibDir))) {
          includeSolibDir = true;
        }
        addDynamicInputLinkOptions(linkConfig, input, optionsList, libOpts, solibDir, rpathRoot);
      } else {
        addStaticInputLinkOptions(linkConfig, input, optionsList);
      }
    }

    if (includeSolibDir && rpathRoot != null) {
      argv.add(rpathRoot);
    }
    argv.addAll(libOpts);

    // Need to wrap static libraries with whole-archive option
    for (String option : linkerInputs) {
      if (!globalNeedWholeArchive && LINK_LIBRARY_FILETYPES.matches(option)) {
        argv.add(macosx ? "-Wl,-all_load" : "-Wl,-whole-archive");
        argv.add(option);
        argv.add(macosx ? "-Wl,-noall_load" : "-Wl,-no-whole-archive");
      } else {
        argv.add(option);
      }
    }

    if (globalNeedWholeArchive) {
      argv.add(macosx ? "-Wl,-noall_load" : "-Wl,-no-whole-archive");
      argv.addAll(noWholeArchiveInputs);
    }

    if (includeLinkopts) {
      /*
       * For compatibility with gconfig, linkopts come _after_ inputFiles.
       * This is needed to allow linkopts to contain libraries and
       * positional library-related options such as
       *    -Wl,--begin-group -lfoo -lbar -Wl,--end-group
       * or
       *    -Wl,--as-needed -lfoo -Wl,--no-as-needed
       *
       * As for the relative order of the three different flavours of linkopts
       * (global defaults, per-target linkopts, and command-line linkopts),
       * we have no idea what the right order should be, or whether this
       * code is compatible with gconfig, or if anyone cares.
       */

      argv.addAll(linkConfig.getLinkopts());
    }
  }

  /**
   * Retrieves the set of input files that should be passed to the command
   * line for the given link, returned as Artifacts.
   */
  public static Collection<Artifact> getCommandLineInputs(LinkConfiguration linkConfig) {
    Collection<Artifact> inputs = new ArrayList<>();
    for (LinkerInput input : linkConfig.getLinkerInputs()) {
      if (input.containsObjectFiles()) {
        Iterables.addAll(inputs, input.getObjectFiles());
      } else {
        inputs.add(input.getArtifact());
      }
    }
    return inputs;
  }

  /**
   * Returns the library identifier of an artifact: a string that is different for different
   * libraries, but is the same for the shared, static and pic versions of the same library.
   */
  public static String libraryIdentifierOf(Artifact libraryArtifact) {
    String name = libraryArtifact.getRootRelativePath().getPathString();
    String basename = FileSystemUtils.removeExtension(name);
    // Need to special-case file types with double extension.
    return name.endsWith(".pic.a") ? FileSystemUtils.removeExtension(basename) :
           name.endsWith(".nopic.a") ? FileSystemUtils.removeExtension(basename) :
           name.endsWith(".pic.lo") ? FileSystemUtils.removeExtension(basename) :
           basename;
  }

  /**
   * Adds command-line options for a dynamic library input file into
   * options and libOpts.
   */
  private static void addDynamicInputLinkOptions(LinkConfiguration linkConfig, LinkerInput input,
      List<String> options, Set<String> libOpts, PathFragment solibDir, String rpathRoot) {
    Preconditions.checkState(isDynamicLibrary(input));
    Preconditions.checkState(
        !useStartEndLib(input, CppHelper.archiveType(linkConfig.getConfiguration())));

    Artifact inputArtifact = input.getArtifact();
    PathFragment libDir = inputArtifact.getExecPath().getParentDirectory();
    PathFragment runtimeSolibDir = linkConfig.getRuntimeSolibDir();
    if (rpathRoot != null &&
        !libDir.equals(solibDir) &&
        (runtimeSolibDir == null || !runtimeSolibDir.equals(libDir))) {
      String dotdots = "";
      PathFragment commonParent = solibDir;
      while (!libDir.startsWith(commonParent)) {
        dotdots += "../";
        commonParent = commonParent.getParentDirectory();
      }

      libOpts.add(rpathRoot + dotdots + libDir.relativeTo(commonParent).getPathString());
    }

    libOpts.add("-L" + inputArtifact.getExecPath().getParentDirectory().getPathString());

    String name = inputArtifact.getFilename();
    if (CppFileTypes.SHARED_LIBRARY.matches(name)) {
      String libName = name.replaceAll("(^lib|\\.so$)", "");
      options.add("-l" + libName);
    } else {
      // Interface shared objects have a non-standard extension
      // that the linker won't be able to find.  So use the
      // filename directly rather than a -l option.  Since the
      // library has an SONAME attribute, this will work fine.
      options.add(inputArtifact.getExecPathString());
    }
  }

  /**
   * Adds command-line options for a static library or non-library input
   * into options.
   */
  private static void addStaticInputLinkOptions(LinkConfiguration linkConfig, LinkerInput input,
      List<String> options) {
    Preconditions.checkState(!isDynamicLibrary(input));

    // start-lib/end-lib library: adds its input object files.
    if (useStartEndLib(input, CppHelper.archiveType(linkConfig.getConfiguration()))) {
      Iterable<Artifact> archiveMembers = input.getObjectFiles();
      if (!Iterables.isEmpty(archiveMembers)) {
        options.add("-Wl,--start-lib");
        for (Artifact member : archiveMembers) {
          options.add(member.getExecPathString());
        }
        options.add("-Wl,--end-lib");
      }
    // For anything else, add the input directly.
    } else {
      Artifact inputArtifact = input.getArtifact();
      if (input.isFake()) {
        options.add(FAKE_OBJECT_PREFIX + inputArtifact.getExecPathString());
      } else {
        options.add(inputArtifact.getExecPathString());
      }
    }
  }

}
