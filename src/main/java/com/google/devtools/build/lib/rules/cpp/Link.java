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

import com.google.common.collect.AbstractIterator;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.collect.CollectionUtils;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.rules.cpp.LinkerInputs.LibraryToLink;
import com.google.devtools.build.lib.util.FileTypeSet;
import java.util.Iterator;

/**
 * Utility types and methods for generating command lines for the linker, given
 * a CppLinkAction or LinkConfiguration.
 *
 * <p>The linker commands, e.g. "ar", may not be functional, i.e.
 * they may mutate the output file rather than overwriting it.
 * To avoid this, we need to delete the output file before invoking the
 * command.  But that is not done by this class; deleting the output
 * file is the responsibility of the classes implementing CppLinkActionContext.
 */
public abstract class Link {

  /** Categories of link action that must be defined with action_configs in any toolchain. */
  static final ImmutableList<LinkTargetType> MANDATORY_LINK_TARGET_TYPES =
      ImmutableList.of(
          LinkTargetType.STATIC_LIBRARY,
          LinkTargetType.PIC_STATIC_LIBRARY,
          LinkTargetType.ALWAYS_LINK_STATIC_LIBRARY,
          LinkTargetType.ALWAYS_LINK_PIC_STATIC_LIBRARY,
          LinkTargetType.DYNAMIC_LIBRARY,
          LinkTargetType.EXECUTABLE,
          LinkTargetType.INTERFACE_DYNAMIC_LIBRARY);

  private Link() {} // uninstantiable

  /**
   * These file are supposed to be added using {@code addLibrary()} calls to {@link CppLinkAction}
   * but will never be expanded to their constituent {@code .o} files. {@link CppLinkAction} checks
   * that these files are never added as non-libraries.
   */
  public static final FileTypeSet SHARED_LIBRARY_FILETYPES = FileTypeSet.of(
      CppFileTypes.SHARED_LIBRARY,
      CppFileTypes.VERSIONED_SHARED_LIBRARY,
      CppFileTypes.INTERFACE_SHARED_LIBRARY);

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
   * Whether a particular link target requires PIC code.
   */
  public enum Picness {
    PIC,
    NOPIC
  }

  /**
   * Whether a particular link target linked in statically or dynamically.
   */
  public enum Staticness {
    STATIC,
    DYNAMIC
  }
  
  /**
   * Whether a particular link target is executable.
   */
  public enum Executable {
    EXECUTABLE,
    NOT_EXECUTABLE
  }

  /**
   * Types of ELF files that can be created by the linker (.a, .so, .lo,
   * executable).
   */
  public enum LinkTargetType {
    /** A normal static archive. */
    STATIC_LIBRARY(
        ".a",
        Staticness.STATIC,
        "c++-link-static-library",
        Picness.NOPIC,
        ArtifactCategory.STATIC_LIBRARY,
        Executable.NOT_EXECUTABLE),
    
    /** An objc static archive. */
    OBJC_ARCHIVE(
        ".a", 
        Staticness.STATIC, 
        "objc-archive", 
        Picness.NOPIC,
        ArtifactCategory.STATIC_LIBRARY,
        Executable.NOT_EXECUTABLE),
    
    /** An objc fully linked static archive. */
    OBJC_FULLY_LINKED_ARCHIVE(
        ".a",
        Staticness.STATIC,
        "objc-fully-link",
        Picness.NOPIC,
        ArtifactCategory.STATIC_LIBRARY,
        Executable.NOT_EXECUTABLE),

    /** An objc executable. */
    OBJC_EXECUTABLE(
        "",
        Staticness.DYNAMIC,
        "objc-executable",
        Picness.NOPIC,
        ArtifactCategory.EXECUTABLE,
        Executable.EXECUTABLE),

    /** An objc executable that includes objc++/c++ source. */
    OBJCPP_EXECUTABLE(
        "",
        Staticness.DYNAMIC,
        "objc++-executable",
        Picness.NOPIC,
        ArtifactCategory.EXECUTABLE,
        Executable.EXECUTABLE),

    /** A static archive with .pic.o object files (compiled with -fPIC). */
    PIC_STATIC_LIBRARY(
        ".pic.a",
        Staticness.STATIC,
        "c++-link-pic-static-library",
        Picness.PIC,
        ArtifactCategory.STATIC_LIBRARY,
        Executable.NOT_EXECUTABLE),

    /** An interface dynamic library. */
    INTERFACE_DYNAMIC_LIBRARY(
        ".ifso",
        Staticness.DYNAMIC,
        "c++-link-interface-dynamic-library",
        Picness.NOPIC,  // Actually PIC but it's not indicated in the file name
        ArtifactCategory.INTERFACE_LIBRARY,
        Executable.NOT_EXECUTABLE),

    /** A dynamic library. */
    DYNAMIC_LIBRARY(
        ".so",
        Staticness.DYNAMIC,
        "c++-link-dynamic-library",
        Picness.NOPIC,  // Actually PIC but it's not indicated in the file name
        ArtifactCategory.DYNAMIC_LIBRARY,
        Executable.NOT_EXECUTABLE),

    /** A static archive without removal of unused object files. */
    ALWAYS_LINK_STATIC_LIBRARY(
        ".lo",
        Staticness.STATIC,
        "c++-link-alwayslink-static-library",
        Picness.NOPIC,
        ArtifactCategory.ALWAYSLINK_STATIC_LIBRARY,
        Executable.NOT_EXECUTABLE),

    /** A PIC static archive without removal of unused object files. */
    ALWAYS_LINK_PIC_STATIC_LIBRARY(
        ".pic.lo",
        Staticness.STATIC,
        "c++-link-alwayslink-pic-static-library",
        Picness.PIC,
        ArtifactCategory.ALWAYSLINK_STATIC_LIBRARY,
        Executable.NOT_EXECUTABLE),

    /** An executable binary. */
    EXECUTABLE(
        "",
        Staticness.DYNAMIC,
        "c++-link-executable",
        Picness.NOPIC,  // Picness is not indicate in the file name
        ArtifactCategory.EXECUTABLE,
        Executable.EXECUTABLE);

    private final String extension;
    private final Staticness staticness;
    private final String actionName;
    private final ArtifactCategory linkerOutput;
    private final Picness picness;
    private final Executable executable;

    LinkTargetType(
        String extension,
        Staticness staticness,
        String actionName,
        Picness picness,
        ArtifactCategory linkerOutput,
        Executable executable) {
      this.extension = extension;
      this.staticness = staticness;
      this.actionName = actionName;
      this.linkerOutput = linkerOutput;
      this.picness = picness;
      this.executable = executable;
    }

    /**
     * Returns whether the name of the output file should denote that the code in the file is PIC.
     */
    public Picness picness() {
      return picness;
    }

    public String getExtension() {
      return extension;
    }

    public Staticness staticness() {
      return staticness;
    }
    
    /** Returns an {@code ArtifactCategory} identifying the artifact type this link action emits. */
    public ArtifactCategory getLinkerOutput() {
      return linkerOutput;
    }

    /**
     * The name of a link action with this LinkTargetType, for the purpose of crosstool feature
     * selection.
     */
    public String getActionName() {
      return actionName;
    }
    
    /** Returns true iff this link type is executable */
    public boolean isExecutable() {
      return (executable == Executable.EXECUTABLE);
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
   * How to pass archives to the linker on the command line.
   */
  public enum ArchiveType {
    REGULAR,        // Put the archive itself on the linker command line.
    START_END_LIB   // Put the object files enclosed by --start-lib / --end-lib on the command line
  }

  static boolean useStartEndLib(LinkerInput linkerInput, ArchiveType archiveType) {
    // TODO(bazel-team): Figure out if PicArchives are actually used. For it to be used, both
    // linkingStatically and linkShared must me true, we must be in opt mode and cpu has to be k8.
    return archiveType == ArchiveType.START_END_LIB
        && (linkerInput.getArtifactCategory() == ArtifactCategory.STATIC_LIBRARY
            || linkerInput.getArtifactCategory() == ArtifactCategory.ALWAYSLINK_STATIC_LIBRARY)
        && linkerInput.containsObjectFiles();
  }

  /**
   * Replace always used archives with its members. This is used to build the linker cmd line.
   */
  public static Iterable<LinkerInput> mergeInputsCmdLine(NestedSet<LibraryToLink> inputs,
      boolean globalNeedWholeArchive, ArchiveType archiveType) {
    return new FilterMembersForLinkIterable(inputs, globalNeedWholeArchive, archiveType, false);
  }

  /**
   * Add in any object files which are implicitly named as inputs by the linker.
   */
  public static Iterable<LinkerInput> mergeInputsDependencies(NestedSet<LibraryToLink> inputs,
      boolean globalNeedWholeArchive, ArchiveType archiveType) {
    return new FilterMembersForLinkIterable(inputs, globalNeedWholeArchive, archiveType, true);
  }

  /**
   * On the fly implementation to filter the members.
   */
  private static final class FilterMembersForLinkIterable implements Iterable<LinkerInput> {
    private final boolean globalNeedWholeArchive;
    private final ArchiveType archiveType;
    private final boolean deps;

    private final Iterable<LibraryToLink> inputs;

    private FilterMembersForLinkIterable(Iterable<LibraryToLink> inputs,
        boolean globalNeedWholeArchive, ArchiveType archiveType, boolean deps) {
      this.globalNeedWholeArchive = globalNeedWholeArchive;
      this.archiveType = archiveType;
      this.deps = deps;
      this.inputs = CollectionUtils.makeImmutable(inputs);
    }

    @Override
    public Iterator<LinkerInput> iterator() {
      return new FilterMembersForLinkIterator(inputs.iterator(), globalNeedWholeArchive,
          archiveType, deps);
    }
  }

  /**
   * On the fly implementation to filter the members.
   */
  private static final class FilterMembersForLinkIterator extends AbstractIterator<LinkerInput> {
    private final boolean globalNeedWholeArchive;
    private final ArchiveType archiveType;
    private final boolean deps;

    private final Iterator<LibraryToLink> inputs;
    private Iterator<LinkerInput> delayList = ImmutableList.<LinkerInput>of().iterator();

    private FilterMembersForLinkIterator(Iterator<LibraryToLink> inputs,
        boolean globalNeedWholeArchive, ArchiveType archiveType, boolean deps) {
      this.globalNeedWholeArchive = globalNeedWholeArchive;
      this.archiveType = archiveType;
      this.deps = deps;
      this.inputs = inputs;
    }

    @Override
    protected LinkerInput computeNext() {
      if (delayList.hasNext()) {
        return delayList.next();
      }

      while (inputs.hasNext()) {
        LibraryToLink inputLibrary = inputs.next();

        // True if the linker might use the members of this file, i.e., if the file is a thin or
        // start_end_lib archive (aka static library). Also check if the library contains object
        // files - otherwise getObjectFiles returns null, which would lead to an NPE in
        // simpleLinkerInputs.
        boolean needMembersForLink = archiveType != ArchiveType.REGULAR
            && (inputLibrary.getArtifactCategory() == ArtifactCategory.STATIC_LIBRARY
                || inputLibrary.getArtifactCategory() == ArtifactCategory.ALWAYSLINK_STATIC_LIBRARY)
            && inputLibrary.containsObjectFiles();

        // True if we will pass the members instead of the original archive.
        boolean passMembersToLinkCmd = needMembersForLink && (globalNeedWholeArchive
            || inputLibrary.getArtifactCategory() == ArtifactCategory.ALWAYSLINK_STATIC_LIBRARY);

        // If deps is false (when computing the inputs to be passed on the command line), then it's
        // an if-then-else, i.e., the passMembersToLinkCmd flag decides whether to pass the object
        // files or the archive itself. This flag in turn is based on whether the archives are fat
        // or not (thin archives or start_end_lib) - we never expand fat archives, but we do expand
        // non-fat archives if we need whole-archives for the entire link, or for the specific
        // library (i.e., if alwayslink=1).
        //
        // If deps is true (when computing the inputs to be passed to the action as inputs), then it
        // becomes more complicated. We always need to pass the members for thin and start_end_lib
        // archives (needMembersForLink). And we _also_ need to pass the archive file itself unless
        // it's a start_end_lib archive (unless it's an alwayslink library).

        // A note about ordering: the order in which the object files and the library are returned
        // does not currently matter - this code results in the library returned first, and the
        // object files returned after, but only if both are returned, which can only happen if
        // deps is true, in which case this code only computes the list of inputs for the link
        // action (so the order isn't critical).
        if (passMembersToLinkCmd || (deps && needMembersForLink)) {
          delayList = LinkerInputs
              .simpleLinkerInputs(inputLibrary.getObjectFiles(), ArtifactCategory.OBJECT_FILE)
              .iterator();
        }

        if (!(passMembersToLinkCmd || (deps && useStartEndLib(inputLibrary, archiveType)))) {
          return inputLibrary;
        }

        if (delayList.hasNext()) {
          return delayList.next();
        }
      }
      return endOfData();
    }
  }
}
