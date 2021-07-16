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

import com.google.devtools.build.lib.util.FileTypeSet;

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

  public static final FileTypeSet ONLY_SHARED_LIBRARY_FILETYPES =
      FileTypeSet.of(CppFileTypes.SHARED_LIBRARY, CppFileTypes.VERSIONED_SHARED_LIBRARY);

  public static final FileTypeSet ONLY_INTERFACE_LIBRARY_FILETYPES =
      FileTypeSet.of(CppFileTypes.INTERFACE_SHARED_LIBRARY);

  public static final FileTypeSet ARCHIVE_LIBRARY_FILETYPES =
      FileTypeSet.of(
          CppFileTypes.ARCHIVE,
          CppFileTypes.PIC_ARCHIVE,
          CppFileTypes.ALWAYS_LINK_LIBRARY,
          CppFileTypes.ALWAYS_LINK_PIC_LIBRARY,
          CppFileTypes.RUST_RLIB);

  public static final FileTypeSet ARCHIVE_FILETYPES =
      FileTypeSet.of(CppFileTypes.ARCHIVE, CppFileTypes.PIC_ARCHIVE, CppFileTypes.RUST_RLIB);

  public static final FileTypeSet LINK_LIBRARY_FILETYPES = FileTypeSet.of(
      CppFileTypes.ALWAYS_LINK_LIBRARY,
      CppFileTypes.ALWAYS_LINK_PIC_LIBRARY);

  /** The set of object files */
  public static final FileTypeSet OBJECT_FILETYPES =
      FileTypeSet.of(
          CppFileTypes.OBJECT_FILE, CppFileTypes.PIC_OBJECT_FILE, CppFileTypes.CLIF_OUTPUT_PROTO);

  /**
   * Whether a particular link target requires PIC code.
   */
  public enum Picness {
    PIC,
    NOPIC
  }

  /** Whether a particular link target linked in statically or dynamically. */
  public enum LinkerOrArchiver {
    ARCHIVER,
    LINKER
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
        LinkerOrArchiver.ARCHIVER,
        CppActionNames.CPP_LINK_STATIC_LIBRARY,
        Picness.NOPIC,
        ArtifactCategory.STATIC_LIBRARY,
        Executable.NOT_EXECUTABLE),

    /** An objc static archive. */
    OBJC_ARCHIVE(
        LinkerOrArchiver.ARCHIVER,
        CppActionNames.OBJC_ARCHIVE,
        Picness.NOPIC,
        ArtifactCategory.STATIC_LIBRARY,
        Executable.NOT_EXECUTABLE),

    /** An objc fully linked static archive. */
    OBJC_FULLY_LINKED_ARCHIVE(
        LinkerOrArchiver.ARCHIVER,
        CppActionNames.OBJC_FULLY_LINK,
        Picness.NOPIC,
        ArtifactCategory.STATIC_LIBRARY,
        Executable.NOT_EXECUTABLE),

    /** An objc executable. */
    OBJC_EXECUTABLE(
        LinkerOrArchiver.LINKER,
        CppActionNames.OBJC_EXECUTABLE,
        Picness.NOPIC,
        ArtifactCategory.EXECUTABLE,
        Executable.EXECUTABLE),

    /** An objc executable that includes objc++/c++ source. */
    OBJCPP_EXECUTABLE(
        LinkerOrArchiver.LINKER,
        CppActionNames.OBJCPP_EXECUTABLE,
        Picness.NOPIC,
        ArtifactCategory.EXECUTABLE,
        Executable.EXECUTABLE),

    /** A static archive with .pic.o object files (compiled with -fPIC). */
    PIC_STATIC_LIBRARY(
        LinkerOrArchiver.ARCHIVER,
        CppActionNames.CPP_LINK_STATIC_LIBRARY,
        Picness.PIC,
        ArtifactCategory.STATIC_LIBRARY,
        Executable.NOT_EXECUTABLE),

    /** An interface dynamic library. */
    INTERFACE_DYNAMIC_LIBRARY(
        LinkerOrArchiver.LINKER,
        CppActionNames.CPP_LINK_DYNAMIC_LIBRARY,
        Picness.NOPIC, // Actually PIC but it's not indicated in the file name
        ArtifactCategory.INTERFACE_LIBRARY,
        Executable.NOT_EXECUTABLE),

    /** A dynamic library built from cc_library srcs. */
    NODEPS_DYNAMIC_LIBRARY(
        LinkerOrArchiver.LINKER,
        CppActionNames.CPP_LINK_NODEPS_DYNAMIC_LIBRARY,
        Picness.NOPIC, // Actually PIC but it's not indicated in the file name
        ArtifactCategory.DYNAMIC_LIBRARY,
        Executable.NOT_EXECUTABLE),
    /** A transitive dynamic library used for distribution. */
    DYNAMIC_LIBRARY(
        LinkerOrArchiver.LINKER,
        CppActionNames.CPP_LINK_DYNAMIC_LIBRARY,
        Picness.NOPIC, // Actually PIC but it's not indicated in the file name
        ArtifactCategory.DYNAMIC_LIBRARY,
        Executable.NOT_EXECUTABLE),

    /** A static archive without removal of unused object files. */
    ALWAYS_LINK_STATIC_LIBRARY(
        LinkerOrArchiver.ARCHIVER,
        CppActionNames.CPP_LINK_STATIC_LIBRARY,
        Picness.NOPIC,
        ArtifactCategory.ALWAYSLINK_STATIC_LIBRARY,
        Executable.NOT_EXECUTABLE),

    /** A PIC static archive without removal of unused object files. */
    ALWAYS_LINK_PIC_STATIC_LIBRARY(
        LinkerOrArchiver.ARCHIVER,
        CppActionNames.CPP_LINK_STATIC_LIBRARY,
        Picness.PIC,
        ArtifactCategory.ALWAYSLINK_STATIC_LIBRARY,
        Executable.NOT_EXECUTABLE),

    /** An executable binary. */
    EXECUTABLE(
        LinkerOrArchiver.LINKER,
        CppActionNames.CPP_LINK_EXECUTABLE,
        Picness.NOPIC, // Picness is not indicate in the file name
        ArtifactCategory.EXECUTABLE,
        Executable.EXECUTABLE);

    private final LinkerOrArchiver linkerOrArchiver;
    private final String actionName;
    private final ArtifactCategory linkerOutput;
    private final Picness picness;
    private final Executable executable;

    LinkTargetType(
        LinkerOrArchiver linkerOrArchiver,
        String actionName,
        Picness picness,
        ArtifactCategory linkerOutput,
        Executable executable) {
      this.linkerOrArchiver = linkerOrArchiver;
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

    public String getPicExtensionWhenApplicable() {
      return picness == Picness.PIC ? ".pic" : "";
    }

    public String getDefaultExtension() {
      return linkerOutput.getDefaultExtension();
    }

    public LinkerOrArchiver linkerOrArchiver() {
      return linkerOrArchiver;
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

    /** Returns true iff this link type is executable. */
    public boolean isExecutable() {
      return (executable == Executable.EXECUTABLE);
    }

    /** Returns true iff this link type is a transitive dynamic library. */
    public boolean isTransitiveDynamicLibrary() {
      return this == DYNAMIC_LIBRARY;
    }

    /** Returns true iff this link type is a dynamic library or transitive dynamic library. */
    public boolean isDynamicLibrary() {
      return this == NODEPS_DYNAMIC_LIBRARY || this == DYNAMIC_LIBRARY;
    }
  }

  /** The degree of "staticness" of symbol resolution during linking. */
  public enum LinkingMode {
    /**
     * Same as {@link STATIC}, but for shared libraries. Will be removed soon. This was added in
     * times when we couldn't control linking mode flags for transitive shared libraries. Now we
     * can, so this is obsolete.
     */
    LEGACY_MOSTLY_STATIC_LIBRARIES,
    /**
     * Everything is linked statically; e.g. {@code gcc -static x.o libfoo.a libbar.a -lm}.
     * Specified by {@code -static} in linkopts. Will be removed soon. This was added in times when
     * features were not expressive enough to specify different flags for {@link STATIC} and for
     * fully static links. This is now obsolete.
     */
    LEGACY_FULLY_STATIC,
    /**
     * Link binaries statically except for system libraries (e.g. {@code gcc x.o libfoo.a libbar.a
     * -lm}).
     */
    STATIC,
    /**
     * All libraries are linked dynamically (if a dynamic version is available), e.g. {@code gcc x.o
     * libfoo.so libbar.so -lm}.
     */
    DYNAMIC,
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
}
