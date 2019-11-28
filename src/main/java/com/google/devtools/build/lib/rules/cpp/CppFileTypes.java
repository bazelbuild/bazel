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

import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.util.FileTypeSet;
import java.util.List;
import java.util.regex.Pattern;

/**
 * C++-related file type definitions.
 */
public final class CppFileTypes {
  public static final FileType CPP_SOURCE = FileType.of(".cc", ".cpp", ".cxx", ".c++", ".C");
  public static final FileType C_SOURCE = FileType.of(".c");
  public static final FileType OBJC_SOURCE = FileType.of(".m");
  public static final FileType OBJCPP_SOURCE = FileType.of(".mm");
  public static final FileType CLIF_INPUT_PROTO = FileType.of(".ipb");
  public static final FileType CLIF_OUTPUT_PROTO = FileType.of(".opb");

  public static final FileTypeSet ALL_C_CLASS_SOURCE =
      FileTypeSet.of(
          CppFileTypes.CPP_SOURCE,
          CppFileTypes.C_SOURCE,
          CppFileTypes.OBJCPP_SOURCE,
          CppFileTypes.OBJC_SOURCE,
          CppFileTypes.CLIF_INPUT_PROTO);

  // Filetypes that generate LLVM bitcode when -flto is specified.
  public static final FileTypeSet LTO_SOURCE =
      FileTypeSet.of(CppFileTypes.CPP_SOURCE, CppFileTypes.C_SOURCE);

  public static final FileType CPP_HEADER =
      FileType.of(
          ".h", ".hh", ".hpp", ".ipp", ".hxx", ".h++", ".inc", ".inl", ".tlh", ".tli", ".H");
  public static final FileType PCH = FileType.of(".pch");
  public static final FileTypeSet OBJC_HEADER = FileTypeSet.of(CPP_HEADER, PCH);

  public static final FileType CPP_TEXTUAL_INCLUDE = FileType.of(".inc");

  public static final FileType PIC_PREPROCESSED_C = FileType.of(".pic.i");
  public static final FileType PREPROCESSED_C =
      new FileType() {
        final String ext = ".i";

        @Override
        public boolean apply(String path) {
          return path.endsWith(ext) && !PIC_PREPROCESSED_C.matches(path);
        }

        @Override
        public List<String> getExtensions() {
          return ImmutableList.of(ext);
        }
      };
  public static final FileType PIC_PREPROCESSED_CPP = FileType.of(".pic.ii");
  public static final FileType PREPROCESSED_CPP =
      new FileType() {
        final String ext = ".ii";

        @Override
        public boolean apply(String path) {
          return path.endsWith(ext) && !PIC_PREPROCESSED_CPP.matches(path);
        }

        @Override
        public List<String> getExtensions() {
          return ImmutableList.of(ext);
        }
      };

  public static final FileType ASSEMBLER_WITH_C_PREPROCESSOR = FileType.of(".S");
  public static final FileType PIC_ASSEMBLER = FileType.of(".pic.s");
  public static final FileType ASSEMBLER =
      new FileType() {
        final String ext = ".s";

        @Override
        public boolean apply(String path) {
          return (path.endsWith(ext) && !PIC_ASSEMBLER.matches(path)) || path.endsWith(".asm");
        }

        @Override
        public List<String> getExtensions() {
          return ImmutableList.of(ext, ".asm");
        }
      };

  public static final FileType PIC_ARCHIVE = FileType.of(".pic.a");
  public static final FileType ARCHIVE =
      new FileType() {
        final List<String> extensions = ImmutableList.of(".a", ".lib");

        @Override
        public boolean apply(String path) {
          if (PIC_ARCHIVE.matches(path)
              || ALWAYS_LINK_LIBRARY.matches(path)
              || path.endsWith(".if.lib")) {
            return false;
          }
          for (String ext : extensions) {
            if (path.endsWith(ext)) {
              return true;
            }
          }
          return false;
        }

        @Override
        public List<String> getExtensions() {
          return ImmutableList.copyOf(extensions);
        }
      };

  public static final FileType ALWAYS_LINK_PIC_LIBRARY = FileType.of(".pic.lo");
  public static final FileType ALWAYS_LINK_LIBRARY =
      new FileType() {
        final String ext = ".lo";

        @Override
        public boolean apply(String path) {
          return (path.endsWith(ext) && !ALWAYS_LINK_PIC_LIBRARY.matches(path))
              || path.endsWith(".lo.lib");
        }

        @Override
        public List<String> getExtensions() {
          return ImmutableList.of(ext, ".lo.lib");
        }
      };

  public static final FileType PIC_OBJECT_FILE = FileType.of(".pic.o");
  public static final FileType OBJECT_FILE =
      new FileType() {
        final String ext = ".o";

        @Override
        public boolean apply(String path) {
          return (path.endsWith(ext) && !PIC_OBJECT_FILE.matches(path)) || path.endsWith(".obj");
        }

        @Override
        public List<String> getExtensions() {
          return ImmutableList.of(ext, ".obj");
        }
      };

  // Minimized bitcode file emitted by the ThinLTO compile step and used just for LTO indexing.
  public static final FileType LTO_INDEXING_OBJECT_FILE = FileType.of(".indexing.o");

  public static final FileType SHARED_LIBRARY = FileType.of(".so", ".dylib", ".dll");
  // Unix shared libraries can be passed to linker, but not .dll on Windows
  public static final FileType UNIX_SHARED_LIBRARY = FileType.of(".so", ".dylib");
  public static final FileType INTERFACE_SHARED_LIBRARY = FileType.of(".ifso", ".tbd", ".lib");
  public static final FileType LINKER_SCRIPT = FileType.of(".ld", ".lds", ".ldscript");

  // Windows DEF file: https://msdn.microsoft.com/en-us/library/28d6s79h.aspx
  public static final FileType WINDOWS_DEF_FILE = FileType.of(".def");

  // Matches shared libraries with version names in the extension, i.e.
  // libmylib.so.2 or libmylib.so.2.10.
  private static final Pattern VERSIONED_SHARED_LIBRARY_PATTERN =
     Pattern.compile("^.+\\.so(\\.\\d+)+$");
  public static final FileType VERSIONED_SHARED_LIBRARY =
      new FileType() {
        @Override
        public boolean apply(String path) {
          // Because regex matching can be slow, we first do a quick digit check on the final
          // character before risking the full-on regex match. This should eliminate the performance
          // hit on practically every non-qualifying file type.
          if (Character.isDigit(path.charAt(path.length() - 1))) {
            return VERSIONED_SHARED_LIBRARY_PATTERN.matcher(path).matches();
          } else {
            return false;
          }
        }
      };

  public static final FileType COVERAGE_NOTES = FileType.of(".gcno");
  public static final FileType GCC_AUTO_PROFILE = FileType.of(".afdo");
  public static final FileType XBINARY_PROFILE = FileType.of(".xfdo");
  public static final FileType LLVM_PROFILE = FileType.of(".profdata");
  public static final FileType LLVM_PROFILE_RAW = FileType.of(".profraw");
  public static final FileType LLVM_PROFILE_ZIP = FileType.of(".zip");

  public static final FileType CPP_MODULE_MAP = FileType.of(".cppmap");
  public static final FileType CPP_MODULE = FileType.of(".pcm");
  public static final FileType OBJC_MODULE_MAP = FileType.of("module.modulemap");

  /** Predicate that matches all artifacts that can be used in an objc Clang module map. */
  public static final Predicate<Artifact> MODULE_MAP_HEADER =
      artifact -> {
        if (artifact.isTreeArtifact()) {
          // Tree artifact is basically a directory, which does not have any information about
          // the contained files and their extensions. Here we assume the passed in tree artifact
          // contains proper header files with .h extension.
          return true;
        } else {
          // The current clang (clang-600.0.57) on Darwin doesn't support 'textual', so we can't
          // have '.inc' files in the module map (since they're implictly textual).
          // TODO(bazel-team): Use HEADERS file type once clang-700 is the base clang we support.
          return artifact.getFilename().endsWith(".h");
        }
      };

  public static final boolean headerDiscoveryRequired(Artifact source) {
    String fileName = source.getFilename();
    return !ASSEMBLER.matches(fileName)
        && !PIC_ASSEMBLER.matches(fileName)
        && !CPP_MODULE.matches(fileName);
  }

}
