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

import com.google.common.collect.ImmutableList;
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

  // Filetypes that generate LLVM bitcode when -flto is specified.
  public static final FileTypeSet LTO_SOURCE =
      FileTypeSet.of(CppFileTypes.CPP_SOURCE, CppFileTypes.C_SOURCE);

  public static final FileType CPP_HEADER = FileType.of(".h", ".hh", ".hpp", ".hxx", ".inc");
  public static final FileType CPP_TEXTUAL_INCLUDE = FileType.of(".inc");

  public static final FileType PIC_PREPROCESSED_C = FileType.of(".pic.i");
  public static final FileType PREPROCESSED_C = new FileType() {
      final String ext = ".i";
      @Override
      public boolean apply(String filename) {
        return filename.endsWith(ext) && !PIC_PREPROCESSED_C.matches(filename);
      }
      @Override
      public List<String> getExtensions() {
        return ImmutableList.of(ext);
      }
    };
  public static final FileType PIC_PREPROCESSED_CPP = FileType.of(".pic.ii");
  public static final FileType PREPROCESSED_CPP = new FileType() {
      final String ext = ".ii";
      @Override
      public boolean apply(String filename) {
        return filename.endsWith(ext) && !PIC_PREPROCESSED_CPP.matches(filename);
      }
      @Override
      public List<String> getExtensions() {
        return ImmutableList.of(ext);
      }
    };

  public static final FileType ASSEMBLER_WITH_C_PREPROCESSOR = FileType.of(".S");
  public static final FileType PIC_ASSEMBLER = FileType.of(".pic.s");
  public static final FileType ASSEMBLER = new FileType() {
      final String ext = ".s";
      @Override
      public boolean apply(String filename) {
        return (filename.endsWith(ext) && !PIC_ASSEMBLER.matches(filename))
               || filename.endsWith(".asm");
      }
      @Override
      public List<String> getExtensions() {
        return ImmutableList.of(ext, ".asm");
      }
    };

  public static final FileType PIC_ARCHIVE = FileType.of(".pic.a");
  public static final FileType ARCHIVE = new FileType() {
      final String ext = ".a";
      @Override
      public boolean apply(String filename) {
        return filename.endsWith(ext) && !PIC_ARCHIVE.matches(filename);
      }
      @Override
      public List<String> getExtensions() {
        return ImmutableList.of(ext);
      }
    };

  public static final FileType ALWAYS_LINK_PIC_LIBRARY = FileType.of(".pic.lo");
  public static final FileType ALWAYS_LINK_LIBRARY = new FileType() {
      final String ext = ".lo";
      @Override
      public boolean apply(String filename) {
        return filename.endsWith(ext) && !ALWAYS_LINK_PIC_LIBRARY.matches(filename);
      }
      @Override
      public List<String> getExtensions() {
        return ImmutableList.of(ext);
      }
    };

  public static final FileType PIC_OBJECT_FILE = FileType.of(".pic.o");
  public static final FileType OBJECT_FILE = new FileType() {
      final String ext = ".o";
      @Override
      public boolean apply(String filename) {
        return filename.endsWith(ext) && !PIC_OBJECT_FILE.matches(filename);
      }
      @Override
      public List<String> getExtensions() {
        return ImmutableList.of(ext);
      }
    };


  public static final FileType SHARED_LIBRARY = FileType.of(".so", ".dylib", ".dll");
  public static final FileType INTERFACE_SHARED_LIBRARY = FileType.of(".ifso");
  public static final FileType LINKER_SCRIPT = FileType.of(".ld", ".lds", ".ldscript");
  // Matches shared libraries with version names in the extension, i.e.
  // libmylib.so.2 or libmylib.so.2.10.
  private static final Pattern VERSIONED_SHARED_LIBRARY_PATTERN =
     Pattern.compile("^.+\\.so(\\.\\d+)+$");
  public static final FileType VERSIONED_SHARED_LIBRARY = new FileType() {
      @Override
      public boolean apply(String filename) {
        // Because regex matching can be slow, we first do a quick digit check on the final
        // character before risking the full-on regex match. This should eliminate the performance
        // hit on practically every non-qualifying file type.
        if (Character.isDigit(filename.charAt(filename.length() - 1))) {
          return VERSIONED_SHARED_LIBRARY_PATTERN.matcher(filename).matches();
        } else {
          return false;
        }
      }
    };

  public static final FileType COVERAGE_NOTES = FileType.of(".gcno");
  public static final FileType COVERAGE_DATA = FileType.of(".gcda");
  public static final FileType COVERAGE_DATA_IMPORTS = FileType.of(".gcda.imports");
  public static final FileType GCC_AUTO_PROFILE = FileType.of(".afdo");
  public static final FileType LLVM_PROFILE = FileType.of(".profdata");

  public static final FileType CPP_MODULE_MAP = FileType.of(".cppmap");
  public static final FileType CPP_MODULE = FileType.of(".pcm");

  // Output of the dwp tool
  public static final FileType DEBUG_INFO_PACKAGE = FileType.of(".dwp");

  public static final FileType CLIF_INPUT_PROTO = FileType.of(".ipb");
  public static final FileType CLIF_OUTPUT_PROTO = FileType.of(".opb");

  public static final boolean mustProduceDotdFile(String source) {
    return !ASSEMBLER.matches(source)
        && !PIC_ASSEMBLER.matches(source)
        && !CLIF_INPUT_PROTO.matches(source);
  }

}
