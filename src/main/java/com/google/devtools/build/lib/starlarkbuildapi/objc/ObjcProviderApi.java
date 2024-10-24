// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.starlarkbuildapi.objc;

import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.StarlarkValue;

/**
 * An interface for an info type that provides all compiling and linking information in the
 * transitive closure of its deps that are needed for building Objective-C rules.
 */
@StarlarkBuiltin(
    name = "ObjcProvider",
    category = DocCategory.PROVIDER,
    doc = "A provider for compilation and linking of objc.")
public interface ObjcProviderApi<FileApiT extends FileApi> extends StarlarkValue {

  @StarlarkMethod(
      name = "strict_include",
      structField = true,
      doc =
          "Non-propagated include search paths specified with '-I' on the command line. Also known "
              + "as header search paths (and distinct from <em>user</em> header search paths).")
  default Depset strictIncludeForStarlark() {
    throw new UnsupportedOperationException(); // just for docs
  }

  @StarlarkMethod(
      name = "j2objc_library",
      structField = true,
      doc = "Static libraries that are built from J2ObjC-translated Java code.")
  default Depset /*<FileApiT>*/ j2objcLibrary() {
    throw new UnsupportedOperationException(); // just for docs
  }

  @StarlarkMethod(
      name = "module_map",
      structField = true,
      doc = "Clang module maps, used to enforce proper use of private header files.")
  default Depset /*<FileApiT>*/ moduleMap() {
    throw new UnsupportedOperationException(); // just for docs
  }

  @StarlarkMethod(
      name = "direct_module_maps",
      structField = true,
      doc =
          "Module map files from this target directly (no transitive module maps). "
              + "Used to enforce proper use of private header files and for Swift compilation.")
  default Sequence<FileApiT> directModuleMaps() {
    throw new UnsupportedOperationException(); // just for docs
  }

  @StarlarkMethod(name = "source", structField = true, doc = "All transitive source files.")
  default Depset /*<FileApiT>*/ sourceForStarlark() {
    throw new UnsupportedOperationException(); // just for docs
  }

  @StarlarkMethod(
      name = "direct_sources",
      structField = true,
      doc =
          "All direct source files from this target (no transitive files), "
              + "including any headers in the 'srcs' attribute.")
  default Sequence<FileApiT> directSources() {
    throw new UnsupportedOperationException(); // just for docs
  }

  @StarlarkMethod(
      name = "umbrella_header",
      structField = true,
      doc =
          "Clang umbrella header. Public headers are #included in umbrella headers to be "
              + "compatible with J2ObjC segmented headers.")
  default Depset /*<FileApiT>*/ umbrellaHeader() {
    throw new UnsupportedOperationException(); // just for docs
  }
}
