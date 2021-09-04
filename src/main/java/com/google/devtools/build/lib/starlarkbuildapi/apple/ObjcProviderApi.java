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

package com.google.devtools.build.lib.starlarkbuildapi.apple;

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

  @StarlarkMethod(name = "cc_library", documented = false, structField = true)
  Depset /*<LibraryToLink>*/ ccLibrariesForStarlark();

  @StarlarkMethod(name = "linkstamp", documented = false, structField = true)
  Depset /*<Linkstamp>*/ linkstampForstarlark();

  @StarlarkMethod(
      name = "dynamic_framework_file",
      structField = true,
      doc =
          "The library files in .framework directories belonging to a dynamically linked "
              + "framework.")
  Depset /*<FileApiT>*/ dynamicFrameworkFileForStarlark();

  @StarlarkMethod(
      name = "force_load_library",
      structField = true,
      doc = "Libraries to load with -force_load.")
  Depset /*<FileApiT>*/ forceLoadLibrary();

  @StarlarkMethod(
      name = "direct_headers",
      structField = true,
      doc =
          "Public header files from this target directly (no transitive headers). "
              + "These are mostly headers from the 'hdrs' attribute.")
  Sequence<FileApiT> directHeaders();

  @StarlarkMethod(
      name = "imported_library",
      structField = true,
      doc = "Imported precompiled static libraries (.a files) to be linked into the binary.")
  Depset /*<FileApiT>*/ importedLibrary();

  @StarlarkMethod(
      name = "strict_include",
      structField = true,
      doc =
          "Non-propagated include search paths specified with '-I' on the command line. Also known "
              + "as header search paths (and distinct from <em>user</em> header search paths).")
  Depset strictIncludeForStarlark();

  @StarlarkMethod(
      name = "j2objc_library",
      structField = true,
      doc = "Static libraries that are built from J2ObjC-translated Java code.")
  Depset /*<FileApiT>*/ j2objcLibrary();

  @StarlarkMethod(
      name = "jre_library",
      structField = true,
      doc = "J2ObjC JRE emulation libraries and their dependencies.")
  Depset /*<FileApiT>*/ jreLibrary();

  @StarlarkMethod(
      name = "library",
      structField = true,
      doc = "Library (.a) files compiled by dependencies of the current target.")
  Depset /*<FileApiT>*/ library();

  @StarlarkMethod(
      name = "link_inputs",
      structField = true,
      doc =
          "Link time artifacts from dependencies that do not fall into any other category such as"
              + " libraries or archives. This catch-all provides a way to add arbitrary data (e.g."
              + " Swift AST files) to the linker. The rule that adds these is also responsible to"
              + " add the necessary linker flags to 'linkopt'.")
  Depset /*<FileApiT>*/ linkInputs();

  @StarlarkMethod(name = "linkopt", structField = true, doc = "Linking options.")
  Depset /*<String>*/ linkopt();

  @StarlarkMethod(
      name = "module_map",
      structField = true,
      doc = "Clang module maps, used to enforce proper use of private header files.")
  Depset /*<FileApiT>*/ moduleMap();

  @StarlarkMethod(
      name = "direct_module_maps",
      structField = true,
      doc =
          "Module map files from this target directly (no transitive module maps). "
              + "Used to enforce proper use of private header files and for Swift compilation.")
  Sequence<FileApiT> directModuleMaps();

  @StarlarkMethod(
      name = "sdk_dylib",
      structField = true,
      doc = "Names of SDK .dylib libraries to link with. For instance, 'libz' or 'libarchive'.")
  Depset /*<String>*/ sdkDylib();

  @StarlarkMethod(
      name = "sdk_framework",
      structField = true,
      doc = "Names of SDK frameworks to link with (e.g. 'AddressBook', 'QuartzCore').")
  Depset sdkFramework();

  @StarlarkMethod(name = "source", structField = true, doc = "All transitive source files.")
  Depset /*<FileApiT>*/ sourceForStarlark();

  @StarlarkMethod(
      name = "direct_sources",
      structField = true,
      doc =
          "All direct source files from this target (no transitive files), "
              + "including any headers in the 'srcs' attribute.")
  Sequence<FileApiT> directSources();

  @StarlarkMethod(
      name = "static_framework_file",
      structField = true,
      doc = "The library files in .framework directories that should be statically linked.")
  Depset /*<FileApiT>*/ staticFrameworkFileForStarlark();

  @StarlarkMethod(
      name = "umbrella_header",
      structField = true,
      doc =
          "Clang umbrella header. Public headers are #included in umbrella headers to be "
              + "compatible with J2ObjC segmented headers.")
  Depset /*<FileApiT>*/ umbrellaHeader();

  @StarlarkMethod(
      name = "weak_sdk_framework",
      structField = true,
      doc =
          "Names of SDK frameworks to weakly link with. For instance, 'MediaAccessibility'. "
              + "In difference to regularly linked SDK frameworks, symbols from weakly linked "
              + "frameworks do not cause an error if they are not present at runtime.")
  Depset weakSdkFramework();

  @StarlarkMethod(
      name = "dynamic_framework_names",
      structField = true,
      doc = "Returns all names of dynamic frameworks in this provider.")
  Depset /*<String>*/ dynamicFrameworkNamesForStarlark();

  @StarlarkMethod(
      name = "dynamic_framework_paths",
      structField = true,
      doc = "Returns all framework paths to dynamic frameworks in this provider.")
  Depset /*<String>*/ dynamicFrameworkPathsForStarlark();

  @StarlarkMethod(
      name = "static_framework_names",
      structField = true,
      doc = "Returns all names of static frameworks in this provider.")
  Depset /*<String>*/ staticFrameworkNamesForStarlark();

  @StarlarkMethod(
      name = "static_framework_paths",
      structField = true,
      doc = "Returns all framework paths to static frameworks in this provider.")
  Depset /*<String>*/ staticFrameworkPathsForStarlark();
}
