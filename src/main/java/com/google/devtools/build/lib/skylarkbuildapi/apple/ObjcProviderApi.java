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

package com.google.devtools.build.lib.skylarkbuildapi.apple;

import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.CcCompilationContextApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.Depset;
import com.google.devtools.build.lib.syntax.Sequence;
import com.google.devtools.build.lib.syntax.StarlarkValue;

/**
 * An interface for an info type that provides all compiling and linking information in the
 * transitive closure of its deps that are needed for building Objective-C rules.
 */
@SkylarkModule(
    name = "ObjcProvider",
    category = SkylarkModuleCategory.PROVIDER,
    doc = "A provider for compilation and linking of objc.")
public interface ObjcProviderApi<FileApiT extends FileApi> extends StarlarkValue {

  @SkylarkCallable(
      name = "define",
      structField = true,
      doc =
          "A set of strings from 'defines' attributes. These are to be passed as '-D' flags to "
              + "all invocations of the compiler for this target and all depending targets.")
  Depset /*<String>*/ defineForStarlark();

  @SkylarkCallable(
      name = "dynamic_framework_file",
      structField = true,
      doc =
          "The library files in .framework directories belonging to a dynamically linked "
              + "framework.")
  Depset /*<FileApiT>*/ dynamicFrameworkFileForStarlark();

  @SkylarkCallable(
      name = "exported_debug_artifacts",
      structField = true,
      doc = "Debug files that should be exported by the top-level target.")
  Depset /*<FileApiT>*/ exportedDebugArtifacts();

  @SkylarkCallable(
      name = "framework_search_path_only",
      structField = true,
      doc =
          "Exec paths of .framework directories corresponding to frameworks to include "
              + "in search paths, but not to link.")
  Depset /*<String>*/ frameworkIncludeForStarlark();

  @SkylarkCallable(
      name = "force_load_library",
      structField = true,
      doc = "Libraries to load with -force_load.")
  Depset /*<FileApiT>*/ forceLoadLibrary();

  @SkylarkCallable(
      name = "header",
      structField = true,
      doc = "All header files. These may be either public or private headers.")
  Depset /*<FileApiT>*/ headerForStarlark();

  @SkylarkCallable(
      name = "direct_headers",
      structField = true,
      doc =
          "Header files from this target directly (no transitive headers). "
              + "These may be either public or private headers.")
  Sequence<FileApiT> directHeaders();

  @SkylarkCallable(
      name = "imported_library",
      structField = true,
      doc = "Imported precompiled static libraries (.a files) to be linked into the binary.")
  Depset /*<FileApiT>*/ importedLibrary();

  @SkylarkCallable(
      name = "include",
      structField = true,
      doc =
          "Include search paths specified with '-I' on the command line. Also known as "
              + "header search paths (and distinct from <em>user</em> header search paths).")
  Depset includeForStarlark();

  @SkylarkCallable(
      name = "strict_include",
      structField = true,
      doc =
          "Non-propagated include search paths specified with '-I' on the command line. Also known "
              + "as header search paths (and distinct from <em>user</em> header search paths).")
  Depset strictIncludeForStarlark();

  @SkylarkCallable(
      name = "include_system",
      structField = true,
      doc = "System include search paths (typically specified with -isystem).")
  Depset systemIncludeForStarlark();

  @SkylarkCallable(
      name = "iquote",
      structField = true,
      doc = "User header search paths (typically specified with -iquote).")
  Depset quoteIncludeForStarlark();

  @SkylarkCallable(
      name = "j2objc_library",
      structField = true,
      doc = "Static libraries that are built from J2ObjC-translated Java code.")
  Depset /*<FileApiT>*/ j2objcLibrary();

  @SkylarkCallable(
      name = "jre_library",
      structField = true,
      doc = "J2ObjC JRE emulation libraries and their dependencies.")
  Depset /*<FileApiT>*/ jreLibrary();

  @SkylarkCallable(
      name = "library",
      structField = true,
      doc = "Library (.a) files compiled by dependencies of the current target.")
  Depset /*<FileApiT>*/ library();

  @SkylarkCallable(
      name = "link_inputs",
      structField = true,
      doc =
          "Link time artifacts from dependencies that do not fall into any other category such as"
              + " libraries or archives. This catch-all provides a way to add arbitrary data (e.g."
              + " Swift AST files) to the linker. The rule that adds these is also responsible to"
              + " add the necessary linker flags to 'linkopt'.")
  Depset /*<FileApiT>*/ linkInputs();

  @SkylarkCallable(
      name = "linked_binary",
      structField = true,
      doc =
          "Single-architecture linked binaries to be combined for the final multi-architecture "
              + "binary.")
  Depset /*<FileApiT>*/ linkedBinary();

  @SkylarkCallable(
      name = "linkmap_file",
      structField = true,
      doc = "Single-architecture link map for a binary.")
  Depset /*<FileApiT>*/ linkmapFile();

  @SkylarkCallable(name = "linkopt", structField = true, doc = "Linking options.")
  Depset /*<String>*/ linkopt();

  @SkylarkCallable(
      name = "merge_zip",
      structField = true,
      doc =
          "Merge zips to include in the bundle. The entries of these zip files are included "
              + "in the final bundle with the same path. The entries in the merge zips should not "
              + "include the bundle root path (e.g. 'Foo.app').")
  Depset /*<FileApiT>*/ mergeZip();

  @SkylarkCallable(
      name = "module_map",
      structField = true,
      doc = "Clang module maps, used to enforce proper use of private header files.")
  Depset /*<FileApiT>*/ moduleMap();

  @SkylarkCallable(
      name = "direct_module_maps",
      structField = true,
      doc =
          "Module map files from this target directly (no transitive module maps). "
              + "Used to enforce proper use of private header files and for Swift compilation.")
  Sequence<FileApiT> directModuleMaps();

  @SkylarkCallable(
      name = "multi_arch_dynamic_libraries",
      structField = true,
      doc = "Combined-architecture dynamic libraries to include in the final bundle.")
  Depset /*<FileApiT>*/ multiArchDynamicLibraries();

  @SkylarkCallable(
      name = "multi_arch_linked_archives",
      structField = true,
      doc = "Combined-architecture archives to include in the final bundle.")
  Depset /*<FileApiT>*/ multiArchLinkedArchives();

  @SkylarkCallable(
      name = "multi_arch_linked_binaries",
      structField = true,
      doc = "Combined-architecture binaries to include in the final bundle.")
  Depset /*<FileApiT>*/ multiArchLinkedBinaries();

  @SkylarkCallable(
      name = "sdk_dylib",
      structField = true,
      doc = "Names of SDK .dylib libraries to link with. For instance, 'libz' or 'libarchive'.")
  Depset /*<String>*/ sdkDylib();

  @SkylarkCallable(
      name = "sdk_framework",
      structField = true,
      doc = "Names of SDK frameworks to link with (e.g. 'AddressBook', 'QuartzCore').")
  Depset sdkFramework();

  @SkylarkCallable(name = "source", structField = true, doc = "All transitive source files.")
  Depset /*<FileApiT>*/ sourceForStarlark();

  @SkylarkCallable(
      name = "direct_sources",
      structField = true,
      doc = "All direct source files from this target (no transitive files).")
  Sequence<FileApiT> directSources();

  @SkylarkCallable(
      name = "static_framework_file",
      structField = true,
      doc = "The library files in .framework directories that should be statically linked.")
  Depset /*<FileApiT>*/ staticFrameworkFileForStarlark();

  @SkylarkCallable(
      name = "umbrella_header",
      structField = true,
      doc =
          "Clang umbrella header. Public headers are #included in umbrella headers to be "
              + "compatible with J2ObjC segmented headers.")
  Depset /*<FileApiT>*/ umbrellaHeader();

  @SkylarkCallable(
      name = "weak_sdk_framework",
      structField = true,
      doc =
          "Names of SDK frameworks to weakly link with. For instance, 'MediaAccessibility'. "
              + "In difference to regularly linked SDK frameworks, symbols from weakly linked "
              + "frameworks do not cause an error if they are not present at runtime.")
  Depset weakSdkFramework();

  @SkylarkCallable(
      name = "dynamic_framework_names",
      structField = true,
      doc = "Returns all names of dynamic frameworks in this provider.")
  Depset /*<String>*/ dynamicFrameworkNamesForStarlark();

  @SkylarkCallable(
      name = "dynamic_framework_paths",
      structField = true,
      doc = "Returns all framework paths to dynamic frameworks in this provider.")
  Depset /*<String>*/ dynamicFrameworkPathsForStarlark();

  @SkylarkCallable(
      name = "static_framework_names",
      structField = true,
      doc = "Returns all names of static frameworks in this provider.")
  Depset /*<String>*/ staticFrameworkNamesForStarlark();

  @SkylarkCallable(
      name = "static_framework_paths",
      structField = true,
      doc = "Returns all framework paths to static frameworks in this provider.")
  Depset /*<String>*/ staticFrameworkPathsForStarlark();

  @SkylarkCallable(
      name = "compilation_context",
      doc =
          "Returns the embedded <code>CcCompilationContext</code> that contains the"
              + "provider's compilation information.",
      structField = true)
  CcCompilationContextApi<FileApiT> getCcCompilationContext();
}
