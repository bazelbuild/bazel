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

import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.skylarkinterface.SkylarkValue;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import com.google.devtools.build.lib.syntax.SkylarkSemantics.FlagIdentifier;

/**
 * An interface for an info type that provides all compiling and linking information in the
 * transitive closure of its deps that are needed for building Objective-C rules.
 */
@SkylarkModule(
    name = "ObjcProvider",
    category = SkylarkModuleCategory.PROVIDER,
    doc = "A provider for compilation and linking of objc."
)
public interface ObjcProviderApi<FileApiT extends FileApi> extends SkylarkValue {

  @SkylarkCallable(
      name = "asset_catalog",
      structField = true,
      doc =
          "Asset catalog resource files.",
      disableWithFlag = FlagIdentifier.INCOMPATIBLE_DISABLE_OBJC_PROVIDER_RESOURCES)
  public NestedSet<FileApiT> assetCatalog();

  @SkylarkCallable(
      name = "bundle_file",
      structField = true,
      doc =
          "Files that are plopped into the final bundle at some arbitrary bundle path.",
      disableWithFlag = FlagIdentifier.INCOMPATIBLE_DISABLE_OBJC_PROVIDER_RESOURCES)
  public SkylarkNestedSet bundleFile();

  @SkylarkCallable(name = "define",
      structField = true,
      doc = "A set of strings from 'defines' attributes. These are to be passed as '-D' flags to "
          + "all invocations of the compiler for this target and all depending targets."
  )
  public NestedSet<String> define();

  @SkylarkCallable(name = "dynamic_framework_dir",
      structField = true,
      doc = "Exec paths of .framework directories corresponding to dynamic frameworks to link."
  )
  public SkylarkNestedSet dynamicFrameworkDir();

  @SkylarkCallable(name = "dynamic_framework_file",
      structField = true,
      doc = "Files in .framework directories belonging to a dynamically linked framework."
  )
  public NestedSet<FileApiT> dynamicFrameworkFile();

  @SkylarkCallable(name = "exported_debug_artifacts",
      structField = true,
      doc = "Debug files that should be exported by the top-level target."
  )
  public NestedSet<FileApiT> exportedDebugArtifacts();

  @SkylarkCallable(name = "framework_search_path_only",
      structField = true,
      doc = "Exec paths of .framework directories corresponding to frameworks to include "
          + "in search paths, but not to link."
  )
  public SkylarkNestedSet frameworkSearchPathOnly();

  @SkylarkCallable(name = "force_load_library",
      structField = true,
      doc = "Libraries to load with -force_load."
  )
  public NestedSet<FileApiT> forceLoadLibrary();

  @SkylarkCallable(name = "header",
      structField = true,
      doc = "All header files. These may be either public or private headers."
  )
  public NestedSet<FileApiT> header();

  @SkylarkCallable(name = "imported_library",
      structField = true,
      doc = "Imported precompiled static libraries (.a files) to be linked into the binary."
  )
  public NestedSet<FileApiT> importedLibrary();

  @SkylarkCallable(name = "include",
      structField = true,
      doc = "Include search paths specified with '-I' on the command line. Also known as "
          + "header search paths (and distinct from <em>user</em> header search paths)."
  )
  public SkylarkNestedSet include();

  @SkylarkCallable(name = "include_system",
      structField = true,
      doc = "System include search paths (typically specified with -isystem)."
  )
  public SkylarkNestedSet includeSystem();

  @SkylarkCallable(name = "iquote",
      structField = true,
      doc = "User header search paths (typically specified with -iquote)."
  )
  public SkylarkNestedSet iquote();

  @SkylarkCallable(name = "j2objc_library",
      structField = true,
      doc = "Static libraries that are built from J2ObjC-translated Java code."
  )
  public NestedSet<FileApiT> j2objcLibrary();

  @SkylarkCallable(name = "jre_library",
      structField = true,
      doc = "J2ObjC JRE emulation libraries and their dependencies."
  )
  public NestedSet<FileApiT> jreLibrary();

  @SkylarkCallable(name = "library",
      structField = true,
      doc = "Library (.a) files compiled by dependencies of the current target."
  )
  public NestedSet<FileApiT> library();

  @SkylarkCallable(name = "link_inputs",
      structField = true,
      doc = "Link time artifacts from dependencies that do not fall into any other category such "
          + "as libraries or archives. This catch-all provides a way to add arbitrary data (e.g. "
          + "Swift AST files) to the linker. The rule that adds these is also responsible to "
          + "add the necessary linker flags to 'linkopt'."
  )
  public NestedSet<FileApiT> linkInputs();

  @SkylarkCallable(name = "linked_binary",
      structField = true,
      doc = "Single-architecture linked binaries to be combined for the final multi-architecture "
          + "binary."
  )
  public NestedSet<FileApiT> linkedBinary();

  @SkylarkCallable(name = "linkmap_file",
      structField = true,
      doc = "Single-architecture link map for a binary."
  )
  public NestedSet<FileApiT> linkmapFile();

  @SkylarkCallable(name = "linkopt",
      structField = true,
      doc = "Linking options."
  )
  public NestedSet<String> linkopt();

  @SkylarkCallable(
      name = "merge_zip",
      structField = true,
      doc =
           "Merge zips to include in the bundle. The entries of these zip files are included "
              + "in the final bundle with the same path. The entries in the merge zips should not "
              + "include the bundle root path (e.g. 'Foo.app').",
      disableWithFlag = FlagIdentifier.INCOMPATIBLE_DISABLE_OBJC_PROVIDER_RESOURCES)
  public NestedSet<FileApiT> mergeZip();

  @SkylarkCallable(name = "module_map",
      structField = true,
      doc = "Clang module maps, used to enforce proper use of private header files."
  )
  public NestedSet<FileApiT> moduleMap();

  @SkylarkCallable(name = "multi_arch_dynamic_libraries",
      structField = true,
      doc = "Combined-architecture dynamic libraries to include in the final bundle."
  )
  public NestedSet<FileApiT> multiArchDynamicLibraries();

  @SkylarkCallable(name = "multi_arch_linked_archives",
      structField = true,
      doc = "Combined-architecture archives to include in the final bundle."
  )
  public NestedSet<FileApiT> multiArchLinkedArchives();

  @SkylarkCallable(name = "multi_arch_linked_binaries",
      structField = true,
      doc = "Combined-architecture binaries to include in the final bundle."
  )
  public NestedSet<FileApiT> multiArchLinkedBinaries();

  @SkylarkCallable(
      name = "root_merge_zip",
      structField = true,
      doc =
          "Merge zips to include in the ipa and outside the bundle root.",
      disableWithFlag = FlagIdentifier.INCOMPATIBLE_DISABLE_OBJC_PROVIDER_RESOURCES)
  public NestedSet<FileApiT> rootMergeZip();

  @SkylarkCallable(name = "sdk_dylib",
      structField = true,
      doc = "Names of SDK .dylib libraries to link with. For instance, 'libz' or 'libarchive'."
  )
  public NestedSet<String> sdkDylib();

  @SkylarkCallable(name = "sdk_framework",
      structField = true,
      doc = "Names of SDK frameworks to link with (e.g. 'AddressBook', 'QuartzCore')."
  )
  public SkylarkNestedSet sdkFramework();

  @SkylarkCallable(name = "source",
      structField = true,
      doc = "All transitive source files."
  )
  public NestedSet<FileApiT> source();

  @SkylarkCallable(name = "static_framework_file",
      structField = true,
      doc = "Files in .framework directories that should be statically included as inputs "
          + "when compiling and linking."
  )
  public NestedSet<FileApiT> staticFrameworkFile();

  @SkylarkCallable(
      name = "storyboard",
      structField = true,
      doc =
          "Files for storyboard sources.",
      disableWithFlag = FlagIdentifier.INCOMPATIBLE_DISABLE_OBJC_PROVIDER_RESOURCES)
  public NestedSet<FileApiT> storyboard();

  @SkylarkCallable(
      name = "strings",
      structField = true,
      doc =
          "Files for strings source files.",
      disableWithFlag = FlagIdentifier.INCOMPATIBLE_DISABLE_OBJC_PROVIDER_RESOURCES)
  public NestedSet<FileApiT> strings();

  @SkylarkCallable(name = "umbrella_header",
      structField = true,
      doc = "Clang umbrella header. Public headers are #included in umbrella headers to be "
          + "compatible with J2ObjC segmented headers."
  )
  public NestedSet<FileApiT> umbrellaHeader();

  @SkylarkCallable(
      name = "weak_sdk_framework",
      structField = true,
      doc =
          "Names of SDK frameworks to weakly link with. For instance, 'MediaAccessibility'. "
              + "In difference to regularly linked SDK frameworks, symbols from weakly linked "
              + "frameworks do not cause an error if they are not present at runtime.")
  public SkylarkNestedSet weakSdkFramework();

  @SkylarkCallable(
      name = "xcassets_dir",
      structField = true,
      doc =
          "The set of all unique asset catalog directories (*.xcassets) containing files "
              + "in 'asset_catalogs'.",
      disableWithFlag = FlagIdentifier.INCOMPATIBLE_DISABLE_OBJC_PROVIDER_RESOURCES)
  public SkylarkNestedSet xcassetsDir();

  @SkylarkCallable(
      name = "xcdatamodel",
      structField = true,
      doc =
          "Files that comprise the data models of the final linked binary.",
      disableWithFlag = FlagIdentifier.INCOMPATIBLE_DISABLE_OBJC_PROVIDER_RESOURCES)
  public NestedSet<FileApiT> xcdatamodel();

  @SkylarkCallable(
      name = "xib",
      structField = true,
      doc =
          ".xib resource files",
      disableWithFlag = FlagIdentifier.INCOMPATIBLE_DISABLE_OBJC_PROVIDER_RESOURCES)
  public NestedSet<FileApiT> xib();

  @SkylarkCallable(
      name = "framework_dir",
      structField = true,
      doc = "Returns all unique static framework directories (directories ending in '.framework') "
          + "for all static framework files in this provider."
  )
  public SkylarkNestedSet getStaticFrameworkDirsForSkylark();
}
