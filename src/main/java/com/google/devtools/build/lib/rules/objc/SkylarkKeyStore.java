// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.objc;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.rules.objc.ObjcProvider.Key;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;

/**
 * A container for valid ObjcProvider keys, to be provided to skylark.
 */
@SkylarkModule(
  name = "objc_provider_keys_store",
  doc = "A container for valid ObjcProvider keys."
)
public class SkylarkKeyStore {
  
  @SkylarkCallable(
      name = "library",
      doc = "Returns a key that gives libraries in this target"
  )
  public static Key<Artifact> getLibrary() {
    return ObjcProvider.LIBRARY;
  }
  
  @SkylarkCallable(
      name = "imported_library",
      doc = "Returns a key that gives imported libraries in this target"
  )
  public static Key<Artifact> getImportedLibrary() {
    return ObjcProvider.IMPORTED_LIBRARY;
  }
  
  @SkylarkCallable(
      name = "linked_binary",
      doc = "Returns a key that gives single-architecture linked binaries to "
          + "be combined into a multi-architecture binary"
  )
  public static Key<Artifact> getLinkedBinary() {
    return ObjcProvider.LINKED_BINARY;
  }
  
  @SkylarkCallable(
      name = "force_load_library",
      doc = "Returns a key that  gives libraries to laod with the "
          + "'-force_load' flag."
  )
  public static Key<Artifact> getForceLoadLibrary() {
    return ObjcProvider.FORCE_LOAD_LIBRARY;
  }
  
  @SkylarkCallable(
      name = "header",
      doc = "Returns a key that gives all header files."
  )
  public static Key<Artifact> getHeader() {
    return ObjcProvider.HEADER;
  }
 
  @SkylarkCallable(
      name = "source",
      doc = "Returns a key that gives all source files."
  )
  public static Key<Artifact> getSource() {
    return ObjcProvider.SOURCE;
  }

  @SkylarkCallable(
      name = "define",
      doc = "Returns a key that  gives all values in 'defines' attributes."
  )
  public static Key<String> getDefine() {
    return ObjcProvider.DEFINE;
  }

  @SkylarkCallable(
      name = "asset_catalog",
      doc = "Returns the 'ASSET_CATALOG' key."
  )
  public static Key<Artifact> getAssetCatalog() {
    return ObjcProvider.ASSET_CATALOG;
  }
  
  @SkylarkCallable(
      name = "sdk_dylib",
      doc = "Returns the 'SDK_DYLIB' key."
  )
  public static Key<String> getSdkDylib() {
    return ObjcProvider.SDK_DYLIB;
  }
  
  @SkylarkCallable(
      name = "xcdatamodel",
      doc = "Returns the 'XCDATAMODEL' key."
  )
  public static Key<Artifact> getXcDataModel() {
    return ObjcProvider.XCDATAMODEL;
  }

  @SkylarkCallable(
      name = "module_map",
      doc = "Returns a key that gives clang module maps."
  )
  public static Key<Artifact> getModuleMap() {
    return ObjcProvider.MODULE_MAP;
  }

  @SkylarkCallable(
      name = "merge_zip",
      doc = "Returns a key that gives zips to include in the bundle."
  )
  public static Key<Artifact> getMergeZip() {
    return ObjcProvider.MERGE_ZIP;
  }

  @SkylarkCallable(
      name = "root_merge_zip",
      doc = "Returns a key that gives zips to include outside the bundle."
  )
  public static Key<Artifact> getRootMergeZip() {
    return ObjcProvider.ROOT_MERGE_ZIP;
  }

  @SkylarkCallable(
      name = "framework_file",
      doc = "Returns a key that gives .framework files to be included in "
          + "compilation and linking."
  )
  public static Key<Artifact> getFrameworkFile() {
    return ObjcProvider.FRAMEWORK_FILE;
  }

  @SkylarkCallable(
      name = "debug_symbols",
      doc = "Returns a key that gives an artifact containing debug symbol "
          + "information."
  )
  public static Key<Artifact> getDebugSymbols() {
    return ObjcProvider.DEBUG_SYMBOLS;
  }

  @SkylarkCallable(
      name = "debug_symbols_plist",
      doc = "Returns a key that gives an artifact containing the plist "
          + "on debug symbols."
  )
  public static Key<Artifact> getDebugSymbolsPlist() {
    return ObjcProvider.DEBUG_SYMBOLS_PLIST;
  }
  
  @SkylarkCallable(
      name = "breakpad_file",
      doc = "Returns a key that gives the generated breakpad file for crash "
          + "reporting."
  )
  public static Key<Artifact> getBreakpadFile() {
    return ObjcProvider.BREAKPAD_FILE;
  }
  
  @SkylarkCallable(
      name = "storyboard",
      doc = "Returns a key that gives artifacts for storyboard sources."
  )
  public static Key<Artifact> getStoryboard() {
    return ObjcProvider.STORYBOARD;
  }

  @SkylarkCallable(
      name = "xib",
      doc = "Returns a key that gives artifacts for .xib file sources."
  )
  public static Key<Artifact> getXib() {
    return ObjcProvider.XIB;
  }

  @SkylarkCallable(
      name = "strings",
      doc = "Returns a key that gives artifacts for strings source files."
  )
  public static Key<Artifact> getStrings() {
    return ObjcProvider.STRINGS;
  }
  
  @SkylarkCallable(
      name = "linkopt",
      doc = "Returns a key that gives linking options from dependencies."
  )
  public static Key<String> getLinkopt() {
    return ObjcProvider.LINKOPT;
  }
 
  @SkylarkCallable(
      name = "j2objc_library",
      doc = "Returns a key that gives static libraries that are built from "
          + "J2ObjC-translated Java code."
  )
  public static Key<Artifact> getJ2ObjcLibrary() {
    return ObjcProvider.J2OBJC_LIBRARY;
  }
  
  /**
   * All keys in ObjcProvider that will be passed in the corresponding Skylark provider.
   */
  // Only keys for Artifact or primitive types can be in the Skylark provider, as other types
  // are not supported as Skylark types.
  // Note: This list is only required to support objcprovider <-> skylarkprovider conversion, which
  // will be removed in favor of native skylark ObjcProvider access once that is implemented.
  static final ImmutableList<ObjcProvider.Key<?>> KEYS_FOR_SKYLARK =
      ImmutableList.<ObjcProvider.Key<?>>of(
          ObjcProvider.LIBRARY,
          ObjcProvider.IMPORTED_LIBRARY,
          ObjcProvider.LINKED_BINARY,
          ObjcProvider.FORCE_LOAD_LIBRARY,
          ObjcProvider.HEADER,
          ObjcProvider.SOURCE,
          ObjcProvider.DEFINE,
          ObjcProvider.ASSET_CATALOG,
          ObjcProvider.SDK_DYLIB,
          ObjcProvider.XCDATAMODEL,
          ObjcProvider.MODULE_MAP,
          ObjcProvider.MERGE_ZIP,
          ObjcProvider.FRAMEWORK_FILE,
          ObjcProvider.DEBUG_SYMBOLS,
          ObjcProvider.DEBUG_SYMBOLS_PLIST,
          ObjcProvider.BREAKPAD_FILE,
          ObjcProvider.STORYBOARD,
          ObjcProvider.XIB,
          ObjcProvider.STRINGS,
          ObjcProvider.LINKOPT,
          ObjcProvider.J2OBJC_LIBRARY,
          ObjcProvider.ROOT_MERGE_ZIP);
}
