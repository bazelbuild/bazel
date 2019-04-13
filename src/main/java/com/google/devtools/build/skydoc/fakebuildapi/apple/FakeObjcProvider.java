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

package com.google.devtools.build.skydoc.fakebuildapi.apple;

import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.apple.ObjcProviderApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;

/**
 * Fake implementation of {@link ObjcProviderApi}.
 */
public class FakeObjcProvider implements ObjcProviderApi<FileApi> {

  @Override
  public NestedSet<FileApi> assetCatalog() {
    return null;
  }

  @Override
  public SkylarkNestedSet bundleFile() {
    return null;
  }

  @Override
  public NestedSet<String> define() {
    return null;
  }

  @Override
  public SkylarkNestedSet dynamicFrameworkDir() {
    return null;
  }

  @Override
  public NestedSet<FileApi> dynamicFrameworkFile() {
    return null;
  }

  @Override
  public NestedSet<FileApi> exportedDebugArtifacts() {
    return null;
  }

  @Override
  public SkylarkNestedSet frameworkSearchPathOnly() {
    return null;
  }

  @Override
  public NestedSet<FileApi> forceLoadLibrary() {
    return null;
  }

  @Override
  public NestedSet<FileApi> header() {
    return null;
  }

  @Override
  public NestedSet<FileApi> importedLibrary() {
    return null;
  }

  @Override
  public SkylarkNestedSet include() {
    return null;
  }

  @Override
  public SkylarkNestedSet includeSystem() {
    return null;
  }

  @Override
  public SkylarkNestedSet iquote() {
    return null;
  }

  @Override
  public NestedSet<FileApi> j2objcLibrary() {
    return null;
  }

  @Override
  public NestedSet<FileApi> jreLibrary() {
    return null;
  }

  @Override
  public NestedSet<FileApi> library() {
    return null;
  }

  @Override
  public NestedSet<FileApi> linkInputs() {
    return null;
  }

  @Override
  public NestedSet<FileApi> linkedBinary() {
    return null;
  }

  @Override
  public NestedSet<FileApi> linkmapFile() {
    return null;
  }

  @Override
  public NestedSet<String> linkopt() {
    return null;
  }

  @Override
  public NestedSet<FileApi> mergeZip() {
    return null;
  }

  @Override
  public NestedSet<FileApi> moduleMap() {
    return null;
  }

  @Override
  public NestedSet<FileApi> multiArchDynamicLibraries() {
    return null;
  }

  @Override
  public NestedSet<FileApi> multiArchLinkedArchives() {
    return null;
  }

  @Override
  public NestedSet<FileApi> multiArchLinkedBinaries() {
    return null;
  }

  @Override
  public NestedSet<FileApi> rootMergeZip() {
    return null;
  }

  @Override
  public NestedSet<String> sdkDylib() {
    return null;
  }

  @Override
  public SkylarkNestedSet sdkFramework() {
    return null;
  }

  @Override
  public NestedSet<FileApi> source() {
    return null;
  }

  @Override
  public NestedSet<FileApi> staticFrameworkFile() {
    return null;
  }

  @Override
  public NestedSet<FileApi> storyboard() {
    return null;
  }

  @Override
  public NestedSet<FileApi> strings() {
    return null;
  }

  @Override
  public NestedSet<FileApi> umbrellaHeader() {
    return null;
  }

  @Override
  public SkylarkNestedSet weakSdkFramework() {
    return null;
  }

  @Override
  public SkylarkNestedSet xcassetsDir() {
    return null;
  }

  @Override
  public NestedSet<FileApi> xcdatamodel() {
    return null;
  }

  @Override
  public NestedSet<FileApi> xib() {
    return null;
  }

  @Override
  public SkylarkNestedSet getStaticFrameworkDirsForSkylark() {
    return null;
  }

  @Override
  public NestedSet<String> dynamicFrameworkNames() {
    return null;
  }

  @Override
  public NestedSet<String> dynamicFrameworkPaths() {
    return null;
  }

  @Override
  public NestedSet<String> staticFrameworkNames() {
    return null;
  }

  @Override
  public NestedSet<String> staticFrameworkPaths() {
    return null;
  }

  @Override
  public void repr(SkylarkPrinter printer) {}
}
