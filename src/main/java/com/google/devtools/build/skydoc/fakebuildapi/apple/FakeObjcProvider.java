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

import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.apple.ObjcProviderApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.syntax.Sequence;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;

/**
 * Fake implementation of {@link ObjcProviderApi}.
 */
public class FakeObjcProvider implements ObjcProviderApi<FileApi> {

  @Override
  public SkylarkNestedSet /*<String>*/ defineForStarlark() {
    return null;
  }

  @Override
  public SkylarkNestedSet /*<FileApi>*/ dynamicFrameworkFileForStarlark() {
    return null;
  }

  @Override
  public SkylarkNestedSet /*<FileApi>*/ exportedDebugArtifacts() {
    return null;
  }

  @Override
  public SkylarkNestedSet frameworkSearchPathOnly() {
    return null;
  }

  @Override
  public SkylarkNestedSet /*<FileApi>*/ forceLoadLibrary() {
    return null;
  }

  @Override
  public SkylarkNestedSet /*<FileApi>*/ headerForStarlark() {
    return null;
  }

  @Override
  public Sequence<FileApi> directHeaders() {
    return null;
  }

  @Override
  public SkylarkNestedSet /*<FileApi>*/ importedLibrary() {
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
  public SkylarkNestedSet /*<FileApi>*/ j2objcLibrary() {
    return null;
  }

  @Override
  public SkylarkNestedSet /*<FileApi>*/ jreLibrary() {
    return null;
  }

  @Override
  public SkylarkNestedSet /*<FileApi>*/ library() {
    return null;
  }

  @Override
  public SkylarkNestedSet /*<FileApi>*/ linkInputs() {
    return null;
  }

  @Override
  public SkylarkNestedSet /*<FileApi>*/ linkedBinary() {
    return null;
  }

  @Override
  public SkylarkNestedSet /*<FileApi>*/ linkmapFile() {
    return null;
  }

  @Override
  public SkylarkNestedSet /*<String>*/ linkopt() {
    return null;
  }

  @Override
  public SkylarkNestedSet /*<FileApi>*/ mergeZip() {
    return null;
  }

  @Override
  public SkylarkNestedSet /*<FileApi>*/ moduleMap() {
    return null;
  }

  @Override
  public Sequence<FileApi> directModuleMaps() {
    return null;
  }

  @Override
  public SkylarkNestedSet /*<FileApi>*/ multiArchDynamicLibraries() {
    return null;
  }

  @Override
  public SkylarkNestedSet /*<FileApi>*/ multiArchLinkedArchives() {
    return null;
  }

  @Override
  public SkylarkNestedSet /*<FileApi>*/ multiArchLinkedBinaries() {
    return null;
  }

  @Override
  public SkylarkNestedSet /*<String>*/ sdkDylib() {
    return null;
  }

  @Override
  public SkylarkNestedSet sdkFramework() {
    return null;
  }

  @Override
  public SkylarkNestedSet /*<FileApi>*/ sourceForStarlark() {
    return null;
  }

  @Override
  public Sequence<FileApi> directSources() {
    return null;
  }

  @Override
  public SkylarkNestedSet /*<FileApi>*/ staticFrameworkFileForStarlark() {
    return null;
  }

  @Override
  public SkylarkNestedSet /*<FileApi>*/ umbrellaHeader() {
    return null;
  }

  @Override
  public SkylarkNestedSet weakSdkFramework() {
    return null;
  }

  @Override
  public SkylarkNestedSet /*<String>*/ dynamicFrameworkNamesForStarlark() {
    return null;
  }

  @Override
  public SkylarkNestedSet /*<String>*/ dynamicFrameworkPathsForStarlark() {
    return null;
  }

  @Override
  public SkylarkNestedSet /*<String>*/ staticFrameworkNamesForStarlark() {
    return null;
  }

  @Override
  public SkylarkNestedSet /*<String>*/ staticFrameworkPathsForStarlark() {
    return null;
  }

  @Override
  public void repr(SkylarkPrinter printer) {}
}
