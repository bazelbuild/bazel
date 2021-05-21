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

import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.starlarkbuildapi.apple.ObjcProviderApi;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.Sequence;

/**
 * Fake implementation of {@link ObjcProviderApi}.
 */
public class FakeObjcProvider implements ObjcProviderApi<FileApi> {
  @Override
  public Depset /*<LibraryToLink>*/ ccLibrariesForStarlark() {
    return null;
  }

  @Override
  public Depset /*<Linkstamp>*/ linkstampForstarlark() {
    return null;
  }

  @Override
  public Depset /*<FileApi>*/ dynamicFrameworkFileForStarlark() {
    return null;
  }

  @Override
  public Depset /*<FileApi>*/ forceLoadLibrary() {
    return null;
  }

  @Override
  public Sequence<FileApi> directHeaders() {
    return null;
  }

  @Override
  public Depset /*<FileApi>*/ importedLibrary() {
    return null;
  }

  @Override
  public Depset /*<String>*/ strictIncludeForStarlark() {
    return null;
  }

  @Override
  public Depset /*<FileApi>*/ j2objcLibrary() {
    return null;
  }

  @Override
  public Depset /*<FileApi>*/ jreLibrary() {
    return null;
  }

  @Override
  public Depset /*<FileApi>*/ library() {
    return null;
  }

  @Override
  public Depset /*<FileApi>*/ linkInputs() {
    return null;
  }

  @Override
  public Depset /*<String>*/ linkopt() {
    return null;
  }

  @Override
  public Depset /*<FileApi>*/ moduleMap() {
    return null;
  }

  @Override
  public Sequence<FileApi> directModuleMaps() {
    return null;
  }

  @Override
  public Depset /*<String>*/ sdkDylib() {
    return null;
  }

  @Override
  public Depset sdkFramework() {
    return null;
  }

  @Override
  public Depset /*<FileApi>*/ sourceForStarlark() {
    return null;
  }

  @Override
  public Sequence<FileApi> directSources() {
    return null;
  }

  @Override
  public Depset /*<FileApi>*/ staticFrameworkFileForStarlark() {
    return null;
  }

  @Override
  public Depset /*<FileApi>*/ umbrellaHeader() {
    return null;
  }

  @Override
  public Depset weakSdkFramework() {
    return null;
  }

  @Override
  public Depset /*<String>*/ dynamicFrameworkNamesForStarlark() {
    return null;
  }

  @Override
  public Depset /*<String>*/ dynamicFrameworkPathsForStarlark() {
    return null;
  }

  @Override
  public Depset /*<String>*/ staticFrameworkNamesForStarlark() {
    return null;
  }

  @Override
  public Depset /*<String>*/ staticFrameworkPathsForStarlark() {
    return null;
  }

  @Override
  public void repr(Printer printer) {}
}
