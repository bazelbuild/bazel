// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.skydoc.fakebuildapi.proto;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.ProtoInfoApi;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Printer;

/** Fake implementation of {@link ProtoInfoApi}. */
public class FakeProtoInfo implements ProtoInfoApi<FileApi> {

  @Override
  public Depset getTransitiveImports() {
    return null;
  }

  @Override
  public Depset getTransitiveProtoSourcesForStarlark() {
    return null;
  }

  @Override
  public ImmutableList<FileApi> getDirectProtoSources() {
    return null;
  }

  @Override
  public Depset getStrictImportableProtoSourcesForDependentsForStarlark() {
    return null;
  }

  @Override
  public FileApi getDirectDescriptorSet() {
    return null;
  }

  @Override
  public Depset getTransitiveDescriptorSetsForStarlark() {
    return null;
  }

  @Override
  public Depset getTransitiveProtoSourceRootsForStarlark() {
    return null;
  }

  @Override
  public String getDirectProtoSourceRoot() {
    return null;
  }

  @Override
  public String toProto() throws EvalException {
    return null;
  }

  @Override
  public String toJson() throws EvalException {
    return null;
  }

  @Override
  public void repr(Printer printer) {}

  /** Fake implementation of {@link ProtoInfoProviderApi}. */
  public static class FakeProtoInfoProvider implements ProtoInfoProviderApi {
    @Override
    public void repr(Printer printer) {}
  }
}
