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

package com.google.devtools.build.skydoc.fakebuildapi.android;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.starlarkbuildapi.android.DataBindingV2ProviderApi;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;

/** Fake implementation of {@link DataBindingV2Provider}. */
public class FakeDataBindingV2Provider implements DataBindingV2ProviderApi<FileApi> {

  @Override
  public ImmutableList<FileApi> getSetterStores() {
    return null;
  }

  @Override
  public ImmutableList<FileApi> getClassInfos() {
    return null;
  }

  @Override
  public Depset getTransitiveBRFilesForStarlark() {
    return null;
  }

  @Override
  public Depset getTransitiveLabelAndJavaPackagesForStarlark() {
    return null;
  }

  @Override
  @Nullable
  public ImmutableList<LabelJavaPackagePair> getLabelAndJavaPackages() {
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

  /** Fake implementation of {@link DataBindingV2ProviderApi.Provider}. */
  public static class FakeProvider implements DataBindingV2ProviderApi.Provider<FileApi> {

    @Override
    public DataBindingV2ProviderApi<FileApi> createInfo(
        Object setterStoreFile,
        Object classInfoFile,
        Object brFile,
        Object label,
        Object javaPackage,
        Sequence<?> databindingV2ProvidersInDeps,
        Sequence<?> databindingV2ProvidersInExports)
        throws EvalException {
      return new FakeDataBindingV2Provider();
    }
  }
}
