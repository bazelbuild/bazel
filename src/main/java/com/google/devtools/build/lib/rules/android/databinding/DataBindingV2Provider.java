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
package com.google.devtools.build.lib.rules.android.databinding;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.skylarkbuildapi.android.DataBindingV2ProviderApi;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;

/**
 * A provider that exposes this enables <a
 * href="https://developer.android.com/topic/libraries/data-binding/index.html">data binding</a>
 * version 2 on its resource processing and Java compilation.
 */
public final class DataBindingV2Provider extends NativeInfo
    implements DataBindingV2ProviderApi<Artifact> {

  public static final Provider PROVIDER = new Provider();

  private final ImmutableList<Artifact> classInfos;

  private final ImmutableList<Artifact> setterStores;

  private final NestedSet<Artifact> transitiveBRFiles;

  public DataBindingV2Provider(
      ImmutableList<Artifact> classInfos,
      ImmutableList<Artifact> setterStores,
      NestedSet<Artifact> transitiveBRFiles) {
    super(PROVIDER);
    this.classInfos = classInfos;
    this.setterStores = setterStores;
    this.transitiveBRFiles = transitiveBRFiles;
  }

  @Override
  public ImmutableList<Artifact> getClassInfos() {
    return classInfos;
  }

  @Override
  public ImmutableList<Artifact> getSetterStores() {
    return setterStores;
  }

  @Override
  public NestedSet<Artifact> getTransitiveBRFiles() {
    return transitiveBRFiles;
  }

  /** The provider can construct the DataBindingV2Provider provider. */
  public static class Provider extends BuiltinProvider<DataBindingV2Provider>
      implements DataBindingV2ProviderApi.Provider<Artifact> {

    private Provider() {
      super(NAME, DataBindingV2Provider.class);
    }

    @Override
    public DataBindingV2ProviderApi<Artifact> createInfo(
        SkylarkList<Artifact> setterStores,
        SkylarkList<Artifact> clientInfos,
        SkylarkNestedSet transitiveBrFiles) throws EvalException {

      return new DataBindingV2Provider(
          setterStores.getImmutableList(),
          clientInfos.getImmutableList(),
          transitiveBrFiles.getSet(Artifact.class));
    }
  }
}
