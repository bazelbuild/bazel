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
package com.google.devtools.build.lib.rules.android.databinding;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.skylarkbuildapi.android.UsesDataBindingProviderApi;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.SkylarkList;
import java.util.Collection;

/**
 * A provider that exposes this enables <a
 * href="https://developer.android.com/topic/libraries/data-binding/index.html">data binding</a> on
 * its resource processing and Java compilation.
 */
public final class UsesDataBindingProvider extends NativeInfo
    implements UsesDataBindingProviderApi<Artifact> {

  public static final Provider PROVIDER = new Provider();

  private final ImmutableList<Artifact> metadataOutputs;

  public UsesDataBindingProvider(Collection<Artifact> metadataOutputs) {
    super(PROVIDER);
    this.metadataOutputs = ImmutableList.copyOf(metadataOutputs);
  }

  @Override
  public ImmutableList<Artifact> getMetadataOutputs() {
    return metadataOutputs;
  }

  /** The provider can construct the UsesDataBindingInfo provider. */
  public static class Provider extends BuiltinProvider<UsesDataBindingProvider>
      implements UsesDataBindingProviderApi.Provider<Artifact> {

    private Provider() {
      super(NAME, UsesDataBindingProvider.class);
    }

    @Override
    public UsesDataBindingProvider createInfo(SkylarkList<Artifact> metadataOutputs)
        throws EvalException {
      return new UsesDataBindingProvider(metadataOutputs.getImmutableList());
    }
  }
}