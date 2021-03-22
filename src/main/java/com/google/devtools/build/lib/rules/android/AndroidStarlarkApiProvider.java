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
package com.google.devtools.build.lib.rules.android;

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.starlark.StarlarkApiProvider;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider.OutputJar;
import com.google.devtools.build.lib.starlarkbuildapi.android.AndroidStarlarkApiProviderApi;
import java.util.function.Function;
import javax.annotation.Nullable;

/**
 * A class that exposes the Android providers to Starlark. It is intended to provide a simple and
 * stable interface for Starlark users.
 */
@Immutable
public class AndroidStarlarkApiProvider extends StarlarkApiProvider
    implements AndroidStarlarkApiProviderApi<Artifact> {
  /** The name of the field in Starlark used to access this class. */
  public static final String NAME = "android";

  private final IdlInfo idlInfo = new IdlInfo();
  private final AndroidResourcesInfo resourceInfo;

  public AndroidStarlarkApiProvider(AndroidResourcesInfo resourceInfo) {
    this.resourceInfo = resourceInfo;
  }

  @Override
  public boolean isImmutable() {
    return true; // immutable and Starlark-hashable
  }

  @Override
  public Artifact getApk() {
    return getIdeInfoProvider().getSignedApk();
  }

  private AndroidIdeInfoProvider getIdeInfoProvider() {
    return getInfo().get(AndroidIdeInfoProvider.PROVIDER);
  }

  @Override
  public String getJavaPackage() {
    return getIdeInfoProvider().getJavaPackage();
  }

  @Override
  public Artifact getManifest() {
    return getIdeInfoProvider().getManifest();
  }

  @Override
  public Artifact getMergedManifest() {
    return getIdeInfoProvider().getGeneratedManifest();
  }

  @Override
  public ImmutableMap<String, Depset> getNativeLibs() {
    return getIdeInfoProvider().getNativeLibsStarlark();
  }

  @Override
  public Artifact getResourceApk() {
    return getIdeInfoProvider().getResourceApk();
  }

  @Override
  public ImmutableCollection<Artifact> getApksUnderTest() {
    return getIdeInfoProvider().getApksUnderTest();
  }

  @Override
  public boolean definesAndroidResources() {
    return getIdeInfoProvider().definesAndroidResources();
  }

  @Override
  public IdlInfo getIdlInfo() {
    return idlInfo;
  }

  @Override
  public Depset /*<Artifact>*/ getResources() {
    return Depset.of(
        Artifact.TYPE, collectDirectArtifacts(ValidatedAndroidResources::getResources));
  }

  @Override
  @Nullable
  public JavaRuleOutputJarsProvider.OutputJar getResourceJar() {
    return getIdeInfoProvider().getResourceJar();
  }

  @Override
  public Artifact getAar() {
    return getIdeInfoProvider().getAar();
  }

  private NestedSet<Artifact> collectDirectArtifacts(
      final Function<ValidatedAndroidResources, Iterable<Artifact>> artifactFunction) {
    if (resourceInfo == null) {
      return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }
    // This will iterate over all (direct) resources. If this turns out to be a performance
    // problem, {@link ValidatedAndroidResources#getArtifacts()} can be changed to return
    // NestedSets.
    return NestedSetBuilder.wrap(
        Order.STABLE_ORDER,
        Iterables.concat(
            Iterables.transform(
                resourceInfo.getDirectAndroidResources().toList(),
                data -> artifactFunction.apply(data))));
  }

  /** Helper class to provide information about IDLs related to this rule. */
  @Immutable
  public class IdlInfo implements IdlInfoApi<Artifact> {
    @Override
    public String getImportRoot() {
      return getIdeInfoProvider().getIdlImportRoot();
    }

    @Override
    public ImmutableCollection<Artifact> getSources() {
      return getIdeInfoProvider().getIdlSrcs();
    }

    @Override
    public ImmutableCollection<Artifact> getIdlGeneratedJavaFiles() {
      return getIdeInfoProvider().getIdlGeneratedJavaFiles();
    }

    @Override
    @Nullable
    public JavaRuleOutputJarsProvider.OutputJar getIdlOutput() {
      if (getIdeInfoProvider().getIdlClassJar() == null) {
        return null;
      }

      Artifact idlSourceJar = getIdeInfoProvider().getIdlSourceJar();
      return OutputJar.builder()
          .setClassJar(getIdeInfoProvider().getIdlClassJar())
          .addSourceJar(idlSourceJar)
          .build();
    }
  }
}
