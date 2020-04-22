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

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.skylarkbuildapi.android.AndroidLibraryResourceClassJarProviderApi;
import com.google.devtools.build.lib.syntax.Depset;
import com.google.devtools.build.lib.syntax.EvalException;

/**
 * A provider which contains the resource class jars from android_library rules. See {@link
 * AndroidRuleClasses#ANDROID_RESOURCES_CLASS_JAR}.
 */
public final class AndroidLibraryResourceClassJarProvider extends NativeInfo
    implements AndroidLibraryResourceClassJarProviderApi<Artifact> {

  public static final Provider PROVIDER = new Provider();

  private final NestedSet<Artifact> resourceClassJars;

  private AndroidLibraryResourceClassJarProvider(NestedSet<Artifact> resourceClassJars) {
    super(PROVIDER);
    this.resourceClassJars = resourceClassJars;
  }

  public static AndroidLibraryResourceClassJarProvider create(
      NestedSet<Artifact> resourceClassJars) {
    return new AndroidLibraryResourceClassJarProvider(resourceClassJars);
  }

  public static AndroidLibraryResourceClassJarProvider getProvider(
      TransitiveInfoCollection target) {
    return (AndroidLibraryResourceClassJarProvider)
        target.get(AndroidLibraryResourceClassJarProvider.PROVIDER.getKey());
  }

  @Override
  public Depset /*<Artifact>*/ getResourceClassJarsForStarlark() {
    return Depset.of(Artifact.TYPE, resourceClassJars);
  }

  public NestedSet<Artifact> getResourceClassJars() {
    return resourceClassJars;
  }

  /** Provider class for {@link AndroidLibraryResourceClassJarProvider} objects. */
  public static class Provider extends BuiltinProvider<AndroidLibraryResourceClassJarProvider>
      implements AndroidLibraryResourceClassJarProviderApi.Provider<Artifact> {

    private Provider() {
      super(NAME, AndroidLibraryResourceClassJarProvider.class);
    }

    public String getName() {
      return NAME;
    }

    @Override
    public AndroidLibraryResourceClassJarProvider create(Depset jars) throws EvalException {
      return new AndroidLibraryResourceClassJarProvider(
          NestedSetBuilder.<Artifact>stableOrder()
              .addTransitive(Depset.cast(jars, Artifact.class, "jars"))
              .build());
    }
  }
}
