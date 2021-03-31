// Copyright 2015 The Bazel Authors. All rights reserved.
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

import static com.google.devtools.build.lib.rules.android.AndroidStarlarkData.fromNoneable;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider.JavaOutput;
import com.google.devtools.build.lib.starlarkbuildapi.android.AndroidIdeInfoProviderApi;
import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;

/** An Android target provider to provide Android-specific info to IDEs. */
@Immutable
public final class AndroidIdeInfoProvider extends NativeInfo
    implements AndroidIdeInfoProviderApi<Artifact, JavaOutput> {

  public static final Provider PROVIDER = new Provider();

  /** Builder for {@link AndroidIdeInfoProvider} */
  public static class Builder {
    private Artifact manifest = null;
    private Artifact generatedManifest = null;
    private Artifact apk = null;
    private Artifact resourceApk = null;
    private Artifact idlClassJar = null;
    private Artifact idlSourceJar = null;
    private JavaOutput resourceJarJavaOutput = null;
    private String javaPackage = null;
    private String idlImportRoot = null;
    private final Set<Artifact> idlSrcs = new LinkedHashSet<>();
    private final Set<Artifact> idlGeneratedJavaFiles = new LinkedHashSet<>();
    private final Set<Artifact> apksUnderTest = new LinkedHashSet<>();
    private boolean definesAndroidResources;
    private Artifact aar = null;
    private Map<String, NestedSet<Artifact>> nativeLibs = null;

    public AndroidIdeInfoProvider build() {
      return new AndroidIdeInfoProvider(
          javaPackage,
          idlImportRoot,
          manifest,
          generatedManifest,
          apk,
          idlClassJar,
          idlSourceJar,
          resourceJarJavaOutput,
          definesAndroidResources,
          aar,
          ImmutableList.copyOf(idlSrcs),
          ImmutableList.copyOf(idlGeneratedJavaFiles),
          ImmutableList.copyOf(apksUnderTest),
          nativeLibs != null
              ? ImmutableMap.copyOf(nativeLibs)
              : ImmutableMap.<String, NestedSet<Artifact>>of(),
          resourceApk);
    }

    public Builder setJavaPackage(String javaPackage) {
      this.javaPackage = javaPackage;
      return this;
    }

    public Builder setDefinesAndroidResources(boolean definesAndroidResources) {
      this.definesAndroidResources = definesAndroidResources;
      return this;
    }

    public Builder setApk(Artifact apk) {
      Preconditions.checkState(this.apk == null);
      this.apk = apk;
      return this;
    }

    public Builder setManifest(Artifact manifest) {
      Preconditions.checkState(this.manifest == null);
      this.manifest = manifest;
      return this;
    }

    public Builder setGeneratedManifest(Artifact manifest) {
      Preconditions.checkState(this.generatedManifest == null);
      this.generatedManifest = manifest;
      return this;
    }

    public Builder setIdlClassJar(@Nullable Artifact idlClassJar) {
      Preconditions.checkState(this.idlClassJar == null);
      this.idlClassJar = idlClassJar;
      return this;
    }

    public Builder setIdlSourceJar(@Nullable Artifact idlSourceJar) {
      Preconditions.checkState(this.idlSourceJar == null);
      this.idlSourceJar = idlSourceJar;
      return this;
    }

    public Builder setResourceJarJavaOutput(JavaOutput resourceJarJavaOutput) {
      this.resourceJarJavaOutput = resourceJarJavaOutput;
      return this;
    }

    public Builder setResourceApk(Artifact resourceApk) {
      this.resourceApk = resourceApk;
      return this;
    }

    public Builder setAar(Artifact aar) {
      this.aar = aar;
      return this;
    }

    public Builder setNativeLibs(Map<String, NestedSet<Artifact>> nativeLibs) {
      this.nativeLibs = nativeLibs;
      return this;
    }

    public Builder addIdlImportRoot(String idlImportRoot) {
      this.idlImportRoot = idlImportRoot;
      return this;
    }

    /** Add "idl_srcs" contents. */
    public Builder addIdlSrcs(Collection<Artifact> idlSrcs) {
      this.idlSrcs.addAll(idlSrcs);
      return this;
    }

    /** Add the java files generated from "idl_srcs". */
    public Builder addIdlGeneratedJavaFiles(Collection<Artifact> idlGeneratedJavaFiles) {
      this.idlGeneratedJavaFiles.addAll(idlGeneratedJavaFiles);
      return this;
    }

    public Builder addAllApksUnderTest(Iterable<Artifact> apks) {
      Iterables.addAll(apksUnderTest, apks);
      return this;
    }
  }

  @Nullable private final String javaPackage;
  @Nullable private final String idlImportRoot;
  @Nullable private final Artifact manifest;
  @Nullable private final Artifact generatedManifest;
  @Nullable private final Artifact signedApk;
  @Nullable private final Artifact idlClassJar;
  @Nullable private final Artifact idlSourceJar;
  @Nullable private final JavaOutput resourceJarJavaOutput;
  @Nullable private final Artifact resourceApk;
  private final boolean definesAndroidResources;
  @Nullable private final Artifact aar;
  private final ImmutableCollection<Artifact> idlSrcs;
  private final ImmutableCollection<Artifact> idlGeneratedJavaFiles;
  private final ImmutableCollection<Artifact> apksUnderTest;
  private final ImmutableMap<String, NestedSet<Artifact>> nativeLibs;

  public AndroidIdeInfoProvider(
      @Nullable String javaPackage,
      @Nullable String idlImportRoot,
      @Nullable Artifact manifest,
      @Nullable Artifact generatedManifest,
      @Nullable Artifact signedApk,
      @Nullable Artifact idlClassJar,
      @Nullable Artifact idlSourceJar,
      @Nullable JavaOutput resourceJarJavaOutput,
      boolean definesAndroidResources,
      @Nullable Artifact aar,
      ImmutableCollection<Artifact> idlSrcs,
      ImmutableCollection<Artifact> idlGeneratedJavaFiles,
      ImmutableCollection<Artifact> apksUnderTest,
      ImmutableMap<String, NestedSet<Artifact>> nativeLibs,
      @Nullable Artifact resourceApk) {
    this.javaPackage = javaPackage;
    this.idlImportRoot = idlImportRoot;
    this.manifest = manifest;
    this.generatedManifest = generatedManifest;
    this.signedApk = signedApk;
    this.idlClassJar = idlClassJar;
    this.idlSourceJar = idlSourceJar;
    this.resourceJarJavaOutput = resourceJarJavaOutput;
    this.definesAndroidResources = definesAndroidResources;
    this.aar = aar;
    this.idlSrcs = idlSrcs;
    this.idlGeneratedJavaFiles = idlGeneratedJavaFiles;
    this.apksUnderTest = apksUnderTest;
    this.nativeLibs = nativeLibs;
    this.resourceApk = resourceApk;
  }

  @Override
  public Provider getProvider() {
    return PROVIDER;
  }

  @Override
  @Nullable
  public String getJavaPackage() {
    return javaPackage;
  }

  @Override
  @Nullable
  public Artifact getManifest() {
    return manifest;
  }

  @Override
  @Nullable
  public Artifact getGeneratedManifest() {
    return generatedManifest;
  }

  @Override
  public boolean definesAndroidResources() {
    return this.definesAndroidResources;
  }

  @Override
  @Nullable
  public String getIdlImportRoot() {
    return idlImportRoot;
  }

  @Override
  @Nullable
  public Artifact getSignedApk() {
    return signedApk;
  }

  @Override
  @Nullable
  public Artifact getIdlClassJar() {
    return idlClassJar;
  }

  @Override
  @Nullable
  public Artifact getIdlSourceJar() {
    return idlSourceJar;
  }

  @Override
  @Nullable
  public JavaOutput getResourceJarJavaOutput() {
    return resourceJarJavaOutput;
  }

  @Override
  @Nullable
  public Artifact getAar() {
    return aar;
  }

  @Override
  @Nullable
  public Artifact getResourceApk() {
    return resourceApk;
  }

  @Override
  public ImmutableCollection<Artifact> getIdlSrcs() {
    return idlSrcs;
  }

  @Override
  public ImmutableCollection<Artifact> getIdlGeneratedJavaFiles() {
    return idlGeneratedJavaFiles;
  }

  @Override
  public ImmutableCollection<Artifact> getApksUnderTest() {
    return apksUnderTest;
  }

  public ImmutableMap<String, NestedSet<Artifact>> getNativeLibs() {
    return nativeLibs;
  }

  @Override
  public ImmutableMap<String, Depset> getNativeLibsStarlark() {
    ImmutableMap.Builder<String, Depset> builder = ImmutableMap.builder();
    for (Map.Entry<String, NestedSet<Artifact>> entry : getNativeLibs().entrySet()) {
      builder.put(entry.getKey(), Depset.of(Artifact.TYPE, entry.getValue()));
    }
    return builder.build();
  }

  /** Provider class for {@link AndroidIdeInfoProvider} objects. */
  public static class Provider extends BuiltinProvider<AndroidIdeInfoProvider>
      implements AndroidIdeInfoProviderApi.Provider<Artifact, JavaOutput> {
    private Provider() {
      super(NAME, AndroidIdeInfoProvider.class);
    }

    @Override
    public AndroidIdeInfoProvider createInfo(
        Object javaPackage,
        Object manifest,
        Object generatedManifest,
        Object idlImportRoot,
        Sequence<?> idlSrcs, // <Artifact>
        Sequence<?> idlGeneratedJavaFiles, // <Artifact>
        Object idlSourceJar,
        Object idlClassJar,
        boolean definesAndroidResources,
        Object resourceJar,
        Object resourceApk,
        Object signedApk,
        Object aar,
        Sequence<?> apksUnderTest, // <Artifact>
        Dict<?, ?> nativeLibs) // <String, Depset>
        throws EvalException {
      Map<String, Depset> nativeLibsMap =
          Dict.cast(nativeLibs, String.class, Depset.class, "native_libs");

      ImmutableMap.Builder<String, NestedSet<Artifact>> builder = ImmutableMap.builder();
      for (Map.Entry<String, Depset> entry : nativeLibsMap.entrySet()) {
        builder.put(entry.getKey(), Depset.cast(entry.getValue(), Artifact.class, "native_libs"));
      }
      return new AndroidIdeInfoProvider(
          fromNoneable(javaPackage, String.class),
          fromNoneable(idlImportRoot, String.class),
          fromNoneable(manifest, Artifact.class),
          fromNoneable(generatedManifest, Artifact.class),
          fromNoneable(signedApk, Artifact.class),
          fromNoneable(idlClassJar, Artifact.class),
          fromNoneable(idlSourceJar, Artifact.class),
          fromNoneable(resourceJar, JavaOutput.class),
          definesAndroidResources,
          fromNoneable(aar, Artifact.class),
          ImmutableList.copyOf(Sequence.cast(idlSrcs, Artifact.class, "idl_srcs")),
          ImmutableList.copyOf(
              Sequence.cast(idlGeneratedJavaFiles, Artifact.class, "idl_generated_java_files")),
          ImmutableList.copyOf(Sequence.cast(apksUnderTest, Artifact.class, "apks_under_test")),
          builder.build(),
          fromNoneable(resourceApk, Artifact.class));
    }
  }
}
