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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider.OutputJar;
import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;

/** An Android target provider to provide Android-specific info to IDEs. */
@Immutable
public final class AndroidIdeInfoProvider implements TransitiveInfoProvider {

  /** Builder for {@link AndroidIdeInfoProvider} */
  public static class Builder {
    private Artifact manifest = null;
    private Artifact generatedManifest = null;
    private Artifact apk = null;
    private Artifact resourceApk = null;
    private Artifact idlClassJar = null;
    private Artifact idlSourceJar = null;
    private OutputJar resourceJar = null;
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
          resourceJar,
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

    public Builder setResourceJar(OutputJar resourceJar) {
      this.resourceJar = resourceJar;
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

  private final String javaPackage;
  private final String idlImportRoot;
  private final Artifact manifest;
  private final Artifact generatedManifest;
  private final Artifact signedApk;
  @Nullable private final Artifact idlClassJar;
  @Nullable private final Artifact idlSourceJar;
  @Nullable private final OutputJar resourceJar;
  @Nullable private final Artifact resourceApk;
  private final boolean definesAndroidResources;
  private final Artifact aar;
  private final ImmutableCollection<Artifact> idlSrcs;
  private final ImmutableCollection<Artifact> idlGeneratedJavaFiles;
  private final ImmutableCollection<Artifact> apksUnderTest;
  private final ImmutableMap<String, NestedSet<Artifact>> nativeLibs;

  AndroidIdeInfoProvider(
      String javaPackage,
      String idlImportRoot,
      @Nullable Artifact manifest,
      @Nullable Artifact generatedManifest,
      @Nullable Artifact signedApk,
      @Nullable Artifact idlClassJar,
      @Nullable Artifact idlSourceJar,
      @Nullable OutputJar resourceJar,
      boolean definesAndroidResources,
      @Nullable Artifact aar,
      ImmutableCollection<Artifact> idlSrcs,
      ImmutableCollection<Artifact> idlGeneratedJavaFiles,
      ImmutableCollection<Artifact> apksUnderTest,
      ImmutableMap<String, NestedSet<Artifact>> nativeLibs,
      Artifact resourceApk) {
    this.javaPackage = javaPackage;
    this.idlImportRoot = idlImportRoot;
    this.manifest = manifest;
    this.generatedManifest = generatedManifest;
    this.signedApk = signedApk;
    this.idlClassJar = idlClassJar;
    this.idlSourceJar = idlSourceJar;
    this.resourceJar = resourceJar;
    this.definesAndroidResources = definesAndroidResources;
    this.aar = aar;
    this.idlSrcs = idlSrcs;
    this.idlGeneratedJavaFiles = idlGeneratedJavaFiles;
    this.apksUnderTest = apksUnderTest;
    this.nativeLibs = nativeLibs;
    this.resourceApk = resourceApk;
  }

  /** Returns java package for this target. */
  public String getJavaPackage() {
    return javaPackage;
  }

  /** Returns the direct AndroidManifest. */
  @Nullable
  public Artifact getManifest() {
    return manifest;
  }

  /** Returns the direct generated AndroidManifest. */
  @Nullable
  public Artifact getGeneratedManifest() {
    return generatedManifest;
  }

  /**
   * Returns true if the target defined Android resources. Exposes {@link
   * LocalResourceContainer#definesAndroidResources(AttributeMap)}
   */
  public boolean definesAndroidResources() {
    return this.definesAndroidResources;
  }

  @Nullable
  public String getIdlImportRoot() {
    return idlImportRoot;
  }

  /** Returns the direct debug key signed apk, if there is one. */
  @Nullable
  public Artifact getSignedApk() {
    return signedApk;
  }

  @Nullable
  public Artifact getIdlClassJar() {
    return idlClassJar;
  }

  @Nullable
  public Artifact getIdlSourceJar() {
    return idlSourceJar;
  }

  @Nullable
  public OutputJar getResourceJar() {
    return resourceJar;
  }

  @Nullable
  public Artifact getAar() {
    return aar;
  }

  @Nullable
  public Artifact getResourceApk() {
    return resourceApk;
  }

  /** A list of sources from the "idl_srcs" attribute. */
  public ImmutableCollection<Artifact> getIdlSrcs() {
    return idlSrcs;
  }

  /** A list of java files generated from the "idl_srcs" attribute. */
  public ImmutableCollection<Artifact> getIdlGeneratedJavaFiles() {
    return idlGeneratedJavaFiles;
  }

  /** A list of the APKs related to the app under test, if any. */
  public ImmutableCollection<Artifact> getApksUnderTest() {
    return apksUnderTest;
  }

  /** A map, keyed on architecture, of the native libs for the app, if any. */
  public ImmutableMap<String, NestedSet<Artifact>> getNativeLibs() {
    return nativeLibs;
  }
}
