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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.rules.android.AndroidConfiguration.AndroidAaptVersion;

/**
 * Builder for creating $android_resource_validator action. This action validates merged resources
 * of an android_library via aapt, and writes out an R.txt file (mostly to serve as a dependency for
 * the android_binary -- otherwise the merger step could have generated the R.txt).
 *
 * <p>This is split from merging, so that it can happen off of the compilation critical path.
 */
public class AndroidResourceValidatorActionBuilder {

  // Inputs
  private ParsedAndroidResources primary;
  private Artifact mergedResources;

  // Outputs
  private Artifact rTxtOut;
  private Artifact sourceJarOut;

  // Flags
  private String customJavaPackage;
  private boolean debug;
  private Artifact staticLibraryOut;
  private ResourceDependencies resourceDeps;
  private Artifact aapt2SourceJarOut;
  private Artifact aapt2RTxtOut;
  private Artifact compiledSymbols;
  private Artifact apkOut;

  public AndroidResourceValidatorActionBuilder setStaticLibraryOut(Artifact staticLibraryOut) {
    this.staticLibraryOut = staticLibraryOut;
    return this;
  }

  /** The primary resource container. We mostly propagate its values, but update the R.txt. */
  private AndroidResourceValidatorActionBuilder withPrimary(ParsedAndroidResources primary) {
    this.primary = primary;
    return this;
  }

  public AndroidResourceValidatorActionBuilder setMergedResources(Artifact mergedResources) {
    this.mergedResources = mergedResources;
    return this;
  }

  public AndroidResourceValidatorActionBuilder setJavaPackage(String customJavaPackage) {
    this.customJavaPackage = customJavaPackage;
    return this;
  }

  public AndroidResourceValidatorActionBuilder setDebug(boolean debug) {
    this.debug = debug;
    return this;
  }

  public AndroidResourceValidatorActionBuilder setRTxtOut(Artifact rTxtOut) {
    this.rTxtOut = rTxtOut;
    return this;
  }

  public AndroidResourceValidatorActionBuilder setSourceJarOut(Artifact sourceJarOut) {
    this.sourceJarOut = sourceJarOut;
    return this;
  }

  /** Used to add the static library from the dependencies. */
  public AndroidResourceValidatorActionBuilder withDependencies(ResourceDependencies resourceDeps) {
    this.resourceDeps = resourceDeps;
    return this;
  }

  public AndroidResourceValidatorActionBuilder setAapt2RTxtOut(Artifact aapt2RTxtOut) {
    this.aapt2RTxtOut = aapt2RTxtOut;
    return this;
  }

  public AndroidResourceValidatorActionBuilder setAapt2SourceJarOut(Artifact aapt2SourceJarOut) {
    this.aapt2SourceJarOut = aapt2SourceJarOut;
    return this;
  }

  private void build(AndroidDataContext dataContext) {
    if (rTxtOut != null) {
      createValidateAction(dataContext);
    }

    if (compiledSymbols != null) {
      createLinkStaticLibraryAction(dataContext);
    }
  }

  public ValidatedAndroidResources build(
      AndroidDataContext dataContext, MergedAndroidResources merged) {
    withPrimary(merged).build(dataContext);

    return ValidatedAndroidResources.of(
        merged, rTxtOut, sourceJarOut, apkOut, aapt2RTxtOut, aapt2SourceJarOut, staticLibraryOut);
  }

  public AndroidResourceValidatorActionBuilder setCompiledSymbols(Artifact compiledSymbols) {
    this.compiledSymbols = compiledSymbols;
    return this;
  }

  public AndroidResourceValidatorActionBuilder setApkOut(Artifact apkOut) {
    this.apkOut = apkOut;
    return this;
  }

  /**
   * This creates a static library using aapt2. It also generates a source jar and R.txt from aapt.
   *
   * <p>This allows the link action to replace the validate action for builds that use aapt2, as
   * opposed to executing both actions.
   */
  private void createLinkStaticLibraryAction(AndroidDataContext dataContext) {
    Preconditions.checkNotNull(resourceDeps);

    BusyBoxActionBuilder builder =
        BusyBoxActionBuilder.create(dataContext, "LINK_STATIC_LIBRARY")
            .addAapt(AndroidAaptVersion.AAPT2)
            .addInput("--libraries", dataContext.getSdk().getAndroidJar())
            .addInput("--compiled", compiledSymbols)
            .addInput("--manifest", primary.getManifest())

            // Sets an alternative java package for the generated R.java
            // this allows android rules to generate resources outside of the java{,tests} tree.
            .maybeAddFlag("--packageForR", customJavaPackage);

    if (!resourceDeps.getTransitiveCompiledSymbols().isEmpty()) {
      builder.addTransitiveVectoredInput(
          "--compiledDep", resourceDeps.getTransitiveCompiledSymbols());
    }

    builder
        .addOutput("--sourceJarOut", aapt2SourceJarOut)
        .addOutput("--rTxtOut", aapt2RTxtOut)
        .addOutput("--staticLibraryOut", staticLibraryOut)
        .buildAndRegister("Linking static android resource library", "AndroidResourceLink");
  }

  private void createValidateAction(AndroidDataContext dataContext) {
    BusyBoxActionBuilder.create(dataContext, "VALIDATE")
        .maybeAddFlag("--buildToolsVersion", dataContext.getSdk().getBuildToolsVersion())
        .addAapt(AndroidAaptVersion.AAPT)
        .addAndroidJar()
        .addInput("--mergedResources", mergedResources)
        .addInput("--manifest", primary.getManifest())
        .maybeAddFlag("--debug", debug)

        // Sets an alternative java package for the generated R.java
        // this allows android rules to generate resources outside of the java{,tests} tree.
        .maybeAddFlag("--packageForR", customJavaPackage)
        .addOutput("--rOutput", rTxtOut)
        .addOutput("--srcJarOutput", sourceJarOut)
        .maybeAddOutput("--packagePath", apkOut)
        .buildAndRegister("Validating Android resources", "AndroidResourceValidator");
  }
}
