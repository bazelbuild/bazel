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

import static java.util.stream.Collectors.joining;

import com.google.common.collect.Iterables;
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.rules.android.AndroidConfiguration.AndroidAaptVersion;
import com.google.devtools.build.lib.rules.android.databinding.DataBindingContext;
import com.google.devtools.build.lib.vfs.PathFragment;
import javax.annotation.Nullable;

/** Builder for creating $android_resource_parser action. */
public class AndroidResourceParsingActionBuilder {

  // These are only needed when parsing resources with data binding
  @Nullable private Artifact manifest;
  @Nullable private String javaPackage;

  private AndroidResources resources = AndroidResources.empty();
  private AndroidAssets assets = AndroidAssets.empty();

  // The symbols file is a required output
  @Nullable private Artifact output;

  // Optional outputs
  @Nullable private Artifact compiledSymbols;
  @Nullable private Artifact dataBindingInfoZip;

  /** Set the artifact location for the output protobuf. */
  public AndroidResourceParsingActionBuilder setOutput(Artifact output) {
    this.output = output;
    return this;
  }

  /** Sets the manifest. Will be ignored except when parsing resources with data binding. */
  public AndroidResourceParsingActionBuilder setManifest(@Nullable Artifact manifest) {
    this.manifest = manifest;
    return this;
  }

  /** Sets the Java package. Will be ignored except when parsing resources with data binding. */
  public AndroidResourceParsingActionBuilder setJavaPackage(@Nullable String javaPackage) {
    this.javaPackage = javaPackage;
    return this;
  }

  public AndroidResourceParsingActionBuilder setResources(AndroidResources resources) {
    this.resources = resources;
    return this;
  }

  public AndroidResourceParsingActionBuilder setAssets(AndroidAssets assets) {
    this.assets = assets;
    return this;
  }

  public AndroidResourceParsingActionBuilder setCompiledSymbolsOutput(
      @Nullable Artifact compiledSymbols) {
    this.compiledSymbols = compiledSymbols;
    return this;
  }

  public AndroidResourceParsingActionBuilder setDataBindingInfoZip(Artifact dataBindingInfoZip) {
    this.dataBindingInfoZip = dataBindingInfoZip;
    return this;
  }

  private static String convertRoots(Iterable<PathFragment> roots) {
    return Streams.stream(roots).map(Object::toString).collect(joining("#"));
  }

  private void build(AndroidDataContext dataContext) {
    String resourceDirectories =
        convertRoots(resources.getResourceRoots()) + ":" + convertRoots(assets.getAssetRoots());
    Iterable<Artifact> resourceArtifacts =
        Iterables.concat(assets.getAssets(), resources.getResources());

    if (output != null) {
      BusyBoxActionBuilder.create(dataContext, "PARSE")
          .addInput("--primaryData", resourceDirectories, resourceArtifacts)
          .addOutput("--output", output)
          .buildAndRegister("Parsing Android resources", "AndroidResourceParser");
    }

    if (compiledSymbols != null) {
      BusyBoxActionBuilder compiledBuilder =
          BusyBoxActionBuilder.create(dataContext, "COMPILE_LIBRARY_RESOURCES")
              .addAapt(AndroidAaptVersion.AAPT2)
              .addInput("--resources", resourceDirectories, resourceArtifacts)
              .addOutput("--output", compiledSymbols);

      if (dataBindingInfoZip != null) {
        compiledBuilder
            .addInput("--manifest", manifest)
            .maybeAddFlag("--packagePath", javaPackage)
            .addOutput("--dataBindingInfoOut", dataBindingInfoZip);
      }

      compiledBuilder.buildAndRegister("Compiling Android resources", "AndroidResourceCompiler");
    }
  }

  /**
   * Builds and registers the action, and returns a copy of the passed resources with artifacts for
   * parsed and compiled information.
   */
  public ParsedAndroidResources build(
      AndroidDataContext dataContext,
      AndroidResources androidResources,
      StampedAndroidManifest manifest,
      DataBindingContext dataBindingContext) {
    if (dataBindingInfoZip != null) {
      // Manifest information is needed for data binding
      setManifest(manifest.getManifest());
      setJavaPackage(manifest.getPackage());
    }

    setResources(androidResources);
    build(dataContext);

    return ParsedAndroidResources.of(
        androidResources,
        output,
        compiledSymbols,
        dataContext.getLabel(),
        manifest,
        dataBindingContext);
  }

  public ParsedAndroidAssets build(AndroidDataContext dataContext, AndroidAssets assets) {
    setAssets(assets);
    build(dataContext);

    return ParsedAndroidAssets.of(assets, output, compiledSymbols, dataContext.getLabel());
  }
}
