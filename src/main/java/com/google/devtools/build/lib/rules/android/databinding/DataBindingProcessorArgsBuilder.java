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

/** Builder for annotation processor arguments that can be passed down to data binding. */
public class DataBindingProcessorArgsBuilder {
  private final boolean useUpdatedArgs;
  private final ImmutableList.Builder<String> flags = ImmutableList.builder();

  public DataBindingProcessorArgsBuilder(boolean useUpdatedArgs) {
    this.useUpdatedArgs = useUpdatedArgs;
  }

  /** Sets directories where data binding reads its input and also writes its output */
  public DataBindingProcessorArgsBuilder metadataOutputDir(String metadataOutputDir) {
    if (useUpdatedArgs) {
      flags.add(
          createProcessorFlag(
              "dependencyArtifactsDir",
              metadataOutputDir + "/" + DataBinding.DEP_METADATA_INPUT_DIR));
      flags.add(
          createProcessorFlag(
              "aarOutDir", metadataOutputDir + "/" + DataBinding.METADATA_OUTPUT_DIR));
    } else {
      flags.add(createProcessorFlag("bindingBuildFolder", metadataOutputDir));
      flags.add(createProcessorFlag("generationalFileOutDir", metadataOutputDir));
    }

    return this;
  }

  /** Path to the Android SDK installation (if available). */
  public DataBindingProcessorArgsBuilder sdkDir(String sdkDir) {
    flags.add(createProcessorFlag("sdkDir", sdkDir));
    return this;
  }

  /** Whether the current rule is a library or binary. */
  public DataBindingProcessorArgsBuilder binary(boolean isBinary) {
    flags.add(createProcessorFlag("artifactType", isBinary ? "APPLICATION" : "LIBRARY"));
    return this;
  }

  /**
   * Where data binding exports the list of classes it has created and that should be removed after
   * compilation. Not used in blaze.
   */
  public DataBindingProcessorArgsBuilder exportClassListTo(String path) {
    if (useUpdatedArgs) {
      flags.add(createProcessorFlag("exportClassListOutFile", path));
    } else {
      flags.add(createProcessorFlag("exportClassListTo", path));
    }
    return this;
  }

  /** The Java package for the current rule. This should match the AndroidManifest package. */
  public DataBindingProcessorArgsBuilder modulePackage(String pkg) {
    flags.add(createProcessorFlag("modulePackage", pkg));
    return this;
  }

  /** Min SDK defined for the app */
  public DataBindingProcessorArgsBuilder minApi(String minApi) {
    flags.add(createProcessorFlag("minApi", minApi));
    return this;
  }

  /**
   * If set, data binding prints its errors in an encoded format that can be consumed by Android
   * Studio
   */
  public DataBindingProcessorArgsBuilder printEncodedErrors() {
    flags.add(createProcessorFlag("printEncodedErrors", "1"));
    return this;
  }

  /** Call this method if we should use Data Binding compiler v2 */
  public DataBindingProcessorArgsBuilder enableV2() {
    flags.add(createProcessorFlag("enableV2", "1"));
    return this;
  }

  /**
   * The location where the processor wrote the information about the layout classes it has
   * generated during layout processing.
   */
  public DataBindingProcessorArgsBuilder classLogDir(String classLogDir) {
    flags.add(createProcessorFlag("classLogDir", classLogDir));
    return this;
  }

  /**
   * The location where the processor wrote the information about the layout classes it has
   * generated during layout processing.
   */
  public DataBindingProcessorArgsBuilder classLogDir(Artifact classLogDir) {
    flags.add(createProcessorFlag("classLogDir", classLogDir));
    return this;
  }

  /**
   * The path where data binding's resource processor wrote its output (the data binding XML
   * expressions). The annotation processor reads this file to translate that XML into Java.
   */
  public DataBindingProcessorArgsBuilder layoutInfoDir(String layoutInfoDir) {
    if (useUpdatedArgs) {
      flags.add(createProcessorFlag("layoutInfoDir", layoutInfoDir));
    } else {
      flags.add(createProcessorFlag("xmlOutDir", layoutInfoDir));
    }
    return this;
  }

  /**
   * The path where data binding's resource processor wrote its output (the data binding XML
   * expressions). The annotation processor reads this file to translate that XML into Java.
   */
  public DataBindingProcessorArgsBuilder layoutInfoDir(Artifact layoutInfoDir) {
    flags.add(createProcessorFlag("layoutInfoDir", layoutInfoDir));
    return this;
  }

  /** Builds annotation processor arguments for Data Binding */
  public ImmutableList<String> build() {
    return flags.build();
  }

  /** Turns a key/value pair into a javac annotation processor flag received by data binding. */
  private static String createProcessorFlag(String flag, String value) {
    return String.format("-Aandroid.databinding.%s=%s", flag, value);
  }

  /** Turns a key/value pair into a javac annotation processor flag received by data binding. */
  private static String createProcessorFlag(String flag, Artifact value) {
    return createProcessorFlag(flag, value.getExecPathString());
  }
}
