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

package com.google.devtools.build.android;

import android.databinding.AndroidDataBinding;
import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Streams;
import com.google.devtools.build.android.AndroidResourceProcessor.AaptConfigOptions;
import com.google.devtools.build.android.Converters.CompatPathConverter;
import java.nio.file.Path;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

/** Generates databinding base classes for an Android target. */
public final class GenerateDatabindingBaseClassesAction {

  /** Options for GenerateDatabindingBaseClassesAction. */
  @Parameters(separators = "= ")
  public static final class Options extends OptionsBaseWithResidue {
    @Parameter(
        names = "--layoutInfoFiles",
        converter = CompatPathConverter.class,
        description = "Path to layout-info.zip file produced by databinding processor")
    public Path layoutInfoFile;

    @Parameter(names = "--package", description = "Package name of the android target")
    public String packageName;

    @Parameter(
        names = "--classInfoOut",
        converter = CompatPathConverter.class,
        description = "Path to write classInfo.zip file")
    public Path classInfoOut;

    @Parameter(
        names = "--sourceOut",
        converter = CompatPathConverter.class,
        description = "Path to write databinding base classes to be used in Java compilation")
    public Path sourceOut;

    @Parameter(
        names = "--dependencyClassInfoList",
        converter = CompatPathConverter.class,
        description = "List of dependency class info zip files")
    public List<Path> dependencyClassInfoList = ImmutableList.of();
  }

  static final Logger logger =
      Logger.getLogger(GenerateDatabindingBaseClassesAction.class.getName());

  public static void main(String[] args) throws Exception {
    Options options = new Options();
    AaptConfigOptions aaptConfigOptions = new AaptConfigOptions();
    Object[] allOptions =
        new Object[] {options, aaptConfigOptions, new ResourceProcessorCommonOptions()};
    JCommander jc = new JCommander(allOptions);
    String[] preprocessedArgs = AndroidOptionsUtils.runArgFilePreprocessor(jc, args);
    String[] normalizedArgs =
        AndroidOptionsUtils.normalizeBooleanOptions(allOptions, preprocessedArgs);
    jc.parse(normalizedArgs);

    if (options.layoutInfoFile == null) {
      throw new IllegalArgumentException("--layoutInfoFiles is required");
    }

    if (options.packageName == null) {
      throw new IllegalArgumentException("--packageName is required");
    }

    if (options.classInfoOut == null) {
      throw new IllegalArgumentException("--classInfoOut is required");
    }

    if (options.sourceOut == null) {
      throw new IllegalArgumentException("--sourceOut is required");
    }

    final List<Path> dependencyClassInfoList =
        options.dependencyClassInfoList == null
            ? ImmutableList.of()
            : options.dependencyClassInfoList;

    final ImmutableList.Builder<String> dbArgsBuilder =
        ImmutableList.<String>builder()
            .add("GEN_BASE_CLASSES")
            .add("-layoutInfoFiles")
            .add(options.layoutInfoFile.toString())
            .add("-package", options.packageName)
            .add("-classInfoOut")
            .add(options.classInfoOut.toString())
            .add("-sourceOut")
            .add(options.sourceOut.toString())
            .add("-zipSourceOutput")
            .add("true")
            .add("-useAndroidX")
            .add(Boolean.toString(aaptConfigOptions.useDataBindingAndroidX));

    dependencyClassInfoList.forEach(
        classInfo -> dbArgsBuilder.add("-dependencyClassInfoList").add(classInfo.toString()));

    try {
      AndroidDataBinding.main(
          Streams.mapWithIndex(
                  dbArgsBuilder.build().stream(), (arg, index) -> index == 0 ? arg : arg + " ")
              .toArray(String[]::new));
    } catch (RuntimeException e) {
      logger.log(Level.SEVERE, "Unexpected", e);
      throw e;
    }
  }

  private GenerateDatabindingBaseClassesAction() {}
}
