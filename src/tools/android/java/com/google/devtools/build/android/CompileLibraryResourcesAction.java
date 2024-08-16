// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import com.google.common.base.Preconditions;
import com.google.devtools.build.android.Converters.CompatExistingPathConverter;
import com.google.devtools.build.android.Converters.CompatPathConverter;
import com.google.devtools.build.android.Converters.CompatUnvalidatedAndroidDirectoriesConverter;
import com.google.devtools.build.android.aapt2.Aapt2ConfigOptions;
import com.google.devtools.build.android.aapt2.ResourceCompiler;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.ShellQuotedParamsFilePreProcessor;
import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.logging.Logger;

/** Compiles resources using aapt2 and archives them to zip. */
@Parameters(separators = "= ")
public class CompileLibraryResourcesAction {
  /** Flag specifications for this action. */
  public static final class Options extends OptionsBaseWithResidue {

    @Parameter(
        names = "--resources",
        converter = CompatUnvalidatedAndroidDirectoriesConverter.class,
        description = "The resources to compile with aapt2.")
    public UnvalidatedAndroidDirectories resources;

    @Parameter(
        names = "--output",
        converter = CompatPathConverter.class,
        description = "Path to write the zip of compiled resources.")
    public Path output;

    @Parameter(
        names = "--packagePath",
        description =
            "The package path of the library being processed."
                + " This value is required for processing data binding.")
    public String packagePath;

    @Parameter(
        names = "--manifest",
        converter = CompatExistingPathConverter.class,
        description =
            "The manifest of the library being processed."
                + " This value is required for processing data binding.")
    public Path manifest;

    @Parameter(
        names = "--dataBindingInfoOut",
        converter = CompatPathConverter.class,
        description =
            "Path for the derived data binding metadata."
                + " This value is required for processing data binding.")
    public Path dataBindingInfoOut;
  }

  static final Logger logger = Logger.getLogger(CompileLibraryResourcesAction.class.getName());

  public static void main(String[] args) throws Exception {
    Options options = new Options();
    JCommander jc = new JCommander(options);
    jc.parse(args);
    List<String> residue = options.getResidue();
    OptionsParser optionsParser =
        OptionsParser.builder()
            .optionsClasses(Aapt2ConfigOptions.class, ResourceProcessorCommonOptions.class)
            .argsPreProcessor(new ShellQuotedParamsFilePreProcessor(FileSystems.getDefault()))
            .build();
    optionsParser.parseAndExitUponError(residue.toArray(new String[0]));

    Aapt2ConfigOptions aapt2Options = optionsParser.getOptions(Aapt2ConfigOptions.class);

    Preconditions.checkNotNull(options.resources);
    Preconditions.checkNotNull(options.output);
    Preconditions.checkNotNull(aapt2Options.aapt2);

    try (ExecutorServiceCloser executorService = ExecutorServiceCloser.createWithFixedPoolOf(15);
        ScopedTemporaryDirectory scopedTmp =
            new ScopedTemporaryDirectory("android_resources_tmp")) {
      final Path tmp = scopedTmp.getPath();
      final Path databindingResourcesRoot =
          Files.createDirectories(tmp.resolve("android_data_binding_resources"));
      final Path compiledResources = Files.createDirectories(tmp.resolve("compiled"));

      final ResourceCompiler compiler =
          ResourceCompiler.create(
              executorService,
              compiledResources,
              aapt2Options.aapt2,
              aapt2Options.buildToolsVersion,
              aapt2Options.generatePseudoLocale);
      options
          .resources
          .toData(options.manifest)
          .processDataBindings(
              options.dataBindingInfoOut,
              options.packagePath,
              databindingResourcesRoot,
              aapt2Options.useDataBindingAndroidX)
          .compile(compiler, compiledResources)
          .copyResourcesZipTo(options.output);
    } catch (IOException | ExecutionException | InterruptedException e) {
      logger.log(java.util.logging.Level.SEVERE, "Unexpected", e);
      throw e;
    }
  }
}
