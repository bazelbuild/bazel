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

import com.google.common.base.Preconditions;
import com.google.devtools.build.android.Converters.ExistingPathConverter;
import com.google.devtools.build.android.Converters.PathConverter;
import com.google.devtools.build.android.Converters.UnvalidatedAndroidDirectoriesConverter;
import com.google.devtools.build.android.aapt2.Aapt2ConfigOptions;
import com.google.devtools.build.android.aapt2.ResourceCompiler;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.ShellQuotedParamsFilePreProcessor;
import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.ExecutionException;
import java.util.logging.Logger;

/** Compiles resources using aapt2 and archives them to zip. */
public class CompileLibraryResourcesAction {
  /** Flag specifications for this action. */
  public static final class Options extends OptionsBase {

    @Option(
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      name = "resources",
      defaultValue = "null",
      converter = UnvalidatedAndroidDirectoriesConverter.class,
      category = "input",
      help = "The resources to compile with aapt2."
    )
    public UnvalidatedAndroidDirectories resources;

    @Option(
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      name = "output",
      defaultValue = "null",
      converter = PathConverter.class,
      category = "output",
      help = "Path to write the zip of compiled resources."
    )
    public Path output;

    @Option(
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      name = "packagePath",
      defaultValue = "null",
      category = "input",
      help =
          "The package path of the library being processed."
              + " This value is required for processing data binding."
    )
    public String packagePath;

    @Option(
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      name = "manifest",
      defaultValue = "null",
      category = "input",
      converter = ExistingPathConverter.class,
      help =
          "The manifest of the library being processed."
              + " This value is required for processing data binding."
    )
    public Path manifest;

    @Option(
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      name = "dataBindingInfoOut",
      defaultValue = "null",
      category = "output",
      converter = PathConverter.class,
      help =
          "Path for the derived data binding metadata."
              + " This value is required for processing data binding."
    )
    public Path dataBindingInfoOut;

  }

  static final Logger logger = Logger.getLogger(CompileLibraryResourcesAction.class.getName());

  public static void main(String[] args) throws Exception {
    OptionsParser optionsParser =
        OptionsParser.newOptionsParser(Options.class, Aapt2ConfigOptions.class);
    optionsParser.enableParamsFileSupport(
        new ShellQuotedParamsFilePreProcessor(FileSystems.getDefault()));
    optionsParser.parseAndExitUponError(args);

    Options options = optionsParser.getOptions(Options.class);
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
              options.dataBindingInfoOut, options.packagePath, databindingResourcesRoot)
          .compile(compiler, compiledResources)
          .copyResourcesZipTo(options.output);
    } catch (IOException | ExecutionException | InterruptedException e) {
      logger.log(java.util.logging.Level.SEVERE, "Unexpected", e);
      throw e;
    }
  }
}
