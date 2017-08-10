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

import com.android.builder.core.VariantType;
import com.android.repository.Revision;
import com.google.common.base.Preconditions;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.android.Converters.ExistingPathConverter;
import com.google.devtools.build.android.Converters.PathConverter;
import com.google.devtools.build.android.Converters.RevisionConverter;
import com.google.devtools.build.android.aapt2.ResourceCompiler;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import java.io.Closeable;
import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.logging.Logger;

/** Compiles resources using aapt2 and archives them to zip. */
public class CompileLibraryResourcesAction {
  /** Flag specifications for this action. */
  public static final class Options extends OptionsBase {

    @Option(
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      name = "resource",
      defaultValue = "",
      allowMultiple = true,
      converter = ExistingPathConverter.class,
      category = "input",
      help = "The resources to compile with aapt2."
    )
    public List<Path> resources;

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
      name = "aapt2",
      defaultValue = "null",
      converter = ExistingPathConverter.class,
      category = "tool",
      help = "Aapt2 tool location for resource compilation."
    )
    public Path aapt2;

    @Option(
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      name = "buildToolsVersion",
      defaultValue = "null",
      converter = RevisionConverter.class,
      category = "config",
      help = "Version of the build tools (e.g. aapt) being used, e.g. 23.0.2"
    )
    public Revision buildToolsVersion;

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
    OptionsParser optionsParser = OptionsParser.newOptionsParser(Options.class);
    optionsParser.enableParamsFileSupport(FileSystems.getDefault());
    optionsParser.parseAndExitUponError(args);

    Options options = optionsParser.getOptions(Options.class);

    Preconditions.checkNotNull(options.resources);
    Preconditions.checkNotNull(options.output);
    Preconditions.checkNotNull(options.aapt2);

    try (ScopedTemporaryDirectory scopedTmp =
        new ScopedTemporaryDirectory("android_resources_tmp")) {
      final Path tmp = scopedTmp.getPath();
      final Path databindingResourcesRoot =
          Files.createDirectories(tmp.resolve("android_data_binding_resources"));
      final Path databindingMetaData =
          Files.createDirectories(tmp.resolve("android_data_binding_metadata"));
      final Path compiledResources = Files.createDirectories(tmp.resolve("compiled"));
      // The reported availableProcessors may be higher than the actual resources
      // (on a shared system). On the other hand, a lot of the work is I/O, so it's not completely
      // CPU bound. As a compromise, divide by 2 the reported availableProcessors.
      int numThreads = Math.max(1, Runtime.getRuntime().availableProcessors() / 2);
      final ListeningExecutorService executorService =
          MoreExecutors.listeningDecorator(Executors.newFixedThreadPool(numThreads));
      try (final Closeable closeable = ExecutorServiceCloser.createWith(executorService)) {
        final ResourceCompiler compiler =
            ResourceCompiler.create(
                executorService, compiledResources, options.aapt2, options.buildToolsVersion);
        for (final Path resource :
            maybeProcessDataBindings(
                databindingResourcesRoot,
                databindingMetaData,
                options.dataBindingInfoOut,
                options.manifest,
                options.packagePath,
                options.resources)) {
          compiler.queueDirectoryForCompilation(resource);
        }
        AndroidResourceOutputs.archiveCompiledResources(
            options.output,
            databindingResourcesRoot,
            compiledResources,
            compiler.getCompiledArtifacts());
      }
    }
  }

  private static List<Path> maybeProcessDataBindings(
      Path resourceRoot,
      Path databindingMetaData,
      Path dataBindingInfoOut,
      Path manifest,
      String packagePath,
      List<Path> resources)
      throws IOException {
    if (dataBindingInfoOut == null) {
      return resources;
    }

    Preconditions.checkNotNull(manifest);
    Preconditions.checkNotNull(packagePath);

    List<Path> processed = new ArrayList<>();
    for (Path resource : resources) {
      processed.add(
          AndroidResourceProcessor.processDataBindings(
              resourceRoot,
              resource,
              databindingMetaData,
              VariantType.LIBRARY,
              packagePath,
              manifest,
              false));
    }

    AndroidResourceOutputs.archiveDirectory(databindingMetaData, dataBindingInfoOut);
    return processed;
  }
}
