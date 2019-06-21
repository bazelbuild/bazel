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

import com.android.builder.dependency.SymbolFileProvider;
import com.android.resources.ResourceType;
import com.google.common.base.Optional;
import com.google.common.base.Stopwatch;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.android.AndroidResourceProcessor.AaptConfigOptions;
import com.google.devtools.build.android.Converters.DependencyAndroidDataListConverter;
import com.google.devtools.build.android.Converters.PathConverter;
import com.google.devtools.build.android.resources.RClassGenerator;
import com.google.devtools.build.android.resources.ResourceSymbols;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.ShellQuotedParamsFilePreProcessor;
import java.io.Closeable;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;

/** This action generates consistent ids R.class files for use in robolectric tests. */
public class GenerateRobolectricResourceSymbolsAction {

  private static final Logger logger =
      Logger.getLogger(GenerateRobolectricResourceSymbolsAction.class.getName());

  private static final class WriteLibraryRClass implements Callable<Boolean> {
    private final Map.Entry<String, Collection<ListenableFuture<ResourceSymbols>>>
        librarySymbolEntry;
    private final RClassGenerator generator;

    private WriteLibraryRClass(
        Map.Entry<String, Collection<ListenableFuture<ResourceSymbols>>> librarySymbolEntry,
        RClassGenerator generator) {
      this.librarySymbolEntry = librarySymbolEntry;
      this.generator = generator;
    }

    @Override
    public Boolean call() throws Exception {
      List<ResourceSymbols> resourceSymbolsList = new ArrayList<>();
      for (final ListenableFuture<ResourceSymbols> resourceSymbolsReader :
          librarySymbolEntry.getValue()) {
        resourceSymbolsList.add(resourceSymbolsReader.get());
      }

      generator.write(
          librarySymbolEntry.getKey(), ResourceSymbols.merge(resourceSymbolsList).asInitializers());
      return true;
    }
  }

  /** Flag specifications for this action. */
  public static final class Options extends OptionsBase {

    @Option(
      name = "data",
      defaultValue = "",
      converter = DependencyAndroidDataListConverter.class,
      category = "input",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "Data dependencies. The expected format is "
              + DependencyAndroidData.EXPECTED_FORMAT
              + "[&...]"
    )
    public List<DependencyAndroidData> data;

    @Option(
      name = "classJarOutput",
      defaultValue = "null",
      converter = PathConverter.class,
      category = "output",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Path for the generated java class jar."
    )
    public Path classJarOutput;

    @Option(
      name = "targetLabel",
      defaultValue = "null",
      category = "input",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "A label to add to the output jar's manifest as 'Target-Label'"
    )
    public String targetLabel;

    @Option(
      name = "injectingRuleKind",
      defaultValue = "null",
      category = "input",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "A string to add to the output jar's manifest as 'Injecting-Rule-Kind'"
    )
    public String injectingRuleKind;
  }

  public static void main(String[] args) throws Exception {

    final Stopwatch timer = Stopwatch.createStarted();
    OptionsParser optionsParser =
        OptionsParser.builder()
            .optionsClasses(Options.class, AaptConfigOptions.class)
            .argsPreProcessor(new ShellQuotedParamsFilePreProcessor(FileSystems.getDefault()))
            .build();
    optionsParser.parseAndExitUponError(args);
    AaptConfigOptions aaptConfigOptions = optionsParser.getOptions(AaptConfigOptions.class);
    Options options = optionsParser.getOptions(Options.class);

    try (ScopedTemporaryDirectory scopedTmp =
        new ScopedTemporaryDirectory("robolectric_resources_tmp")) {
      Path tmp = scopedTmp.getPath();
      Path generatedSources = Files.createDirectories(tmp.resolve("generated_resources"));
      // The reported availableProcessors may be higher than the actual resources
      // (on a shared system). On the other hand, a lot of the work is I/O, so it's not completely
      // CPU bound. As a compromise, divide by 2 the reported availableProcessors.
      int numThreads = Math.max(1, Runtime.getRuntime().availableProcessors() / 2);
      ListeningExecutorService executorService =
          MoreExecutors.listeningDecorator(Executors.newFixedThreadPool(numThreads));
      try (Closeable closeable = ExecutorServiceCloser.createWith(executorService)) {

        logger.fine(String.format("Setup finished at %sms", timer.elapsed(TimeUnit.MILLISECONDS)));

        final PlaceholderIdFieldInitializerBuilder robolectricIds =
            PlaceholderIdFieldInitializerBuilder.from(aaptConfigOptions.androidJar);
        ParsedAndroidData.loadedFrom(
                options.data, executorService, AndroidParsedDataDeserializer.create())
            .writeResourcesTo(
                new AndroidResourceSymbolSink() {

                  @Override
                  public void acceptSimpleResource(ResourceType type, String name) {
                    robolectricIds.addSimpleResource(type, name);
                  }

                  @Override
                  public void acceptPublicResource(
                      ResourceType type, String name, Optional<Integer> value) {
                    robolectricIds.addPublicResource(type, name, value);
                  }

                  @Override
                  public void acceptStyleableResource(
                      FullyQualifiedName key, Map<FullyQualifiedName, Boolean> attrs) {
                    robolectricIds.addStyleableResource(key, attrs);
                  }
                });

        final RClassGenerator generator =
            RClassGenerator.with(generatedSources, robolectricIds.build(), false);

        List<SymbolFileProvider> libraries = new ArrayList<>();
        for (DependencyAndroidData dataDep : options.data) {
          SymbolFileProvider library = dataDep.asSymbolFileProvider();
          libraries.add(library);
        }
        List<ListenableFuture<Boolean>> writeSymbolsTask = new ArrayList<>();
        for (final Map.Entry<String, Collection<ListenableFuture<ResourceSymbols>>>
            librarySymbolEntry :
                ResourceSymbols.loadFrom(libraries, executorService, null).asMap().entrySet()) {
          writeSymbolsTask.add(
              executorService.submit(new WriteLibraryRClass(librarySymbolEntry, generator)));
        }
        FailedFutureAggregator.forIOExceptionsWithMessage("Errors writing symbols.")
            .aggregateAndMaybeThrow(writeSymbolsTask);
      }

      logger.fine(String.format("Merging finished at %sms", timer.elapsed(TimeUnit.MILLISECONDS)));

      AndroidResourceOutputs.createClassJar(
          generatedSources, options.classJarOutput, options.targetLabel, options.injectingRuleKind);
      logger.fine(
          String.format("Create classJar finished at %sms", timer.elapsed(TimeUnit.MILLISECONDS)));

    } catch (Exception e) {
      logger.log(Level.SEVERE, "Unexpected", e);
      throw e;
    }
    logger.fine(String.format("Resources merged in %sms", timer.elapsed(TimeUnit.MILLISECONDS)));
  }
}
