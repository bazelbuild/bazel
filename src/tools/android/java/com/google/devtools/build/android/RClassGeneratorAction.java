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
package com.google.devtools.build.android;

import com.android.builder.core.VariantConfiguration;
import com.android.utils.StdLogger;
import com.google.common.base.Preconditions;
import com.google.common.base.Stopwatch;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.Multimap;
import com.google.devtools.build.android.Converters.DependencySymbolFileProviderConverter;
import com.google.devtools.build.android.Converters.PathConverter;
import com.google.devtools.build.android.resources.ResourceSymbols;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.ShellQuotedParamsFilePreProcessor;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

/**
 * Provides an entry point for the compiling resource classes using a custom compiler (simply parse
 * R.txt and make a jar, which is simpler than parsing R.java and running errorprone, etc.).
 *
 * <p>For now, we assume this is only worthwhile for android_binary and not libraries.
 *
 * <pre>
 * Example Usage:
 *   java/com/google/build/android/RClassGeneratorAction\
 *      --primaryRTxt path/to/R.txt\
 *      --primaryManifest path/to/AndroidManifest.xml\
 *      --library p/t/1/AndroidManifest.txt,p/t/1/R.txt\
 *      --library p/t/2/AndroidManifest.txt,p/t/2/R.txt\
 *      --classJarOutput path/to/write/archive_resources.jar
 * </pre>
 */
public class RClassGeneratorAction {

  private static final StdLogger STD_LOGGER = new StdLogger(StdLogger.Level.WARNING);

  private static final Logger logger = Logger.getLogger(RClassGeneratorAction.class.getName());

  /** Flag specifications for this action. */
  public static final class Options extends OptionsBase {

    @Option(
      name = "primaryRTxt",
      defaultValue = "null",
      converter = PathConverter.class,
      category = "input",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "The path to the binary's R.txt file"
    )
    public Path primaryRTxt;

    @Option(
      name = "primaryManifest",
      defaultValue = "null",
      converter = PathConverter.class,
      category = "input",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "The path to the binary's AndroidManifest.xml file. This helps provide the package."
    )
    public Path primaryManifest;

    @Option(
      name = "packageForR",
      defaultValue = "null",
      category = "config",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Custom java package to generate the R class files."
    )
    public String packageForR;

    @Option(
        name = "library",
        allowMultiple = true,
        defaultValue = "null",
        converter = DependencySymbolFileProviderConverter.class,
        category = "input",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help =
            "R.txt and manifests for the libraries in this binary's deps. We will write "
                + "class files for the libraries as well. Expected format: lib1/R.txt[:lib2/R.txt]")
    public List<DependencySymbolFileProvider> libraries;

    @Option(
      name = "classJarOutput",
      defaultValue = "null",
      converter = PathConverter.class,
      category = "output",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Path for the generated jar of R.class files."
    )
    public Path classJarOutput;

    @Option(
        name = "finalFields",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help = "A boolean to control whether fields get declared as final, defaults to true.")
    public boolean finalFields;

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
            .optionsClasses(Options.class, ResourceProcessorCommonOptions.class)
            .argsPreProcessor(new ShellQuotedParamsFilePreProcessor(FileSystems.getDefault()))
            .build();
    optionsParser.parseAndExitUponError(args);
    Options options = optionsParser.getOptions(Options.class);
    Preconditions.checkNotNull(options.classJarOutput);
    final AndroidResourceProcessor resourceProcessor = new AndroidResourceProcessor(STD_LOGGER);
    try (ScopedTemporaryDirectory scopedTmp =
        new ScopedTemporaryDirectory("android_res_compile_tmp")) {
      Path tmp = scopedTmp.getPath();
      Path classOutPath = tmp.resolve("compiled_classes");

      logger.fine(String.format("Setup finished at %sms", timer.elapsed(TimeUnit.MILLISECONDS)));
      // Note that we need to write the R class for the main binary (so proceed even if there
      // are no libraries).
      if (options.primaryRTxt != null) {
        String appPackageName = options.packageForR;
        if (appPackageName == null) {
          appPackageName =
              VariantConfiguration.getManifestPackage(options.primaryManifest.toFile());
        }
        Multimap<String, ResourceSymbols> libSymbolMap = ArrayListMultimap.create();
        ResourceSymbols fullSymbolValues =
            resourceProcessor.loadResourceSymbolTable(
                options.libraries, appPackageName, options.primaryRTxt, libSymbolMap);
        logger.fine(
            String.format("Load symbols finished at %sms", timer.elapsed(TimeUnit.MILLISECONDS)));
        // For now, assuming not used for libraries and setting final access for fields.
        fullSymbolValues.writeClassesTo(
            libSymbolMap, appPackageName, classOutPath, options.finalFields);
        logger.fine(
            String.format("Finished R.class at %sms", timer.elapsed(TimeUnit.MILLISECONDS)));
      } else if (!options.libraries.isEmpty()) {
        Multimap<String, ResourceSymbols> libSymbolMap = ArrayListMultimap.create();
        ResourceSymbols fullSymbolValues =
            resourceProcessor.loadResourceSymbolTable(options.libraries, null, null, libSymbolMap);
        logger.fine(
            String.format("Load symbols finished at %sms", timer.elapsed(TimeUnit.MILLISECONDS)));
        // For now, assuming not used for libraries and setting final access for fields.
        fullSymbolValues.writeClassesTo(libSymbolMap, null, classOutPath, true);
        logger.fine(
            String.format("Finished R.class at %sms", timer.elapsed(TimeUnit.MILLISECONDS)));
      } else {
        Files.createDirectories(classOutPath);
      }
      // We write .class files to temp, then jar them up after (we create a dummy jar, even if
      // there are no class files).
      AndroidResourceOutputs.createClassJar(
          classOutPath, options.classJarOutput, options.targetLabel, options.injectingRuleKind);
      logger.fine(
          String.format("createClassJar finished at %sms", timer.elapsed(TimeUnit.MILLISECONDS)));
    } finally {
      resourceProcessor.shutdown();
    }
    logger.fine(String.format("Compile action done in %sms", timer.elapsed(TimeUnit.MILLISECONDS)));
  }
}
