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

import com.android.builder.core.DefaultManifestParser;
import com.android.utils.StdLogger;
import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import com.google.common.base.Preconditions;
import com.google.common.base.Stopwatch;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Multimap;
import com.google.devtools.build.android.Converters.CompatDependencySymbolFileProviderConverter;
import com.google.devtools.build.android.Converters.CompatPathConverter;
import com.google.devtools.build.android.Converters.NoOpSplitter;
import com.google.devtools.build.android.resources.RPackageId;
import com.google.devtools.build.android.resources.ResourceSymbols;
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
  @Parameters(separators = "= ")
  public static final class Options {

    @Parameter(
        names = "--primaryRTxt",
        converter = CompatPathConverter.class,
        description = "The path to the binary's R.txt file")
    public Path primaryRTxt;

    @Parameter(
        names = "--primaryManifest",
        converter = CompatPathConverter.class,
        description =
            "The path to the binary's AndroidManifest.xml file. This helps provide the package.")
    public Path primaryManifest;

    @Parameter(
        names = "--packageForR",
        description = "Custom java package to generate the R class files.")
    public String packageForR;

    @Parameter(
        names = "--library",
        converter = CompatDependencySymbolFileProviderConverter.class,
        splitter = NoOpSplitter.class,
        description =
            "R.txt and manifests for the libraries in this binary's deps. We will write "
                + "class files for the libraries as well. Expected format: lib1/R.txt[:lib2/R.txt]")
    public List<DependencySymbolFileProvider> libraries = ImmutableList.of();

    @Parameter(
        names = "--classJarOutput",
        converter = CompatPathConverter.class,
        description = "Path for the generated jar of R.class files.")
    public Path classJarOutput;

    @Parameter(
        names = "--finalFields",
        arity = 1,
        description =
            "A boolean to control whether fields get declared as final, defaults to true.")
    public boolean finalFields = true;

    @Parameter(
        names = "--targetLabel",
        description = "A label to add to the output jar's manifest as 'Target-Label'")
    public String targetLabel;

    @Parameter(
        names = "--injectingRuleKind",
        description = "A string to add to the output jar's manifest as 'Injecting-Rule-Kind'")
    public String injectingRuleKind;

    @Parameter(
        names = "--useRPackage",
        arity = 1,
        description =
            "A boolean to control whether fields should be generated with an RPackage"
                + " class, defaults to false. Used for privacy sandbox.")
    public boolean useRPackage;
  }

  public static void main(String[] args) throws Exception {
    final Stopwatch timer = Stopwatch.createStarted();
    Options options = new Options();
    JCommander jc = new JCommander(new Object[] {options, new ResourceProcessorCommonOptions()});
    String[] preprocessedArgs = AndroidOptionsUtils.runArgFilePreprocessor(jc, args);
    String[] normalizedArgs =
        AndroidOptionsUtils.normalizeBooleanOptions(options, preprocessedArgs);
    jc.parse(normalizedArgs);

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
              new DefaultManifestParser(
                      options.primaryManifest.toFile(),
                      /* canParseManifest= */ () -> true,
                      /* isManifestFileRequired= */ true,
                      /* issueReporter= */ null)
                  .getPackage();
        }
        Multimap<String, ResourceSymbols> libSymbolMap = ArrayListMultimap.create();
        ResourceSymbols fullSymbolValues =
            resourceProcessor.loadResourceSymbolTable(
                options.libraries, appPackageName, options.primaryRTxt, libSymbolMap);
        logger.fine(
            String.format("Load symbols finished at %sms", timer.elapsed(TimeUnit.MILLISECONDS)));
        final RPackageId rPackageId =
            options.useRPackage ? RPackageId.createFor(appPackageName) : null;
        // For now, assuming not used for libraries and setting final access for fields.
        fullSymbolValues.writeClassesTo(
            libSymbolMap, appPackageName, classOutPath, options.finalFields, rPackageId);
        logger.fine(
            String.format("Finished R.class at %sms", timer.elapsed(TimeUnit.MILLISECONDS)));
      } else if (!options.libraries.isEmpty()) {
        Multimap<String, ResourceSymbols> libSymbolMap = ArrayListMultimap.create();
        ResourceSymbols fullSymbolValues =
            resourceProcessor.loadResourceSymbolTable(options.libraries, null, null, libSymbolMap);
        logger.fine(
            String.format("Load symbols finished at %sms", timer.elapsed(TimeUnit.MILLISECONDS)));
        // For now, assuming not used for libraries and setting final access for fields.
        fullSymbolValues.writeClassesTo(
            libSymbolMap, null, classOutPath, true, /* rPackageId= */ null);
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
