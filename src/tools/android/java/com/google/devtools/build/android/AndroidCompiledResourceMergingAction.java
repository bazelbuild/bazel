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

import com.android.builder.core.DefaultManifestParser;
import com.android.utils.StdLogger;
import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import com.google.common.base.Preconditions;
import com.google.common.base.Stopwatch;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.android.AndroidDataMerger.MergeConflictException;
import com.google.devtools.build.android.AndroidResourceMerger.MergingException;
import com.google.devtools.build.android.AndroidResourceProcessor.AaptConfigOptions;
import com.google.devtools.build.android.Converters.AmpersandSplitter;
import com.google.devtools.build.android.Converters.CompatExistingPathConverter;
import com.google.devtools.build.android.Converters.CompatPathConverter;
import com.google.devtools.build.android.Converters.CompatSerializedAndroidDataConverter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Provides an entry point for the compiled resource merging action.
 *
 * <p>This action merges compiled intermediate resource files from aapt2 and reports merge
 * conflicts. It also provides a merged manifest file to {@link ValidateAndLinkResourcesAction} and
 * builds the resource class jar for the lib jar
 */
public class AndroidCompiledResourceMergingAction {

  private static final StdLogger stdLogger = new StdLogger(StdLogger.Level.WARNING);

  private static final Logger logger =
      Logger.getLogger(AndroidCompiledResourceMergingAction.class.getName());

  /** Flag specifications for this action. */
  @Parameters(separators = "= ")
  public static class Options {
    @Parameter(
        names = "--primaryData",
        converter = CompatSerializedAndroidDataConverter.class,
        description =
            "The directory containing the primary resource directory. The contents will override"
                + " the contents of any other resource directories during merging."
                + " The expected format is "
                + SerializedAndroidData.EXPECTED_FORMAT)
    public SerializedAndroidData primaryData;

    @Parameter(
        names = "--primaryManifest",
        converter = CompatExistingPathConverter.class,
        description = "Path to primary resource's manifest file.")
    public Path primaryManifest;

    @Parameter(
        names = "--data",
        converter = CompatSerializedAndroidDataConverter.class,
        splitter = AmpersandSplitter.class,
        description =
            "Transitive Data dependencies. These values will be used if not defined in the "
                + "primary resources. The expected format is "
                + SerializedAndroidData.EXPECTED_FORMAT
                + "[&...]")
    public List<SerializedAndroidData> transitiveData = ImmutableList.of();

    @Parameter(
        names = "--directData",
        converter = CompatSerializedAndroidDataConverter.class,
        splitter = AmpersandSplitter.class,
        description =
            "Direct Data dependencies. These values will be used if not defined in the "
                + "primary resources. The expected format is "
                + SerializedAndroidData.EXPECTED_FORMAT
                + "[&...]")
    public List<SerializedAndroidData> directData = ImmutableList.of();

    @Parameter(
        names = "--classJarOutput",
        converter = CompatPathConverter.class,
        description = "Path for the generated java class jar.")
    public Path classJarOutput;

    @Parameter(
        names = "--manifestOutput",
        converter = CompatPathConverter.class,
        description = "Path for the output processed AndroidManifest.xml.")
    public Path manifestOutput;

    @Parameter(
        names = "--packageForR",
        description = "Custom java package to generate the R symbols files.")
    public String packageForR;

    @Parameter(
        names = "--throwOnResourceConflict",
        arity = 1,
        description =
            "If passed, resource merge conflicts will be treated as errors instead of warnings")
    public boolean throwOnResourceConflict;

    @Parameter(
        names = "--targetLabel",
        description = "A label to add to the output jar's manifest as 'Target-Label'")
    public String targetLabel;

    @Parameter(
        names = "--injectingRuleKind",
        description = "A string to add to the output jar's manifest as 'Injecting-Rule-Kind'")
    public String injectingRuleKind;

    @Parameter(
        names = "--annotate_r_fields_from_transitive_deps",
        arity = 1,
        description =
            "If enabled, annotates R with 'targetLabel' and transitive fields with their"
                + " respective labels.")
    public boolean annotateTransitiveFields;

    @Parameter(
        names = "--rTxtOut",
        converter = CompatPathConverter.class,
        description =
            "Path to where an R.txt file declaring potentially-used resources should be written.")
    public Path rTxtOut;
  }

  public static void main(String[] args) throws Exception {
    final Stopwatch timer = Stopwatch.createStarted();
    Options options = new Options();
    AaptConfigOptions aaptConfigOptions = new AaptConfigOptions();
    ResourceProcessorCommonOptions resourceProcessorCommonOptions =
        new ResourceProcessorCommonOptions();
    Object[] allOptions = new Object[] {options, aaptConfigOptions, resourceProcessorCommonOptions};
    JCommander jc = new JCommander(allOptions);
    String[] preprocessedArgs = AndroidOptionsUtils.runArgFilePreprocessor(jc, args);
    String[] normalizedArgs =
        AndroidOptionsUtils.normalizeBooleanOptions(allOptions, preprocessedArgs);
    jc.parse(normalizedArgs);

    Preconditions.checkNotNull(options.primaryData);
    Preconditions.checkNotNull(options.primaryManifest);
    Preconditions.checkNotNull(options.manifestOutput);
    Preconditions.checkNotNull(options.classJarOutput);

    try (ScopedTemporaryDirectory scopedTmp =
            new ScopedTemporaryDirectory("android_resource_merge_tmp");
        ExecutorServiceCloser executorService = ExecutorServiceCloser.createWithFixedPoolOf(15)) {
      Path tmp = scopedTmp.getPath();
      Path generatedSources = tmp.resolve("generated_resources");
      Path processedManifest = tmp.resolve("manifest-processed/AndroidManifest.xml");

      logger.fine(String.format("Setup finished at %sms", timer.elapsed(TimeUnit.MILLISECONDS)));

      String packageForR = options.packageForR;
      if (packageForR == null) {
        packageForR =
            Strings.nullToEmpty(
                new DefaultManifestParser(
                        options.primaryManifest.toFile(),
                        /* canParseManifest= */ () -> true,
                        /* isManifestFileRequired= */ true,
                        /* issueReporter= */ null)
                    .getPackage());
      }
      AndroidResourceClassWriter resourceClassWriter =
          AndroidResourceClassWriter.createWith(
              options.targetLabel, aaptConfigOptions.androidJar, generatedSources, packageForR);
      resourceClassWriter.setIncludeClassFile(true);
      resourceClassWriter.setIncludeJavaFile(false);
      resourceClassWriter.setAnnotateTransitiveFields(options.annotateTransitiveFields);

      PlaceholderRTxtWriter rTxtWriter =
          options.rTxtOut != null ? PlaceholderRTxtWriter.create(options.rTxtOut) : null;

      AndroidResourceMerger.mergeCompiledData(
          options.primaryData,
          options.primaryManifest,
          options.directData,
          options.transitiveData,
          resourceClassWriter,
          rTxtWriter,
          options.throwOnResourceConflict,
          executorService);
      logger.fine(String.format("Merging finished at %sms", timer.elapsed(TimeUnit.MILLISECONDS)));

      AndroidResourceOutputs.createClassJar(
          generatedSources, options.classJarOutput, options.targetLabel, options.injectingRuleKind);
      logger.fine(
          String.format("Create classJar finished at %sms", timer.elapsed(TimeUnit.MILLISECONDS)));

      // Until enough users with manifest placeholders migrate to the new manifest merger,
      // we need to replace ${applicationId} and ${packageName} with options.packageForR to make
      // the manifests compatible with the old manifest merger.
      processedManifest =
          AndroidManifestProcessor.with(stdLogger)
              .processLibraryManifest(
                  options.packageForR,
                  options.primaryManifest,
                  processedManifest,
                  resourceProcessorCommonOptions.logWarnings);

      Files.createDirectories(options.manifestOutput.getParent());
      Files.copy(processedManifest, options.manifestOutput);
    } catch (MergeConflictException e) {
      logger.log(Level.SEVERE, e.getMessage());
      throw e;
    } catch (MergingException e) {
      logger.log(Level.SEVERE, "Error during merging resources", e);
      throw e;
    } catch (AndroidManifestProcessor.ManifestProcessingException e) {
      throw e;
    } catch (Exception e) {
      logger.log(Level.SEVERE, "Unexpected", e);
      throw e;
    }
    logger.fine(String.format("Resources merged in %sms", timer.elapsed(TimeUnit.MILLISECONDS)));
  }
}
