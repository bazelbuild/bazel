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

import com.android.builder.core.VariantConfiguration;
import com.android.utils.StdLogger;
import com.google.common.base.Preconditions;
import com.google.common.base.Stopwatch;
import com.google.common.base.Strings;
import com.google.devtools.build.android.AndroidDataMerger.MergeConflictException;
import com.google.devtools.build.android.AndroidResourceMerger.MergingException;
import com.google.devtools.build.android.AndroidResourceMergingAction.Options;
import com.google.devtools.build.android.AndroidResourceProcessor.AaptConfigOptions;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.ShellQuotedParamsFilePreProcessor;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
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
                VariantConfiguration.getManifestPackage(options.primaryManifest.toFile()));
      }
      AndroidResourceClassWriter resourceClassWriter =
          AndroidResourceClassWriter.createWith(
              aaptConfigOptions.androidJar, generatedSources, packageForR);
      resourceClassWriter.setIncludeClassFile(true);
      resourceClassWriter.setIncludeJavaFile(false);

      AndroidResourceMerger.mergeCompiledData(
          options.primaryData,
          options.primaryManifest,
          options.directData,
          options.transitiveData,
          resourceClassWriter,
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
                  options.packageForR, options.primaryManifest, processedManifest);

      Files.createDirectories(options.manifestOutput.getParent());
      Files.copy(processedManifest, options.manifestOutput);
    } catch (MergeConflictException e) {
      logger.log(Level.SEVERE, e.getMessage());
      System.exit(1);
    } catch (MergingException e) {
      logger.log(Level.SEVERE, "Error during merging resources", e);
      throw e;
    } catch (AndroidManifestProcessor.ManifestProcessingException e) {
      System.exit(1);
    } catch (Exception e) {
      logger.log(Level.SEVERE, "Unexpected", e);
      throw e;
    }
    logger.fine(String.format("Resources merged in %sms", timer.elapsed(TimeUnit.MILLISECONDS)));
  }
}
