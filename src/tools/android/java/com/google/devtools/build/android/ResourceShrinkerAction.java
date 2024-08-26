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

import com.android.build.gradle.tasks.ResourceUsageAnalyzer;
import com.android.builder.core.VariantTypeImpl;
import com.android.builder.internal.aapt.AaptOptions;
import com.android.ide.common.xml.AndroidManifestParser;
import com.android.ide.common.xml.ManifestData;
import com.android.io.StreamException;
import com.android.utils.StdLogger;
import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import com.google.common.base.Stopwatch;
import com.google.common.collect.ImmutableList;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.android.AndroidResourceProcessor.AaptConfigOptions;
import com.google.devtools.build.android.Converters.CompatExistingPathConverter;
import com.google.devtools.build.android.Converters.CompatPathConverter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import javax.xml.parsers.ParserConfigurationException;
import org.xml.sax.SAXException;

/**
 * An action to perform resource shrinking using the Gradle resource shrinker.
 *
 * <pre>
 * Example Usage:
 *   java/com/google/build/android/ResourceShrinkerAction
 *       --aapt path to sdk/aapt
 *       --androidJar path to sdk/androidJar
 *       --shrunkJar path to proguard dead code removal jar
 *       --resources path to processed resources zip
 *       --rTxt path to processed resources R.txt
 *       --primaryManifest path to processed resources AndroidManifest.xml
 *       --dependencyManifest path to dependency library manifest (repeated flag)
 *       --shrunkResourceApk path to write shrunk ap_
 *       --shrunkResources path to write shrunk resources zip
 * </pre>
 */
@Deprecated
public class ResourceShrinkerAction {
  private static final StdLogger stdLogger = new StdLogger(StdLogger.Level.WARNING);
  private static final Logger logger = Logger.getLogger(ResourceShrinkerAction.class.getName());

  /** Flag specifications for this action. */
  @Parameters(separators = "= ")
  public static class Options {
    @Parameter(
        names = "--shrunkJar",
        converter = CompatExistingPathConverter.class,
        description = "Path to the shrunk jar from a Proguard run with shrinking enabled.")
    public Path shrunkJar;

    @Parameter(
        names = "--proguardMapping",
        converter = CompatPathConverter.class,
        description = "Path to the Proguard obfuscation mapping of shrunkJar.")
    public Path proguardMapping;

    @Parameter(
        names = "--resources",
        converter = CompatExistingPathConverter.class,
        description = "Path to the resources zip to be shrunk.")
    public Path resourcesZip;

    @Parameter(
        names = "--rTxt",
        converter = CompatExistingPathConverter.class,
        description = "Path to the R.txt of the complete resource tree.")
    public Path rTxt;

    @Parameter(
        names = "--primaryManifest",
        converter = CompatExistingPathConverter.class,
        description = "Path to the primary manifest for the resources to be shrunk.")
    public Path primaryManifest;

    @Parameter(
        names = "--dependencyManifest",
        converter = CompatPathConverter.class,
        description = "Paths to the manifests of the dependencies. Specify one path per flag.")
    public List<Path> dependencyManifests = new ArrayList<>();

    @Parameter(
        names = "--resourcePackages",
        description = "A list of packages that resources have been generated for.")
    public List<String> resourcePackages = new ArrayList<>();

    @Parameter(
        names = "--shrunkResourceApk",
        converter = CompatPathConverter.class,
        description = "Path to where the shrunk resource.ap_ should be written.")
    public Path shrunkApk;

    @Parameter(
        names = "--shrunkResources",
        converter = CompatPathConverter.class,
        description = "Path to where the shrunk resource.ap_ should be written.")
    public Path shrunkResources;

    @Parameter(
        names = "--rTxtOutput",
        converter = CompatPathConverter.class,
        description = "Path to where the R.txt should be written.")
    public Path rTxtOutput;

    @Parameter(
        names = "--log",
        converter = CompatPathConverter.class,
        description = "Path to where the shrinker log should be written.")
    public Path log;

    @Parameter(
        names = "--resourcesConfigOutput",
        converter = CompatPathConverter.class,
        description =
            "Path to where the list of resources configuration directives should be written.")
    public Path resourcesConfigOutput;

    @Parameter(
        names = "--packageType",
        description =
            "Variant configuration type for packaging the resources."
                + " Acceptable values BASE_APK, LIBRARY, ANDROID_TEST, UNIT_TEST")
    public VariantTypeImpl packageType = VariantTypeImpl.BASE_APK;
  }

  private static String getManifestPackage(Path manifest)
      throws SAXException, IOException, StreamException, ParserConfigurationException {
    ManifestData manifestData = AndroidManifestParser.parse(Files.newInputStream(manifest));
    return manifestData.getPackage();
  }

  private static Set<String> getManifestPackages(Path primaryManifest, List<Path> otherManifests)
      throws SAXException, IOException, StreamException, ParserConfigurationException {
    Set<String> manifestPackages = new LinkedHashSet<>();
    manifestPackages.add(getManifestPackage(primaryManifest));
    for (Path manifest : otherManifests) {
      manifestPackages.add(getManifestPackage(manifest));
    }
    return manifestPackages;
  }

  public static void main(String[] args) throws Exception {
    final Stopwatch timer = Stopwatch.createStarted();
    // Parse arguments.
    Options options = new Options();
    AaptConfigOptions aaptConfigOptions = new AaptConfigOptions();
    ResourceProcessorCommonOptions resourceProcessorCommonOptions =
        new ResourceProcessorCommonOptions();
    Object[] allOptions = new Object[] {options, resourceProcessorCommonOptions, aaptConfigOptions};
    JCommander jc = new JCommander(allOptions);
    String[] preprocessedArgs = AndroidOptionsUtils.runArgFilePreprocessor(jc, args);
    String[] normalizedArgs =
        AndroidOptionsUtils.normalizeBooleanOptions(allOptions, preprocessedArgs);
    jc.parse(normalizedArgs);

    AndroidResourceProcessor resourceProcessor = new AndroidResourceProcessor(stdLogger);
    // Setup temporary working directories.
    try (ScopedTemporaryDirectory scopedTmp =
        new ScopedTemporaryDirectory("resource_shrinker_tmp")) {
      Path working = scopedTmp.getPath();
      final Path resourceFiles = working.resolve("resource_files");

      final Path shrunkResources = working.resolve("shrunk_resources");

      // Gather package list from manifests.
      Set<String> resourcePackages =
          getManifestPackages(options.primaryManifest, options.dependencyManifests);
      resourcePackages.addAll(options.resourcePackages);

      // Expand resource files zip into working directory.
      try (ZipInputStream zin =
          new ZipInputStream(new FileInputStream(options.resourcesZip.toFile()))) {
        ZipEntry entry;
        while ((entry = zin.getNextEntry()) != null) {
          if (!entry.isDirectory()) {
            Path output = resourceFiles.resolve(entry.getName());
            Files.createDirectories(output.getParent());
            try (FileOutputStream fos = new FileOutputStream(output.toFile())) {
              ByteStreams.copy(zin, fos);
            }
          }
        }
      }

      // Shrink resources.
      ResourceUsageAnalyzer resourceShrinker =
          new ResourceUsageAnalyzer(
              resourcePackages,
              options.rTxt,
              options.shrunkJar,
              options.primaryManifest,
              options.proguardMapping,
              resourceFiles.resolve("res"),
              options.log);

      resourceShrinker.shrink(shrunkResources);
      logger.fine(
          String.format(
              "Shrinking resources finished at %sms", timer.elapsed(TimeUnit.MILLISECONDS)));

      Path generatedSources = null;
      if (options.rTxtOutput != null) {
        generatedSources = working.resolve("generated_resources");
      }

      // Build ap_ with shrunk resources.
      resourceProcessor.processResources(
          working,
          aaptConfigOptions.aapt,
          aaptConfigOptions.androidJar,
          aaptConfigOptions.buildToolsVersion,
          VariantTypeImpl.BASE_APK,
          aaptConfigOptions.debug,
          /* customPackageForR= */ null,
          new AaptOptions(
              /* noCompress= */ aaptConfigOptions.uncompressedExtensions,
              /* additionalParameters= */ ImmutableList.of()),
          aaptConfigOptions.resourceConfigs,
          aaptConfigOptions.useDataBindingAndroidX,
          new MergedAndroidData(
              shrunkResources, resourceFiles.resolve("assets"), options.primaryManifest),
          /* dependencyData= */ ImmutableList.<DependencyAndroidData>of(),
          generatedSources,
          options.shrunkApk,
          /* proguardOut= */ null,
          /* mainDexProguardOut= */ null,
          /* publicResourcesOut= */ null,
          /* dataBindingInfoOut= */ null);
      if (options.shrunkResources != null) {
        ResourcesZip.from(shrunkResources, resourceFiles.resolve("assets"))
            .writeTo(options.shrunkResources, /* compress= */ false);
      }
      if (options.rTxtOutput != null) {
        AndroidResourceOutputs.copyRToOutput(
            generatedSources, options.rTxtOutput, options.packageType == VariantTypeImpl.LIBRARY);
      }
      logger.fine(
          String.format(
              "Packing resources finished at %sms", timer.elapsed(TimeUnit.MILLISECONDS)));
    } catch (Exception e) {
      logger.log(Level.SEVERE, "Error shrinking resources", e);
      throw e;
    } finally {
      resourceProcessor.shutdown();
    }
  }
}
