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

import com.google.common.base.Stopwatch;
import com.google.common.collect.ImmutableList;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.android.AndroidResourceProcessor.AaptConfigOptions;
import com.google.devtools.build.android.AndroidResourceProcessor.FlagAaptOptions;
import com.google.devtools.build.android.Converters.ExistingPathConverter;
import com.google.devtools.build.android.Converters.PathConverter;
import com.google.devtools.build.android.Converters.PathListConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;

import com.android.builder.core.VariantConfiguration;
import com.android.ide.common.xml.AndroidManifestParser;
import com.android.ide.common.xml.ManifestData;
import com.android.io.StreamException;
import com.android.utils.StdLogger;

import org.xml.sax.SAXException;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

import javax.xml.parsers.ParserConfigurationException;

/**
 * An action to perform resource shrinking using the Gradle resource shrinker.
 *
 * <pre>
 * Example Usage:
 *   java/com/google/build/android/ResourceShrinkerAction
 *       --aapt path to sdk/aapt
 *       --annotationJar path to sdk/annotationJar
 *       --androidJar path to sdk/androidJar
 *       --shrunkJar path to proguard dead code removal jar
 *       --resources path to processed resources zip
 *       --rTxt path to processed resources R.txt
 *       --primaryManifest path to processed resources AndroidManifest.xml
 *       --dependencyManifests paths to dependency library manifests
 *       --shrunkResourceApk path to write shrunk ap_
 *       --shrunkResources path to write shrunk resources zip
 * </pre>
 */
public class ResourceShrinkerAction {
  private static final StdLogger stdLogger = new StdLogger(StdLogger.Level.WARNING);
  private static final Logger logger = Logger.getLogger(ResourceShrinkerAction.class.getName());

  /** Flag specifications for this action. */
  public static final class Options extends OptionsBase {
    @Option(name = "shrunkJar",
        defaultValue = "null",
        category = "input",
        converter = ExistingPathConverter.class,
        help = "Path to the shrunk jar from a Proguard run with shrinking enabled.")
    public Path shrunkJar;

    @Option(name = "resources",
        defaultValue = "null",
        category = "input",
        converter = ExistingPathConverter.class,
        help = "Path to the resources zip to be shrunk.")
    public Path resourcesZip;

    @Option(name = "rTxt",
        defaultValue = "null",
        category = "input",
        converter = ExistingPathConverter.class,
        help = "Path to the R.txt of the complete resource tree.")
    public Path rTxt;

    @Option(name = "primaryManifest",
        defaultValue = "null",
        category = "input",
        converter = ExistingPathConverter.class,
        help = "Path to the primary manifest for the resources to be shrunk.")
    public Path primaryManifest;

    @Option(name = "dependencyManifests",
        defaultValue = "",
        category = "input",
        converter = PathListConverter.class,
        help = "A list of paths to the manifests of the dependencies.")
    public List<Path> dependencyManifests;

    @Option(name = "shrunkResourceApk",
        defaultValue = "null",
        category = "output",
        converter = PathConverter.class,
        help = "Path to where the shrunk resource.ap_ should be written.")
    public Path shrunkApk;

    @Option(name = "shrunkResources",
        defaultValue = "null",
        category = "output",
        converter = PathConverter.class,
        help = "Path to where the shrunk resource.ap_ should be written.")
    public Path shrunkResources;
  }

  private static AaptConfigOptions aaptConfigOptions;
  private static Options options;

  private static String getManifestPackage(Path manifest)
      throws SAXException, IOException, StreamException, ParserConfigurationException {
    ManifestData manifestData = AndroidManifestParser.parse(Files.newInputStream(manifest));
    return manifestData.getPackage();
  }

  private static List<String> getManifestPackages(Path primaryManifest, List<Path> otherManifests)
          throws SAXException, IOException, StreamException, ParserConfigurationException {
    ImmutableList.Builder<String> manifestPackages = ImmutableList.builder();
    manifestPackages.add(getManifestPackage(primaryManifest));
    for (Path manifest : otherManifests) {
      manifestPackages.add(getManifestPackage(manifest));
    }
    return manifestPackages.build();
  }

  public static void main(String[] args) throws Exception {
    final Stopwatch timer = Stopwatch.createStarted();
    // Parse arguments.
    OptionsParser optionsParser = OptionsParser.newOptionsParser(
        Options.class, AaptConfigOptions.class);
    optionsParser.parseAndExitUponError(args);
    aaptConfigOptions = optionsParser.getOptions(AaptConfigOptions.class);
    options = optionsParser.getOptions(Options.class);

    AndroidResourceProcessor resourceProcessor = new AndroidResourceProcessor(stdLogger);
    try {
      // Setup temporary working directories.
      Path working = Files.createTempDirectory("resource_shrinker_tmp");
      working.toFile().deleteOnExit();

      final Path resourceFiles = working.resolve("resource_files");

      final Path shrunkResources = working.resolve("shrunk_resources");

      // Gather package list from manifests.
      List<String> resourcePackages = getManifestPackages(
          options.primaryManifest, options.dependencyManifests);

      // Expand resource files zip into working directory.
      try (ZipInputStream zin = new ZipInputStream(
          new FileInputStream(options.resourcesZip.toFile()))) {
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
      ResourceShrinker resourceShrinker = new ResourceShrinker(
          resourcePackages,
          options.rTxt,
          options.shrunkJar,
          options.primaryManifest,
          resourceFiles.resolve("res"));

      resourceShrinker.shrink(shrunkResources);
      logger.fine(String.format("Shrinking resources finished at %sms",
          timer.elapsed(TimeUnit.MILLISECONDS)));

      // Build ap_ with shrunk resources.
      resourceProcessor.processResources(
          aaptConfigOptions.aapt,
          aaptConfigOptions.androidJar,
          aaptConfigOptions.buildToolsVersion,
          VariantConfiguration.Type.DEFAULT,
          aaptConfigOptions.debug,
          null /* packageForR */,
          new FlagAaptOptions(aaptConfigOptions),
          aaptConfigOptions.resourceConfigs,
          new MergedAndroidData(
              shrunkResources, resourceFiles.resolve("assets"), options.primaryManifest),
          ImmutableList.<DependencyAndroidData>of() /* libraries */,
          null /* sourceOutputDir */,
          options.shrunkApk,
          null /* proguardOutput */,
          null /* publicResourcesOut */);
      if (options.shrunkResources != null) {
        resourceProcessor.createResourcesZip(shrunkResources, resourceFiles.resolve("assets"),
            options.shrunkResources);
      }
      logger.fine(String.format("Packing resources finished at %sms",
          timer.elapsed(TimeUnit.MILLISECONDS)));
    } catch (Exception e) {
      logger.log(Level.SEVERE, "Error shrinking resources", e);
      throw e;
    } finally {
      resourceProcessor.shutdown();
    }
  }
}
