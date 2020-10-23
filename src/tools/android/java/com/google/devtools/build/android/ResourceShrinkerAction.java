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
import com.android.builder.core.VariantType;
import com.android.ide.common.xml.AndroidManifestParser;
import com.android.ide.common.xml.ManifestData;
import com.android.io.StreamException;
import com.android.utils.StdLogger;
import com.google.common.base.Stopwatch;
import com.google.common.collect.ImmutableList;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.android.AndroidResourceProcessor.AaptConfigOptions;
import com.google.devtools.build.android.AndroidResourceProcessor.FlagAaptOptions;
import com.google.devtools.build.android.Converters.ExistingPathConverter;
import com.google.devtools.build.android.Converters.PathConverter;
import com.google.devtools.build.android.Converters.VariantTypeConverter;
import com.google.devtools.common.options.Converters.CommaSeparatedOptionListConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.ShellQuotedParamsFilePreProcessor;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
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
  public static final class Options extends OptionsBase {
    @Option(
      name = "shrunkJar",
      defaultValue = "null",
      category = "input",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      converter = ExistingPathConverter.class,
      help = "Path to the shrunk jar from a Proguard run with shrinking enabled."
    )
    public Path shrunkJar;

    @Option(
      name = "proguardMapping",
      defaultValue = "null",
      category = "input",
      converter = PathConverter.class,
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Path to the Proguard obfuscation mapping of shrunkJar."
    )
    public Path proguardMapping;

    @Option(
      name = "resources",
      defaultValue = "null",
      category = "input",
      converter = ExistingPathConverter.class,
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Path to the resources zip to be shrunk."
    )
    public Path resourcesZip;

    @Option(
      name = "rTxt",
      defaultValue = "null",
      category = "input",
      converter = ExistingPathConverter.class,
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Path to the R.txt of the complete resource tree."
    )
    public Path rTxt;

    @Option(
      name = "primaryManifest",
      defaultValue = "null",
      category = "input",
      converter = ExistingPathConverter.class,
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Path to the primary manifest for the resources to be shrunk."
    )
    public Path primaryManifest;

    @Option(
        name = "dependencyManifest",
        allowMultiple = true,
        defaultValue = "null",
        category = "input",
        converter = PathConverter.class,
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help = "Paths to the manifests of the dependencies. Specify one path per flag.")
    public List<Path> dependencyManifests;

    @Option(
      name = "resourcePackages",
      defaultValue = "",
      category = "input",
      converter = CommaSeparatedOptionListConverter.class,
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "A list of packages that resources have been generated for."
    )
    public List<String> resourcePackages;

    @Option(
      name = "shrunkResourceApk",
      defaultValue = "null",
      category = "output",
      converter = PathConverter.class,
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Path to where the shrunk resource.ap_ should be written."
    )
    public Path shrunkApk;

    @Option(
      name = "shrunkResources",
      defaultValue = "null",
      category = "output",
      converter = PathConverter.class,
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Path to where the shrunk resource.ap_ should be written."
    )
    public Path shrunkResources;

    @Option(
      name = "rTxtOutput",
      defaultValue = "null",
      converter = PathConverter.class,
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      category = "output",
      help = "Path to where the R.txt should be written."
    )
    public Path rTxtOutput;

    @Option(
      name = "log",
      defaultValue = "null",
      category = "output",
      converter = PathConverter.class,
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Path to where the shrinker log should be written."
    )
    public Path log;

    @Option(
        name = "resourcesConfigOutput",
        defaultValue = "null",
        converter = PathConverter.class,
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help = "Path to where the list of resources configuration directives should be written.")
    public Path resourcesConfigOutput;

    @Option(
        name = "packageType",
        defaultValue = "DEFAULT",
        converter = VariantTypeConverter.class,
        category = "config",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help =
            "Variant configuration type for packaging the resources."
                + " Acceptable values DEFAULT, LIBRARY, ANDROID_TEST, UNIT_TEST")
    public VariantType packageType;
  }

  private static AaptConfigOptions aaptConfigOptions;
  private static Options options;

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
    OptionsParser optionsParser =
        OptionsParser.builder()
            .optionsClasses(Options.class, AaptConfigOptions.class)
            .argsPreProcessor(new ShellQuotedParamsFilePreProcessor(FileSystems.getDefault()))
            .build();
    optionsParser.parseAndExitUponError(args);
    aaptConfigOptions = optionsParser.getOptions(AaptConfigOptions.class);
    options = optionsParser.getOptions(Options.class);

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
          VariantType.DEFAULT,
          aaptConfigOptions.debug,
          null /* packageForR */,
          new FlagAaptOptions(aaptConfigOptions),
          aaptConfigOptions.resourceConfigs,
          new MergedAndroidData(
              shrunkResources, resourceFiles.resolve("assets"), options.primaryManifest),
          ImmutableList.<DependencyAndroidData>of() /* libraries */,
          generatedSources,
          options.shrunkApk,
          null /* proguardOutput */,
          null /* mainDexProguardOutput */,
          /* publicResourcesOut= */ null,
          /* dataBindingInfoOut= */ null);
      if (options.shrunkResources != null) {
        ResourcesZip.from(shrunkResources, resourceFiles.resolve("assets"))
            .writeTo(options.shrunkResources, /* compress= */ false);
      }
      if (options.rTxtOutput != null) {
        AndroidResourceOutputs.copyRToOutput(
            generatedSources, options.rTxtOutput, options.packageType == VariantType.LIBRARY);
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
