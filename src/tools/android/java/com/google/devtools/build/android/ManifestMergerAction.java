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

import static java.util.logging.Level.SEVERE;

import com.android.manifmerger.ManifestMerger2.MergeType;
import com.android.utils.StdLogger;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.android.Converters.ExistingPathConverter;
import com.google.devtools.build.android.Converters.ExistingPathStringDictionaryConverter;
import com.google.devtools.build.android.Converters.MergeTypeConverter;
import com.google.devtools.build.android.Converters.PathConverter;
import com.google.devtools.build.android.Converters.StringDictionaryConverter;
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
import java.nio.file.StandardCopyOption;
import java.util.Map;
import java.util.logging.Logger;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.TransformerConfigurationException;
import javax.xml.transform.TransformerException;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.TransformerFactoryConfigurationError;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;
import org.w3c.dom.Document;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

/**
 * An action to perform manifest merging using the Gradle manifest merger.
 *
 * <pre>
 * Example Usage:
 *   java/com/google/build/android/ManifestMergerAction
 *       --manifest path to primary manifest
 *       --mergeeManifests colon separated list of manifests to merge
 *       --mergeType APPLICATION|LIBRARY
 *       --manifestValues key value pairs of manifest overrides
 *       --customPackage package to write for library manifest
 *       --manifestOutput path to write output manifest
 * </pre>
 */
public class ManifestMergerAction {
  /** Flag specifications for this action. */
  public static final class Options extends OptionsBase {
    @Option(
      name = "manifest",
      defaultValue = "null",
      converter = ExistingPathConverter.class,
      category = "input",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "Path of primary manifest. If not passed, a dummy manifest will be generated and used as"
              + " the primary."
    )
    public Path manifest;

    @Option(
      name = "mergeeManifests",
      defaultValue = "",
      converter = ExistingPathStringDictionaryConverter.class,
      category = "input",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "A dictionary of manifests, and originating target, to be merged into manifest."
    )
    public Map<Path, String> mergeeManifests;

    @Option(
      name = "mergeType",
      defaultValue = "APPLICATION",
      converter = MergeTypeConverter.class,
      category = "config",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "The type of merging to perform."
    )
    public MergeType mergeType;

    @Option(
      name = "manifestValues",
      defaultValue = "",
      converter = StringDictionaryConverter.class,
      category = "config",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "A dictionary string of values to be overridden in the manifest. Any instance of "
              + "${name} in the manifest will be replaced with the value corresponding to name in "
              + "this dictionary. applicationId, versionCode, versionName, minSdkVersion, "
              + "targetSdkVersion and maxSdkVersion have a dual behavior of also overriding the "
              + "corresponding attributes of the manifest and uses-sdk tags. packageName will be "
              + "ignored and will be set from either applicationId or the package in manifest. The "
              + "expected format of this string is: key:value[,key:value]*. The keys and values "
              + "may contain colons and commas as long as they are escaped with a backslash."
    )
    public Map<String, String> manifestValues;

    @Option(
      name = "customPackage",
      defaultValue = "null",
      category = "config",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Custom java package to insert in the package attribute of the manifest tag."
    )
    public String customPackage;

    @Option(
      name = "manifestOutput",
      defaultValue = "null",
      converter = PathConverter.class,
      category = "output",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Path for the merged manifest."
    )
    public Path manifestOutput;

    @Option(
      name = "log",
      defaultValue = "null",
      category = "output",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      converter = PathConverter.class,
      help = "Path to where the merger log should be written."
    )
    public Path log;
  }

  private static final String[] PERMISSION_TAGS =
      new String[] {"uses-permission", "uses-permission-sdk-23"};
  private static final StdLogger stdLogger = new StdLogger(StdLogger.Level.WARNING);
  private static final Logger logger = Logger.getLogger(ManifestMergerAction.class.getName());

  private static Options options;

  private static Path removePermissions(Path manifest, Path outputDir)
      throws IOException, ParserConfigurationException, TransformerConfigurationException,
          TransformerException, TransformerFactoryConfigurationError, SAXException {
    DocumentBuilder docBuilder = DocumentBuilderFactory.newInstance().newDocumentBuilder();
    Document doc = docBuilder.parse(manifest.toFile());
    for (String tag : PERMISSION_TAGS) {
      NodeList permissions = doc.getElementsByTagName(tag);
      if (permissions != null) {
        for (int i = permissions.getLength() - 1; i >= 0; i--) {
          Node permission = permissions.item(i);
          permission.getParentNode().removeChild(permission);
        }
      }
    }
    // Write resulting manifest to the output directory, maintaining full path to prevent collisions
    Path output = outputDir.resolve(manifest.toString().replaceFirst("^/", ""));
    Files.createDirectories(output.getParent());
    TransformerFactory.newInstance()
        .newTransformer()
        .transform(new DOMSource(doc), new StreamResult(output.toFile()));
    return output;
  }

  public static void main(String[] args) throws Exception {
    OptionsParser optionsParser = OptionsParser.newOptionsParser(Options.class);
    optionsParser.enableParamsFileSupport(
        new ShellQuotedParamsFilePreProcessor(FileSystems.getDefault()));
    optionsParser.parseAndExitUponError(args);
    options = optionsParser.getOptions(Options.class);

    try {
      Path mergedManifest;
      AndroidManifestProcessor manifestProcessor = AndroidManifestProcessor.with(stdLogger);

      // Remove uses-permission tags from mergees before the merge.
      Path tmp = Files.createTempDirectory("manifest_merge_tmp");
      tmp.toFile().deleteOnExit();
      ImmutableMap.Builder<Path, String> mergeeManifests = ImmutableMap.builder();
      for (Map.Entry<Path, String> mergeeManifest : options.mergeeManifests.entrySet()) {
        mergeeManifests.put(
            removePermissions(mergeeManifest.getKey(), tmp), mergeeManifest.getValue());
      }

      Path manifest = options.manifest;
      if (manifest == null) {
        // No primary manifest was passed. Generate a dummy primary.

        manifest = AndroidManifest.asEmpty().writeDummyManifestForAapt(tmp, options.customPackage);
      }

      mergedManifest =
          manifestProcessor.mergeManifest(
              manifest,
              mergeeManifests.build(),
              options.mergeType,
              options.manifestValues,
              options.customPackage,
              options.manifestOutput,
              options.log);

      if (!mergedManifest.equals(options.manifestOutput)) {
        // manifestProcess.mergeManifest returns the merged manifest, or, if merging was a no-op,
        // the original primary manifest. In the latter case, explicitly copy that primary manifest
        // to the expected location of the output.
        Files.copy(manifest, options.manifestOutput, StandardCopyOption.REPLACE_EXISTING);
      }
    } catch (AndroidManifestProcessor.ManifestProcessingException e) {
      // We special case ManifestProcessingExceptions here to indicate that this is
      // caused by a build error, not an Bazel-internal error.
      logger.log(SEVERE, "Error during merging manifests", e);
      System.exit(1); // Don't duplicate the error to the user or bubble up the exception.
    } catch (Exception e) {
      logger.log(SEVERE, "Error during merging manifests", e);
      throw e; // This is a proper internal exception, so we bubble it up.
    }
  }
}
