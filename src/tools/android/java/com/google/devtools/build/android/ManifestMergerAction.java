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
import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.android.Converters.CompatExistingPathConverter;
import com.google.devtools.build.android.Converters.CompatExistingPathStringDictionaryConverter;
import com.google.devtools.build.android.Converters.CompatPathConverter;
import com.google.devtools.build.android.Converters.CompatStringDictionaryConverter;
import java.io.IOException;
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
 *       --mergeManifestPermissions merge manifest uses-permissions
 * </pre>
 */
public class ManifestMergerAction {
  /** Flag specifications for this action. */
  @Parameters(separators = "= ")
  public static final class Options {
    @Parameter(
        names = "--manifest",
        converter = CompatExistingPathConverter.class,
        description =
            "Path of primary manifest. If not passed, a dummy manifest will be generated and used"
                + " as the primary.")
    public Path manifest;

    @Parameter(
        names = "--mergeeManifests",
        converter = CompatExistingPathStringDictionaryConverter.class,
        description =
            "A dictionary of manifests, and originating target, to be merged into manifest.")
    public Map<Path, String> mergeeManifests = ImmutableMap.of();

    @Parameter(names = "--mergeType", description = "The type of merging to perform.")
    public MergeType mergeType = MergeType.APPLICATION;

    @Parameter(
        names = "--manifestValues",
        converter = CompatStringDictionaryConverter.class,
        description =
            "A dictionary string of values to be overridden in the manifest. Any instance of"
                + " ${name} in the manifest will be replaced with the value corresponding to name"
                + " in this dictionary. applicationId, versionCode, versionName, minSdkVersion,"
                + " targetSdkVersion and maxSdkVersion have a dual behavior of also overriding the"
                + " corresponding attributes of the manifest and uses-sdk tags. packageName will be"
                + " ignored and will be set from either applicationId or the package in manifest."
                + " The expected format of this string is: key:value[,key:value]*. The keys and"
                + " values may contain colons and commas as long as they are escaped with a"
                + " backslash.")
    public Map<String, String> manifestValues = ImmutableMap.of();

    @Parameter(
        names = "--customPackage",
        description = "Custom java package to insert in the package attribute of the manifest tag.")
    public String customPackage;

    @Parameter(
        names = "--manifestOutput",
        converter = CompatPathConverter.class,
        description = "Path for the merged manifest.")
    public Path manifestOutput;

    @Parameter(
        names = "--log",
        converter = CompatPathConverter.class,
        description = "Path to where the merger log should be written.")
    public Path log;

    @Parameter(
        names = "--mergeManifestPermissions",
        arity = 1,
        description = "If enabled, manifest permissions will be merged.")
    public boolean mergeManifestPermissions;
  }

  private static final String[] PERMISSION_TAGS =
      new String[] {"uses-permission", "uses-permission-sdk-23"};
  private static final StdLogger stdLogger = new StdLogger(StdLogger.Level.WARNING);
  private static final Logger logger = Logger.getLogger(ManifestMergerAction.class.getName());

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
    // Write resulting manifest to a tmp file to prevent collisions
    Path output = Files.createTempFile(outputDir, "AndroidManifest", ".xml");
    TransformerFactory.newInstance()
        .newTransformer()
        .transform(new DOMSource(doc), new StreamResult(output.toFile()));
    return output;
  }

  public static void main(String[] args) throws Exception {
    // First parse the local Action flags using JCommander, then parse the remaining common flags
    // using OptionsParser.
    Options options = new Options();
    ResourceProcessorCommonOptions resourceProcessorCommonOptions =
        new ResourceProcessorCommonOptions();
    Object[] allOptions = new Object[] {options, resourceProcessorCommonOptions};
    JCommander jc = new JCommander(allOptions);
    String[] preprocessedArgs = AndroidOptionsUtils.runArgFilePreprocessor(jc, args);
    String[] normalizedArgs =
        AndroidOptionsUtils.normalizeBooleanOptions(allOptions, preprocessedArgs);
    jc.parse(normalizedArgs);

    try {
      Path mergedManifest;
      AndroidManifestProcessor manifestProcessor = AndroidManifestProcessor.with(stdLogger);

      Path tmp = Files.createTempDirectory("manifest_merge_tmp");
      tmp.toFile().deleteOnExit();
      ImmutableMap.Builder<Path, String> mergeeManifests = ImmutableMap.builder();
      for (Map.Entry<Path, String> mergeeManifest : options.mergeeManifests.entrySet()) {
        if (!options.mergeManifestPermissions) {
          // Remove uses-permission tags from mergees before the merge.
          mergeeManifests.put(
              removePermissions(mergeeManifest.getKey(), tmp), mergeeManifest.getValue());
        } else {
          mergeeManifests.put(mergeeManifest);
        }
      }

      Path manifest = options.manifest;
      if (manifest == null) {
        // No primary manifest was passed. Generate a dummy primary.

        String minSdk = options.manifestValues.getOrDefault("minSdkVersion", "1");
        manifest =
            AndroidManifest.of("", minSdk).writeDummyManifestForAapt(tmp, options.customPackage);
      }

      mergedManifest =
          manifestProcessor.mergeManifest(
              manifest,
              mergeeManifests.buildOrThrow(),
              options.mergeType,
              options.manifestValues,
              options.customPackage,
              options.manifestOutput,
              options.log,
              resourceProcessorCommonOptions.logWarnings);
      // Bazel expects a log file output as a result of manifest merging, even if it is a no-op.
      if (options.log != null && !options.log.toFile().exists()) {
        options.log.toFile().createNewFile();
      }
      if (!mergedManifest.equals(options.manifestOutput)) {
        // manifestProcess.mergeManifest returns the merged manifest, or, if merging was a no-op,
        // the original primary manifest. In the latter case, explicitly copy that primary manifest
        // to the expected location of the output.
        Files.copy(manifest, options.manifestOutput, StandardCopyOption.REPLACE_EXISTING);
      }
    } catch (AndroidManifestProcessor.ManifestProcessingException e) {
      // ManifestProcessingExceptions represent build errors that should be delivered directly to
      // ResourceProcessorBusyBox where the exception can be delivered with a non-zero status code
      // to the worker/process
      // Note that this exception handler is nearly identical to the generic case, except that it
      // does not have a log print associated with it. This is because the exception will bubble up
      // to ResourceProcessorBusyBox, which will print an identical error message. It is preferable
      // to slightly convolute this try/catch block, rather than pollute the user's console with
      // extra repeated error messages.
      throw e;
    } catch (Exception e) {
      logger.log(SEVERE, "Error during merging manifests", e);
      throw e; // This is a proper internal exception, so we bubble it up.
    }
  }
}
