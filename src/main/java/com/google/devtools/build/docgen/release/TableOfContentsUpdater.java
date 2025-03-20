// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.docgen.release;

import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.stream.Collectors.joining;

import com.google.common.flogger.GoogleLogger;
import com.google.devtools.common.options.OptionsParser;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;
import org.yaml.snakeyaml.DumperOptions;
import org.yaml.snakeyaml.Yaml;

/** The main class for the TOC contents updater. */
public class TableOfContentsUpdater {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private static final String VERSION_ROOT = "/versions/";

  private static final String VERSION_INDICATOR_START = "<!-- BEGIN_VERSION_INDICATOR -->";

  private static final String VERSION_INDICATOR_END = "<!-- END_VERSION_INDICATOR -->";

  private static final String VERSION_INDICATOR_TEMPLATE =
      """
·
{% dynamic if setvar.version == "{canonical_version}" %}
<strong>{pretty_version}</strong>
{% dynamic else %}
<a href="{version_root}{canonical_version}/{% dynamic print setvar.original_path %}">
{pretty_version}</a>
{% dynamic endif %}
""";

  private TableOfContentsUpdater() {}

  public static void main(String[] args) {
    OptionsParser parser =
        OptionsParser.builder().optionsClasses(TableOfContentsOptions.class).build();
    parser.parseAndExitUponError(args);
    TableOfContentsOptions options = parser.getOptions(TableOfContentsOptions.class);

    if (options.printHelp) {
      printUsage();
      Runtime.getRuntime().exit(0);
    }

    if (!options.isValid()) {
      printUsage();
      Runtime.getRuntime().exit(1);
    }

    Yaml yaml = new Yaml(getYamlOptions());
    List<String> versions;
    try (FileInputStream fis = new FileInputStream(options.inputPath)) {
      Object data = yaml.load(fis);
      versions = updateTocAndGetVersions(data, options.version, options.maxReleases);
      yaml.dump(data, new OutputStreamWriter(new FileOutputStream(options.outputPath), UTF_8));
    } catch (Throwable t) {
      System.err.printf("ERROR: %s\n", t.getMessage());
      logger.atSevere().withCause(t).log(
          "Failed to transform TOC from %s to %s", options.inputPath, options.outputPath);
      Runtime.getRuntime().exit(1);
      throw new IllegalStateException("Not reached");
    }

    if (!options.versionIndicatorInputPath.isEmpty()) {
      try {
        Files.writeString(
            Path.of(options.versionIndicatorOutputPath),
            makeUpdatedVersionIndicator(
                Files.readString(Path.of(options.versionIndicatorInputPath)), versions));
      } catch (Throwable t) {
        System.err.printf("ERROR: %s\n", t.getMessage());
        logger.atSevere().withCause(t).log(
            "Failed to update version indicator from %s to %s",
            options.versionIndicatorInputPath, options.versionIndicatorOutputPath);
        Runtime.getRuntime().exit(1);
      }
    }
  }

  private static void printUsage() {
    System.err.println(
        """
Usage: toc-updater -i src_toc_path -o dest_toc_path -v version [-m max_releases] [-h] \
[--version_indicator_input path --version_indicator_output path]

Reads the input TOC, adds an entry for the specified version and saves the new TOC\
 at the specified location.
""");
  }

  private static DumperOptions getYamlOptions() {
    DumperOptions opts = new DumperOptions();
    opts.setIndent(2);
    opts.setPrettyFlow(true);
    opts.setDefaultFlowStyle(DumperOptions.FlowStyle.BLOCK);
    return opts;
  }

  private static List<String> updateTocAndGetVersions(
      Object data, String version, int maxReleases) {
    @SuppressWarnings("unchecked") // yaml deserialization
    Map<String, List<Map<String, String>>> m = (Map<String, List<Map<String, String>>>) data;
    List<Map<String, String>> toc = m.get("toc");
    if (toc == null) {
      throw new IllegalStateException("Missing 'toc' element.");
    }

    Map<String, String> newEntry = new HashMap<>();
    newEntry.put("path", String.format("%s%s", VERSION_ROOT, version));
    newEntry.put("label", version);

    toc.addFirst(newEntry);
    if (toc.size() > maxReleases) {
      m.put("toc", toc.subList(0, maxReleases));
    }

    return m.get("toc").stream()
        // Exclude legacy doc versions.
        .filter(e -> e.get("path").startsWith(VERSION_ROOT))
        .map(e -> e.get("label"))
        .map(TableOfContentsUpdater::canonicalizeVersion)
        .toList();
  }

  private static String makeUpdatedVersionIndicator(
      String oldVersionIndicator, List<String> versions) {
    int beginPos = oldVersionIndicator.indexOf(VERSION_INDICATOR_START);
    int endPos = oldVersionIndicator.indexOf(VERSION_INDICATOR_END);
    if (beginPos == -1 || endPos == -1) {
      throw new IllegalStateException("Version indicator markers not found.");
    }
    // Include the line terminator.
    String prefix =
        oldVersionIndicator.substring(0, beginPos + VERSION_INDICATOR_START.length() + 1);
    String suffix = oldVersionIndicator.substring(endPos);
    return versions.stream()
        .map(
            version ->
                VERSION_INDICATOR_TEMPLATE
                    .replace("{canonical_version}", version)
                    .replace("{pretty_version}", prettifyVersion(version))
                    .replace("{version_root}", VERSION_ROOT))
        .collect(joining("", prefix, suffix));
  }

  private static String canonicalizeVersion(String version) {
    if (version.split(Pattern.quote(".")).length < 3) {
      return version + ".0";
    } else {
      return version;
    }
  }

  private static String prettifyVersion(String version) {
    if (version.endsWith(".0")) {
      return version.substring(0, version.length() - 2);
    } else {
      return version;
    }
  }
}
