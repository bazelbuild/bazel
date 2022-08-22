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

import com.google.common.flogger.GoogleLogger;
import com.google.devtools.common.options.OptionsParser;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.yaml.snakeyaml.DumperOptions;
import org.yaml.snakeyaml.Yaml;

/** The main class for the TOC contents updater. */
public class TableOfContentsUpdater {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private static final String VERSION_ROOT = "/versions/";

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
    try (FileInputStream fis = new FileInputStream(options.inputPath)) {
      Object data = yaml.load(fis);
      update(data, options.version, options.maxReleases);
      yaml.dump(data, new OutputStreamWriter(new FileOutputStream(options.outputPath), UTF_8));
    } catch (Throwable t) {
      System.err.printf("ERROR: %s\n", t.getMessage());
      logger.atSevere().withCause(t).log(
          "Failed to transform TOC from %s to %s", options.inputPath, options.outputPath);
      Runtime.getRuntime().exit(1);
    }
  }

  private static void printUsage() {
    System.err.println(
        "Usage: toc-updater -i src_toc_path -o dest_toc_path -v version [-m max_releases] [-h]\n\n"
            + "Reads the input TOC, adds an entry for the specified version and saves the new TOC"
            + " at the specified location.\n");
  }

  private static DumperOptions getYamlOptions() {
    DumperOptions opts = new DumperOptions();
    opts.setIndent(2);
    opts.setPrettyFlow(true);
    opts.setDefaultFlowStyle(DumperOptions.FlowStyle.BLOCK);
    return opts;
  }

  private static void update(Object data, String version, int maxReleases) {
    @SuppressWarnings("unchecked") // yaml deserialization
    Map<String, List<Map<String, String>>> m = (Map<String, List<Map<String, String>>>) data;
    List<Map<String, String>> toc = (List<Map<String, String>>) m.get("toc");
    if (toc == null) {
      throw new IllegalStateException("Missing 'toc' element.");
    }

    Map<String, String> newEntry = new HashMap<>();
    newEntry.put("path", String.format("%s%s", VERSION_ROOT, version));
    newEntry.put("label", version);

    toc.add(0, newEntry);
    if (toc.size() > maxReleases) {
      m.put("toc", toc.subList(0, maxReleases));
    }
  }
}
