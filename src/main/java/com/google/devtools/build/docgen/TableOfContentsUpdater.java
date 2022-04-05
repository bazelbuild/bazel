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
package com.google.devtools.build.docgen;

import java.io.InputStreamReader;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.HashMap;

import org.yaml.snakeyaml.DumperOptions;
import org.yaml.snakeyaml.Yaml;
import com.google.devtools.common.options.OptionsParser;

public class TableOfContentsUpdater {
    private final static String VERSION_ROOT = "/versions/";

    public static void main(String[] args) throws Exception {
        OptionsParser parser =
        OptionsParser.builder().optionsClasses(TableOfContentsOptions.class).build();
        parser.parseAndExitUponError(args);
        TableOfContentsOptions options = parser.getOptions(TableOfContentsOptions.class);

        if (options.help) {
            printUsage();
            Runtime.getRuntime().exit(0);
        }

        if (options.inputPath.isEmpty()
            || options.outputPath.isEmpty()
            || options.version.isEmpty()
            || options.maxReleases < 1) {
            printUsage();
            Runtime.getRuntime().exit(1);
        }

        Yaml yaml = new Yaml(getYamlOptions());
        try {
            try (FileInputStream fis = new FileInputStream(options.inputPath)) {
                Object data = yaml.load(fis);
                update(data, options.version, options.maxReleases);
                yaml.dump(data, new FileWriter(options.outputPath));
            }
          } catch (Throwable t) {
            System.err.printf("ERROR: %s\n", t.getMessage());
            t.printStackTrace();
          }
    }

    private static void printUsage() {
        System.err.println(
            "Usage: toc-updater -i src_toc_path -o dest_toc_path -v version [-m max_releases] [-h]\n\n"
                + "Reads the input TOC, adds an entry for the specified version and saves the new TOC at the specified location.\n");
    }

    private static DumperOptions getYamlOptions() {
        DumperOptions opts = new DumperOptions();
        opts.setIndent(2);
        opts.setPrettyFlow(true);
        opts.setDefaultFlowStyle(DumperOptions.FlowStyle.BLOCK);
        return opts;
    }

    private static void update(Object data, String version, int maxReleases) {
        Map m = (Map) data;
        List toc = (List) m.get("toc");
        if (toc == null) {
            throw new IllegalStateException("Missing 'toc' element.");
        }
        
        Map<String, String> newEntry = new HashMap<>();
        newEntry.put("path", String.format("%s%s", VERSION_ROOT, version));
        newEntry.put("label", version);
        
        toc.add(0, newEntry);
        while (toc.size() > maxReleases) {
            toc.remove(toc.size() - 1);
        }
    }
}