// Copyright 2025 The Bazel Authors. All rights reserved.
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

package com.google.devtools.coverageoutputgenerator;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.util.List;
import java.util.zip.GZIPOutputStream;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link GcovJsonParser}. */
@RunWith(JUnit4.class)
public final class GcovJsonParserTest {

  @Test
  public void testParseJsonData() throws Exception {
    String jsonData =
        """
            {
              "format_version": "2",
              "gcc_version": "14.2.0",
              "current_working_directory": "/foo/bar",
              "data_file": "baz.gcno",
              "files": [{
                  "file": "baz.cc",
                  "functions": [
                      {
                        "name": "main",
                        "demangled_name": "main",
                        "start_line": 4,
                        "start_column": 5,
                        "end_line": 11,
                        "end_columnn": 1,
                        "blocks": 7,
                        "block_executed": 5,
                        "execution_count": 2
                      }
                ],
                "lines": [
                  {
                    "line_number": 4,
                    "function_name": "main",
                    "count": 3,
                    "unexecuted_block": false,
                    "block_ids": [2, 3],
                    "branches": [],
                    "calls": [],
                    "conditions": []
                  },
                  {
                    "line_number": 5,
                    "function_name": "main",
                    "count": 3,
                    "unexecuted_block": false,
                    "block_ids": [2, 3],
                    "branches": [
                        {
                          "count": 1,
                          "throw": false,
                          "fallthrough": true,
                          "source_block_id": 3,
                          "destination_block_id": 4
                        },
                        {
                          "count": 2,
                          "throw": false,
                          "fallthrough": false,
                          "source_block_id": 3,
                          "destination_block_id": 5
                        }
                    ],
                    "calls": [],
                    "conditions": []
                  },
                  {
                    "line_number": 6,
                    "function_name": "main",
                    "count": 1,
                    "unexecuted_block": false,
                    "block_ids": [4],
                    "branches": [],
                    "calls": [],
                    "conditions": []
                  },
                  {
                    "line_number": 7,
                    "function_name": "main",
                    "count": 3,
                    "unexecuted_block": false,
                    "block_ids": [5],
                    "branches": [],
                    "calls": [],
                    "conditions": []
                  }
                ]
              }]
            }
        """;
    ByteArrayOutputStream gzipBytes = new ByteArrayOutputStream();
    try (GZIPOutputStream gzipStream = new GZIPOutputStream(gzipBytes)) {
      gzipStream.write(jsonData.getBytes(UTF_8));
    }
    ByteArrayInputStream inputStream = new ByteArrayInputStream(gzipBytes.toByteArray());

    List<SourceFileCoverage> sourceFiles = GcovJsonParser.parse(inputStream);

    assertThat(sourceFiles).hasSize(1);
    assertThat(sourceFiles.get(0).sourceFileName()).isEqualTo("baz.cc");
    assertThat(sourceFiles.get(0).getFunctionsExecution()).containsExactly("main", 2L);
    assertThat(sourceFiles.get(0).getLines()).containsExactly(4, 3L, 5, 3L, 6, 1L, 7, 3L);
    assertThat(sourceFiles.get(0).getAllBranches())
        .containsExactly(
            BranchCoverage.createWithBlockAndBranch(5, "0", "0", true, 1),
            BranchCoverage.createWithBlockAndBranch(5, "0", "1", true, 2));
  }
}
