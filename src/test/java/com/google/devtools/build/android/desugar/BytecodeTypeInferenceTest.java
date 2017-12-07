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
package com.google.devtools.build.android.desugar;

import com.google.common.io.Files;
import com.google.common.truth.Truth;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test for {@link BytecodeTypeInference} */
@RunWith(JUnit4.class)
public class BytecodeTypeInferenceTest {

  private static final Path JAR_PATH = Paths.get(System.getProperty("jar_path"));
  private static final Path GOLDEN_PATH = Paths.get(System.getProperty("golden_file"));

  @Test
  public void test() throws IOException {
    StringWriter stringWriter = new StringWriter();
    try (PrintWriter printWriter = new PrintWriter(stringWriter)) {
      ByteCodeTypePrinter.printClassesWithTypes(JAR_PATH, printWriter);
      printWriter.close();
    }
    String inferenceResult = stringWriter.toString().trim();
    String golden = Files.asCharSource(GOLDEN_PATH.toFile(), StandardCharsets.UTF_8).read().trim();
    Truth.assertThat(inferenceResult).isEqualTo(golden);
  }
}
