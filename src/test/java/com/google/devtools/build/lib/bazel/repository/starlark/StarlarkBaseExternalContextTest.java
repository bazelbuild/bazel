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

package com.google.devtools.build.lib.bazel.repository.starlark;

import static org.junit.Assert.fail;

import com.google.devtools.build.lib.bazel.repository.decompressor.DecompressorValue;
import org.apache.commons.lang3.text.WordUtils;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ErrorCollector;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class StarlarkBaseExternalContextTest {
  @Rule public ErrorCollector collector = new ErrorCollector();

  @Test
  public void docSupportedFormats() {
    String expected = DecompressorValue.readableSupportedFormats("\"", "\"", "or");
    String observed = StarlarkBaseExternalContext.SUPPORTED_DECOMPRESSION_FORMATS;
    String copyPasteCode =
        "  static final String SUPPORTED_DECOMPRESSION_FORMATS =\n"
            + "\"\"\"\n"
            + WordUtils.wrap(
                expected,
                /* wrapLength= */ 80,
                /* newLineStr= */ " \\\n",
                /* wrapLongWords= */ false)
            + "\"\"\";";

    // Delimit the quote at the end of the text block (distinguish from ending three quotes """).
    copyPasteCode =
        copyPasteCode.substring(0, copyPasteCode.length() - 5)
            + "\\"
            + copyPasteCode.substring(copyPasteCode.length() - 5, copyPasteCode.length());

    if (!observed.equals(expected)) {
      fail(
          String.format(
              """


              Expected:
              \t%1$s
              Got:
              \t%2$s

              Copy-paste string to replace in StarlarkBaseExternalContext.java: \s

              %3$s
              """,
              expected, observed, copyPasteCode));
    }
  }
}
