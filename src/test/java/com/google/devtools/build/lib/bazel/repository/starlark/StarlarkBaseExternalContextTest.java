package com.google.devtools.build.lib.bazel.repository.starlark;

import com.google.devtools.build.lib.bazel.repository.decompressor.DecompressorValue;
import org.apache.commons.lang3.text.WordUtils;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ErrorCollector;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import static org.junit.Assert.fail;

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
              "\n\nExpected:\n\t%1$s\n"
                  + "Got:\n\t%2$s\n\n"
                  + "Copy-paste string to replace in StarlarkBaseExternalContext.java:  \n\n%3$s\n",
              expected, observed, copyPasteCode));
    }
  }
}
