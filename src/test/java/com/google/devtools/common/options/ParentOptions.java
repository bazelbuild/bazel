package com.google.devtools.common.options;

public class ParentOptions extends OptionsBase {
  public static final String TEST_STRING_DEFAULT = "parent test string default";

  @Option(
      name = "parent_test_string",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = TEST_STRING_DEFAULT,
      help = "a string-valued option to test simple option operations")
  public String testString;
}
