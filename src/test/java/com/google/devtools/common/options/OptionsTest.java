// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.common.options;

import static com.google.common.truth.Truth.assertThat;
import static java.util.Arrays.asList;
import static org.junit.Assert.assertThrows;

import com.google.common.testing.EqualsTester;
import java.net.MalformedURLException;
import java.net.URL;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Test for {@link Options}.
 */
@RunWith(JUnit4.class)
public class OptionsTest {

  private static final String[] NO_ARGS = {};

  public static class HttpOptions extends OptionsBase {

    @Option(
      name = "host",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "www.google.com",
      help = "The URL at which the server will be running."
    )
    public String host;

    @Option(
      name = "port",
      abbrev = 'p',
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "80",
      help = "The port at which the server will be running."
    )
    public int port;

    @Option(
      name = "debug",
      abbrev = 'd',
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "false",
      help = "debug"
    )
    public boolean isDebugging;

    @Option(
      name = "tristate",
      abbrev = 't',
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "auto",
      help = "tri-state option returning auto by default"
    )
    public TriState triState;

    @Option(
      name = "special",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "null",
      expansion = {"--host=special.google.com", "--port=8080"}
    )
    public Void special;
  }

  @Test
  public void paragraphFill() throws Exception {
    // TODO(bazel-team): don't include trailing space after last word in line.
    String input = "The quick brown fox jumps over the lazy dog.";

    assertThat(OptionsUsage.paragraphFill(input, 2, 13))
        .isEqualTo("  The quick \n  brown fox \n  jumps over \n  the lazy \n" + "  dog.");
    assertThat(OptionsUsage.paragraphFill(input, 3, 19))
        .isEqualTo("   The quick brown \n   fox jumps over \n   the lazy dog.");

    String input2 = "The quick brown fox jumps\nAnother paragraph.";
    assertThat(OptionsUsage.paragraphFill(input2, 2, 23))
        .isEqualTo("  The quick brown fox \n  jumps\n  Another paragraph.");
  }

  @Test
  public void getsDefaults() throws OptionsParsingException {
    Options<HttpOptions> options = Options.parse(HttpOptions.class, NO_ARGS);
    String[] remainingArgs = options.getRemainingArgs();
    HttpOptions webFlags = options.getOptions();

    assertThat(webFlags.host).isEqualTo("www.google.com");
    assertThat(webFlags.port).isEqualTo(80);
    assertThat(webFlags.isDebugging).isFalse();
    assertThat(webFlags.triState).isEqualTo(TriState.AUTO);
    assertThat(remainingArgs).hasLength(0);
  }

  @Test
  public void objectMethods() throws OptionsParsingException {
    String[] args = { "--host", "foo", "--port", "80" };
    HttpOptions left =
        Options.parse(HttpOptions.class, args).getOptions();
    HttpOptions likeLeft =
        Options.parse(HttpOptions.class, args).getOptions();
    String [] rightArgs = {"--host", "other", "--port", "90" };
    HttpOptions right =
        Options.parse(HttpOptions.class, rightArgs).getOptions();

    String toString = left.toString();
    // Don't rely on Set.toString iteration order:
    assertThat(toString)
        .startsWith("com.google.devtools.common.options.OptionsTest" + "$HttpOptions{");
    assertThat(toString).contains("host=foo");
    assertThat(toString).contains("port=80");
    assertThat(toString).endsWith("}");

    new EqualsTester().addEqualityGroup(left).testEquals();
    assertThat(left.toString()).isEqualTo(likeLeft.toString());
    assertThat(left).isEqualTo(likeLeft);
    assertThat(likeLeft).isEqualTo(left);
    assertThat(left).isNotEqualTo(right);
    assertThat(right).isNotEqualTo(left);
    assertThat(left).isNotNull();
    assertThat(likeLeft).isNotNull();
    assertThat(likeLeft.hashCode()).isEqualTo(likeLeft.hashCode());
    assertThat(likeLeft.hashCode()).isEqualTo(left.hashCode());
    // Strictly speaking this is not required for hashCode to be correct,
    // but a good hashCode should be different at least for some values. So,
    // we're making sure that at least this particular pair of inputs yields
    // different values.
    assertThat(left.hashCode()).isNotEqualTo(right.hashCode());
  }

  @Test
  public void equals() throws OptionsParsingException {
    String[] args = { "--host", "foo", "--port", "80" };
    HttpOptions options1 =  Options.parse(HttpOptions.class, args).getOptions();

    String[] args2 = { "-p", "80", "--host", "foo" };
    HttpOptions options2 =  Options.parse(HttpOptions.class, args2).getOptions();
    // Order/abbreviations shouldn't matter.
    assertThat(options1).isEqualTo(options2);

    // Explicitly setting a default shouldn't matter.
    assertThat(Options.parse(HttpOptions.class, "--port", "80").getOptions())
        .isEqualTo(Options.parse(HttpOptions.class).getOptions());

    assertThat(Options.parse(HttpOptions.class, "--port", "3").getOptions())
        .isNotEqualTo(Options.parse(HttpOptions.class).getOptions());
  }

  @Test
  public void getsFlagsProvidedInArguments()
      throws OptionsParsingException {
    String[] args = {"--host", "google.com",
                     "-p", "8080",  // short form
                     "--debug"};
    Options<HttpOptions> options = Options.parse(HttpOptions.class, args);
    String[] remainingArgs = options.getRemainingArgs();
    HttpOptions webFlags = options.getOptions();

    assertThat(webFlags.host).isEqualTo("google.com");
    assertThat(webFlags.port).isEqualTo(8080);
    assertThat(webFlags.isDebugging).isTrue();
    assertThat(remainingArgs).hasLength(0);
  }

  @Test
  public void getsFlagsProvidedWithEquals() throws OptionsParsingException {
    String[] args = {"--host=google.com",
                     "--port=8080",
                     "--debug"};
    Options<HttpOptions> options = Options.parse(HttpOptions.class, args);
    String[] remainingArgs = options.getRemainingArgs();
    HttpOptions webFlags = options.getOptions();

    assertThat(webFlags.host).isEqualTo("google.com");
    assertThat(webFlags.port).isEqualTo(8080);
    assertThat(webFlags.isDebugging).isTrue();
    assertThat(remainingArgs).hasLength(0);
  }

  @Test
  public void booleanNo() throws OptionsParsingException {
    Options<HttpOptions> options =
        Options.parse(HttpOptions.class, new String[]{"--nodebug", "--notristate"});
    HttpOptions webFlags = options.getOptions();
    assertThat(webFlags.isDebugging).isFalse();
    assertThat(webFlags.triState).isEqualTo(TriState.NO);
  }

  @Test
  public void booleanAbbrevMinus() throws OptionsParsingException {
    Options<HttpOptions> options =
        Options.parse(HttpOptions.class, new String[]{"-d-", "-t-"});
    HttpOptions webFlags = options.getOptions();
    assertThat(webFlags.isDebugging).isFalse();
    assertThat(webFlags.triState).isEqualTo(TriState.NO);
  }

  @Test
  public void boolean0() throws OptionsParsingException {
    Options<HttpOptions> options =
        Options.parse(HttpOptions.class, new String[]{"--debug=0", "--tristate=0"});
    HttpOptions webFlags = options.getOptions();
    assertThat(webFlags.isDebugging).isFalse();
    assertThat(webFlags.triState).isEqualTo(TriState.NO);
  }

  @Test
  public void boolean1() throws OptionsParsingException {
    Options<HttpOptions> options =
        Options.parse(HttpOptions.class, new String[]{"--debug=1", "--tristate=1"});
    HttpOptions webFlags = options.getOptions();
    assertThat(webFlags.isDebugging).isTrue();
    assertThat(webFlags.triState).isEqualTo(TriState.YES);
  }

  @Test
  public void retainsStuffThatsNotOptions() throws OptionsParsingException {
    String[] args = {"these", "aint", "options"};
    Options<HttpOptions> options = Options.parse(HttpOptions.class, args);
    String[] remainingArgs = options.getRemainingArgs();
    assertThat(asList(remainingArgs)).isEqualTo(asList(args));
  }

  @Test
  public void retainsStuffThatsNotComplexOptions()
      throws OptionsParsingException {
    String[] args = {"--host", "google.com",
                     "notta",
                     "--port=8080",
                     "option",
                     "--debug=true"};
    String[] notoptions = {"notta", "option" };
    Options<HttpOptions> options = Options.parse(HttpOptions.class, args);
    String[] remainingArgs = options.getRemainingArgs();
    assertThat(asList(remainingArgs)).isEqualTo(asList(notoptions));
  }

  @Test
  public void wontParseUnknownOptions() {
    String [] args = { "--unknown", "--other=23", "--options" };
    OptionsParsingException e =
        assertThrows(OptionsParsingException.class, () -> Options.parse(HttpOptions.class, args));
    assertThat(e).hasMessageThat().isEqualTo("Unrecognized option: --unknown");
  }

  @Test
  public void requiresOptionValue() {
    String[] args = {"--port"};
    OptionsParsingException e =
        assertThrows(OptionsParsingException.class, () -> Options.parse(HttpOptions.class, args));
    assertThat(e).hasMessageThat().isEqualTo("Expected value after --port");
  }

  @Test
  public void handlesDuplicateOptions_full() throws Exception {
    String[] args = {"--port=80", "--port", "81"};
    Options<HttpOptions> options = Options.parse(HttpOptions.class, args);
    HttpOptions webFlags = options.getOptions();
    assertThat(webFlags.port).isEqualTo(81);
  }

  @Test
  public void handlesDuplicateOptions_abbrev() throws Exception {
    String[] args = {"--port=80", "-p", "81"};
    Options<HttpOptions> options = Options.parse(HttpOptions.class, args);
    HttpOptions webFlags = options.getOptions();
    assertThat(webFlags.port).isEqualTo(81);
  }

  @Test
  public void duplicateOptionsOkWithSameValues() throws Exception {
    // These would throw OptionsParsingException if they failed.
    Options.parse(HttpOptions.class,"--port=80", "--port", "80");
    Options.parse(HttpOptions.class, "--port=80", "-p", "80");
  }

  @Test
  public void isPickyAboutBooleanValues() {
    OptionsParsingException e =
        assertThrows(
            OptionsParsingException.class,
            () -> Options.parse(HttpOptions.class, new String[] {"--debug=not_a_boolean"}));
    assertThat(e)
        .hasMessageThat()
        .isEqualTo(
            "While parsing option --debug=not_a_boolean: " + "\'not_a_boolean\' is not a boolean");
  }

  @Test
  public void isPickyAboutBooleanNos() {
    OptionsParsingException e =
        assertThrows(
            OptionsParsingException.class,
            () -> Options.parse(HttpOptions.class, new String[] {"--nodebug=1"}));
    assertThat(e).hasMessageThat().isEqualTo("Unexpected value after boolean option: --nodebug=1");
  }

  @Test
  public void usageForBuiltinTypesNoExpansion() {
    String usage = Options.getUsage(HttpOptions.class);
    // We can't rely on the option ordering.
    assertThat(usage)
        .contains("  --[no]debug [-d] (a boolean; default: \"false\")\n" + "    debug");
    assertThat(usage)
        .contains(
            "  --host (a string; default: \"www.google.com\")\n"
                + "    The URL at which the server will be running.");
    assertThat(usage)
        .contains(
            "  --port [-p] (an integer; default: \"80\")\n"
                + "    The port at which the server will be running.");
    assertThat(usage)
        .contains(
            "  --[no]tristate [-t] (a tri-state (auto, yes, no); default: \"auto\")\n"
                + "    tri-state option returning auto by default");
  }

  @Test
  public void usageForStaticExpansion() {
    String usage = Options.getUsage(HttpOptions.class);
    assertThat(usage)
        .contains("  --special\n      Expands to: --host=special.google.com --port=8080");
  }

  public static class NullTestOptions extends OptionsBase {
    @Option(
      name = "host",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "null",
      help = "The URL at which the server will be running."
    )
    public String host;

    @Option(
      name = "none",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "null",
      expansion = {"--host=www.google.com"},
      help = "An expanded option."
    )
    public Void none;
  }

  @Test
  public void usageForNullDefault() {
    String usage = Options.getUsage(NullTestOptions.class);
    assertThat(usage)
        .contains(
            "  --host (a string; default: see description)\n"
                + "    The URL at which the server will be running.");
    assertThat(usage)
        .contains(
            "  --none\n" + "    An expanded option.\n" + "      Expands to: --host=www.google.com");
  }

  public static class MyURLConverter extends Converter.Contextless<URL> {

    @Override
    public URL convert(String input) throws OptionsParsingException {
      try {
        return new URL(input);
      } catch (MalformedURLException e) {
        throw new OptionsParsingException("Could not convert '" + input + "': "
                                          + e.getMessage());
      }
    }

    @Override
    public String getTypeDescription() {
      return "a url";
    }

  }

  public static class UsesCustomConverter extends OptionsBase {

    @Option(
      name = "url",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "http://www.google.com/",
      converter = MyURLConverter.class
    )
    public URL url;
  }

  @Test
  public void customConverter() throws Exception {
    Options<UsesCustomConverter> options =
      Options.parse(UsesCustomConverter.class, new String[0]);
    URL expected = new URL("http://www.google.com/");
    assertThat(options.getOptions().url).isEqualTo(expected);
  }

  @Test
  public void customConverterThrowsException() throws Exception {
    String[] args = {"--url=a_malformed:url"};
    OptionsParsingException e =
        assertThrows(
            OptionsParsingException.class, () -> Options.parse(UsesCustomConverter.class, args));
    assertThat(e)
        .hasMessageThat()
        .isEqualTo(
            "While parsing option --url=a_malformed:url: "
                + "Could not convert 'a_malformed:url': "
                + "no protocol: a_malformed:url");
  }

  @Test
  public void usageWithCustomConverter() {
    assertThat(Options.getUsage(UsesCustomConverter.class))
        .isEqualTo("  --url (a url; default: \"http://www.google.com/\")\n");
  }

  @Test
  public void unknownBooleanOption() {
    OptionsParsingException e =
        assertThrows(
            OptionsParsingException.class,
            () -> Options.parse(HttpOptions.class, new String[] {"--no-debug"}));
    assertThat(e)
        .hasMessageThat()
        .isEqualTo("Unrecognized option: --no-debug (did you mean '--nodebug'?)");
  }

  public static class J extends OptionsBase {
    @Option(
      name = "j",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "null"
    )
    public String string;
  }
  @Test
  public void nullDefaultForReferenceTypeOption() throws Exception {
    J options = Options.parse(J.class, NO_ARGS).getOptions();
    assertThat(options.string).isNull();
  }

  @Test
  public void nullIsNotInterpretedSpeciallyExceptAsADefaultValue() throws Exception {
    HttpOptions options =
        Options.parse(HttpOptions.class,
                      new String[] { "--host", "null" }).getOptions();
    assertThat(options.host).isEqualTo("null");
  }

  @Test
  public void nonDecimalRadicesForIntegerOptions() throws Exception {
    Options<HttpOptions> options =
        Options.parse(HttpOptions.class, new String[] { "--port", "0x51"});
    assertThat(options.getOptions().port).isEqualTo(81);
  }

  @Test
  public void expansionOptionSimple() throws Exception {
    Options<HttpOptions> options =
      Options.parse(HttpOptions.class, new String[] {"--special"});
    assertThat(options.getOptions().host).isEqualTo("special.google.com");
    assertThat(options.getOptions().port).isEqualTo(8080);
  }

  @Test
  public void expansionOptionOverride() throws Exception {
    Options<HttpOptions> options =
      Options.parse(HttpOptions.class, new String[] {"--port=90", "--special", "--host=foo"});
    assertThat(options.getOptions().host).isEqualTo("foo");
    assertThat(options.getOptions().port).isEqualTo(8080);
  }

  @Test
  public void expansionOptionEquals() throws Exception {
    Options<HttpOptions> options1 =
      Options.parse(HttpOptions.class, new String[] { "--host=special.google.com", "--port=8080"});
    Options<HttpOptions> options2 =
      Options.parse(HttpOptions.class, new String[] { "--special" });
    assertThat(options1.getOptions()).isEqualTo(options2.getOptions());
  }
}
