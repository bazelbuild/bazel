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
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

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

    @Option(name = "host",
            defaultValue = "www.google.com",
            help = "The URL at which the server will be running.")
    public String host;

    @Option(name = "port",
            abbrev = 'p',
            defaultValue = "80",
            help = "The port at which the server will be running.")
    public int port;

    @Option(name = "debug",
            abbrev = 'd',
            defaultValue = "false",
            help = "debug")
    public boolean isDebugging;

    @Option(name = "tristate",
        abbrev = 't',
        defaultValue = "auto",
        help = "tri-state option returning auto by default")
    public TriState triState;

    @Option(name = "special",
            defaultValue = "null",
            expansion = { "--host=special.google.com", "--port=8080"})
    public Void special;
  }

  @Test
  public void paragraphFill() throws Exception {
    // TODO(bazel-team): don't include trailing space after last word in line.
    String input = "The quick brown fox jumps over the lazy dog.";

    assertEquals("  The quick \n  brown fox \n  jumps over \n  the lazy \n"
                 + "  dog.",
                 OptionsUsage.paragraphFill(input, 2, 13));
    assertEquals("   The quick brown \n   fox jumps over \n   the lazy dog.",
                 OptionsUsage.paragraphFill(input, 3, 19));

    String input2 = "The quick brown fox jumps\nAnother paragraph.";
    assertEquals("  The quick brown fox \n  jumps\n  Another paragraph.",
                 OptionsUsage.paragraphFill(input2, 2, 23));
  }

  @Test
  public void getsDefaults() throws OptionsParsingException {
    Options<HttpOptions> options = Options.parse(HttpOptions.class, NO_ARGS);
    String[] remainingArgs = options.getRemainingArgs();
    HttpOptions webFlags = options.getOptions();

    assertEquals("www.google.com", webFlags.host);
    assertEquals(80, webFlags.port);
    assertEquals(false, webFlags.isDebugging);
    assertEquals(TriState.AUTO, webFlags.triState);
    assertEquals(0, remainingArgs.length);
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
    assertTrue(toString.startsWith(
                   "com.google.devtools.common.options.OptionsTest"
                   + "$HttpOptions{"));
    assertTrue(toString.contains("host=foo"));
    assertTrue(toString.contains("port=80"));
    assertTrue(toString.endsWith("}"));

    new EqualsTester().addEqualityGroup(left).testEquals();
    assertTrue(left.toString().equals(likeLeft.toString()));
    assertTrue(left.equals(likeLeft));
    assertTrue(likeLeft.equals(left));
    assertFalse(left.equals(right));
    assertFalse(right.equals(left));
    assertFalse(left.equals(null));
    assertFalse(likeLeft.equals(null));
    assertEquals(likeLeft.hashCode(), likeLeft.hashCode());
    assertEquals(left.hashCode(), likeLeft.hashCode());
    // Strictly speaking this is not required for hashCode to be correct,
    // but a good hashCode should be different at least for some values. So,
    // we're making sure that at least this particular pair of inputs yields
    // different values.
    assertFalse(left.hashCode() == right.hashCode());
  }

  @Test
  public void equals() throws OptionsParsingException {
    String[] args = { "--host", "foo", "--port", "80" };
    HttpOptions options1 =  Options.parse(HttpOptions.class, args).getOptions();

    String[] args2 = { "-p", "80", "--host", "foo" };
    HttpOptions options2 =  Options.parse(HttpOptions.class, args2).getOptions();
    assertEquals("order/abbreviations shouldn't matter", options1, options2);

    assertEquals("explicitly setting a default shouldn't matter",
        Options.parse(HttpOptions.class, "--port", "80").getOptions(),
        Options.parse(HttpOptions.class).getOptions());

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

    assertEquals("google.com", webFlags.host);
    assertEquals(8080, webFlags.port);
    assertEquals(true, webFlags.isDebugging);
    assertEquals(0, remainingArgs.length);
  }

  @Test
  public void getsFlagsProvidedWithEquals() throws OptionsParsingException {
    String[] args = {"--host=google.com",
                     "--port=8080",
                     "--debug"};
    Options<HttpOptions> options = Options.parse(HttpOptions.class, args);
    String[] remainingArgs = options.getRemainingArgs();
    HttpOptions webFlags = options.getOptions();

    assertEquals("google.com", webFlags.host);
    assertEquals(8080, webFlags.port);
    assertEquals(true, webFlags.isDebugging);
    assertEquals(0, remainingArgs.length);
  }

  @Test
  public void booleanNo() throws OptionsParsingException {
    Options<HttpOptions> options =
        Options.parse(HttpOptions.class, new String[]{"--nodebug", "--notristate"});
    HttpOptions webFlags = options.getOptions();
    assertEquals(false, webFlags.isDebugging);
    assertEquals(TriState.NO, webFlags.triState);
  }

  @Test
  public void booleanNoUnderscore() throws OptionsParsingException {
    Options<HttpOptions> options =
        Options.parse(HttpOptions.class, new String[]{"--no_debug", "--no_tristate"});
    HttpOptions webFlags = options.getOptions();
    assertEquals(false, webFlags.isDebugging);
    assertEquals(TriState.NO, webFlags.triState);
  }

  @Test
  public void booleanAbbrevMinus() throws OptionsParsingException {
    Options<HttpOptions> options =
        Options.parse(HttpOptions.class, new String[]{"-d-", "-t-"});
    HttpOptions webFlags = options.getOptions();
    assertEquals(false, webFlags.isDebugging);
    assertEquals(TriState.NO, webFlags.triState);
  }

  @Test
  public void boolean0() throws OptionsParsingException {
    Options<HttpOptions> options =
        Options.parse(HttpOptions.class, new String[]{"--debug=0", "--tristate=0"});
    HttpOptions webFlags = options.getOptions();
    assertEquals(false, webFlags.isDebugging);
    assertEquals(TriState.NO, webFlags.triState);
  }

  @Test
  public void boolean1() throws OptionsParsingException {
    Options<HttpOptions> options =
        Options.parse(HttpOptions.class, new String[]{"--debug=1", "--tristate=1"});
    HttpOptions webFlags = options.getOptions();
    assertEquals(true, webFlags.isDebugging);
    assertEquals(TriState.YES, webFlags.triState);
  }

  @Test
  public void retainsStuffThatsNotOptions() throws OptionsParsingException {
    String[] args = {"these", "aint", "options"};
    Options<HttpOptions> options = Options.parse(HttpOptions.class, args);
    String[] remainingArgs = options.getRemainingArgs();
    assertEquals(asList(args), asList(remainingArgs));
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
    assertEquals(asList(notoptions), asList(remainingArgs));
  }

  @Test
  public void wontParseUnknownOptions() {
    String [] args = { "--unknown", "--other=23", "--options" };
    try {
      Options.parse(HttpOptions.class, args);
      fail();
    } catch (OptionsParsingException e) {
      assertEquals("Unrecognized option: --unknown", e.getMessage());
    }
  }

  @Test
  public void requiresOptionValue() {
    String[] args = {"--port"};
    try {
      Options.parse(HttpOptions.class, args);
      fail();
    } catch (OptionsParsingException e) {
      assertEquals("Expected value after --port", e.getMessage());
    }
  }

  @Test
  public void handlesDuplicateOptions_full() throws Exception {
    String[] args = {"--port=80", "--port", "81"};
    Options<HttpOptions> options = Options.parse(HttpOptions.class, args);
    HttpOptions webFlags = options.getOptions();
    assertEquals(81, webFlags.port);
  }

  @Test
  public void handlesDuplicateOptions_abbrev() throws Exception {
    String[] args = {"--port=80", "-p", "81"};
    Options<HttpOptions> options = Options.parse(HttpOptions.class, args);
    HttpOptions webFlags = options.getOptions();
    assertEquals(81, webFlags.port);
  }

  @Test
  public void duplicateOptionsOkWithSameValues() throws Exception {
    // These would throw OptionsParsingException if they failed.
    Options.parse(HttpOptions.class,"--port=80", "--port", "80");
    Options.parse(HttpOptions.class, "--port=80", "-p", "80");
  }

  @Test
  public void isPickyAboutBooleanValues() {
    try {
      Options.parse(HttpOptions.class, new String[]{"--debug=not_a_boolean"});
      fail();
    } catch (OptionsParsingException e) {
      assertEquals("While parsing option --debug=not_a_boolean: "
                   + "\'not_a_boolean\' is not a boolean", e.getMessage());
    }
  }

  @Test
  public void isPickyAboutBooleanNos() {
    try {
      Options.parse(HttpOptions.class, new String[]{"--nodebug=1"});
      fail();
    } catch (OptionsParsingException e) {
      assertEquals("Unexpected value after boolean option: --nodebug=1", e.getMessage());
    }
  }

  @Test
  public void usageForBuiltinTypes() {
    String usage = Options.getUsage(HttpOptions.class);
    // We can't rely on the option ordering.
    assertTrue(usage.contains(
            "  --[no]debug [-d] (a boolean; default: \"false\")\n" +
            "    debug"));
    assertTrue(usage.contains(
            "  --host (a string; default: \"www.google.com\")\n" +
            "    The URL at which the server will be running."));
    assertTrue(usage.contains(
            "  --port [-p] (an integer; default: \"80\")\n" +
            "    The port at which the server will be running."));
    assertTrue(usage.contains(
            "  --special\n" +
            "    Expands to: --host=special.google.com --port=8080"));
    assertTrue(usage.contains(
        "  --[no]tristate [-t] (a tri-state (auto, yes, no); default: \"auto\")\n" +
        "    tri-state option returning auto by default"));
  }

  public static class NullTestOptions extends OptionsBase {
    @Option(name = "host",
            defaultValue = "null",
            help = "The URL at which the server will be running.")
    public String host;

    @Option(name = "none",
        defaultValue = "null",
        expansion = {"--host=www.google.com"},
        help = "An expanded option.")
    public Void none;
  }

  @Test
  public void usageForNullDefault() {
    String usage = Options.getUsage(NullTestOptions.class);
    assertTrue(usage.contains(
            "  --host (a string; default: see description)\n" +
            "    The URL at which the server will be running."));
    assertTrue(usage.contains(
            "  --none\n" +
            "    An expanded option.\n" +
            "    Expands to: --host=www.google.com"));
  }

  public static class MyURLConverter implements Converter<URL> {

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

    @Option(name = "url",
            defaultValue = "http://www.google.com/",
            converter = MyURLConverter.class)
    public URL url;

  }

  @Test
  public void customConverter() throws Exception {
    Options<UsesCustomConverter> options =
      Options.parse(UsesCustomConverter.class, new String[0]);
    URL expected = new URL("http://www.google.com/");
    assertEquals(expected, options.getOptions().url);
  }

  @Test
  public void customConverterThrowsException() throws Exception {
    String[] args = {"--url=a_malformed:url"};
    try {
      Options.parse(UsesCustomConverter.class, args);
      fail();
    } catch (OptionsParsingException e) {
      assertEquals("While parsing option --url=a_malformed:url: "
                   + "Could not convert 'a_malformed:url': "
                   + "no protocol: a_malformed:url", e.getMessage());
    }
  }

  @Test
  public void usageWithCustomConverter() {
    assertEquals(
        "  --url (a url; default: \"http://www.google.com/\")\n",
        Options.getUsage(UsesCustomConverter.class));
  }

  @Test
  public void unknownBooleanOption() {
    try {
      Options.parse(HttpOptions.class, new String[]{"--no-debug"});
      fail();
    } catch (OptionsParsingException e) {
      assertEquals("Unrecognized option: --no-debug", e.getMessage());
    }
  }

  public static class J extends OptionsBase {
    @Option(name = "j", defaultValue = "null")
    public String string;
  }
  @Test
  public void nullDefaultForReferenceTypeOption() throws Exception {
    J options = Options.parse(J.class, NO_ARGS).getOptions();
    assertNull(options.string);
  }

  public static class K extends OptionsBase {
    @Option(name = "1", defaultValue = "null")
    public int int1;
  }
  @Test
  public void nullDefaultForPrimitiveTypeOption() throws Exception {
    // defaultValue() = "null" is not treated specially for primitive types, so
    // we get an NumberFormatException from the converter (not a
    // ClassCastException from casting null to int), just as we would for any
    // other non-integer-literal string default.
    try {
      Options.parse(K.class, NO_ARGS).getOptions();
      fail();
    } catch (IllegalStateException e) {
      assertEquals("OptionsParsingException while retrieving default for "
                   + "int1: 'null' is not an int",
                   e.getMessage());
    }
  }

  @Test
  public void nullIsntInterpretedSpeciallyExceptAsADefaultValue()
      throws Exception {
    HttpOptions options =
        Options.parse(HttpOptions.class,
                      new String[] { "--host", "null" }).getOptions();
    assertEquals("null", options.host);
  }

  @Test
  public void nonDecimalRadicesForIntegerOptions() throws Exception {
    Options<HttpOptions> options =
        Options.parse(HttpOptions.class, new String[] { "--port", "0x51"});
    assertEquals(81, options.getOptions().port);
  }

  @Test
  public void expansionOptionSimple() throws Exception {
    Options<HttpOptions> options =
      Options.parse(HttpOptions.class, new String[] {"--special"});
    assertEquals("special.google.com", options.getOptions().host);
    assertEquals(8080, options.getOptions().port);
  }

  @Test
  public void expansionOptionOverride() throws Exception {
    Options<HttpOptions> options =
      Options.parse(HttpOptions.class, new String[] {"--port=90", "--special", "--host=foo"});
    assertEquals("foo", options.getOptions().host);
    assertEquals(8080, options.getOptions().port);
  }

  @Test
  public void expansionOptionEquals() throws Exception {
    Options<HttpOptions> options1 =
      Options.parse(HttpOptions.class, new String[] { "--host=special.google.com", "--port=8080"});
    Options<HttpOptions> options2 =
      Options.parse(HttpOptions.class, new String[] { "--special" });
    assertEquals(options1.getOptions(), options2.getOptions());
  }
}
