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
package com.google.devtools.build.lib.util;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.util.OptionsUtils.PathFragmentListConverter;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionPriority;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import java.util.Arrays;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Test for {@link OptionsUtils}.
 */
@RunWith(JUnit4.class)
public class OptionsUtilsTest {

  public static class IntrospectionExample extends OptionsBase {
    @Option(name = "alpha",
            category = "one",
            defaultValue = "alpha")
    public String alpha;

    @Option(name = "beta",
            category = "one",
            defaultValue = "beta")
    public String beta;

    @Option(name = "gamma",
            category = "undocumented",
            defaultValue = "gamma")
    public String gamma;

    @Option(name = "delta",
            category = "undocumented",
            defaultValue = "delta")
    public String delta;

    @Option(name = "echo",
            category = "hidden",
            defaultValue = "echo")
    public String echo;
  }

  @Test
  public void asStringOfExplicitOptions() throws Exception {
    OptionsParser parser = OptionsParser.newOptionsParser(IntrospectionExample.class);
    parser.parse("--alpha=no", "--gamma=no", "--echo=no");
    assertEquals("--alpha=no --gamma=no", OptionsUtils.asShellEscapedString(parser));
    assertEquals(ImmutableList.of("--alpha=no", "--gamma=no"), OptionsUtils.asArgumentList(parser));
  }

  @Test
  public void asStringOfExplicitOptionsCorrectSortingByPriority() throws Exception {
    OptionsParser parser = OptionsParser.newOptionsParser(IntrospectionExample.class);
    parser.parse(OptionPriority.COMMAND_LINE, null, Arrays.asList("--alpha=no"));
    parser.parse(OptionPriority.COMPUTED_DEFAULT, null, Arrays.asList("--beta=no"));
    assertEquals("--beta=no --alpha=no", OptionsUtils.asShellEscapedString(parser));
    assertEquals(ImmutableList.of("--beta=no", "--alpha=no"), OptionsUtils.asArgumentList(parser));
  }

  public static class BooleanOpts extends OptionsBase {
    @Option(name = "b_one",
        category = "xyz",
        defaultValue = "true")
    public boolean bOne;

    @Option(name = "b_two",
        category = "123", // Not printed in usage messages!
        defaultValue = "false")
    public boolean bTwo;
  }

  @Test
  public void asStringOfExplicitOptionsWithBooleans() throws Exception {
    OptionsParser parser = OptionsParser.newOptionsParser(BooleanOpts.class);
    parser.parse(OptionPriority.COMMAND_LINE, null, Arrays.asList("--b_one", "--nob_two"));
    assertEquals("--b_one --nob_two", OptionsUtils.asShellEscapedString(parser));
    assertEquals(ImmutableList.of("--b_one", "--nob_two"), OptionsUtils.asArgumentList(parser));

    parser = OptionsParser.newOptionsParser(BooleanOpts.class);
    parser.parse(OptionPriority.COMMAND_LINE, null, Arrays.asList("--b_one=true", "--b_two=0"));
    assertTrue(parser.getOptions(BooleanOpts.class).bOne);
    assertFalse(parser.getOptions(BooleanOpts.class).bTwo);
    assertEquals("--b_one --nob_two", OptionsUtils.asShellEscapedString(parser));
    assertEquals(ImmutableList.of("--b_one", "--nob_two"), OptionsUtils.asArgumentList(parser));
  }

  @Test
  public void asStringOfExplicitOptionsMultipleOptionsAreMultipleTimes() throws Exception {
    OptionsParser parser = OptionsParser.newOptionsParser(IntrospectionExample.class);
    parser.parse(OptionPriority.COMMAND_LINE, null, Arrays.asList("--alpha=one"));
    parser.parse(OptionPriority.COMMAND_LINE, null, Arrays.asList("--alpha=two"));
    assertEquals("--alpha=one --alpha=two", OptionsUtils.asShellEscapedString(parser));
    assertEquals(
        ImmutableList.of("--alpha=one", "--alpha=two"), OptionsUtils.asArgumentList(parser));
  }

  private static List<PathFragment> list(PathFragment... fragments) {
    return Lists.newArrayList(fragments);
  }

  private PathFragment fragment(String string) {
    return new PathFragment(string);
  }

  private List<PathFragment> convert(String input) throws Exception {
    return new PathFragmentListConverter().convert(input);
  }

  @Test
  public void emptyStringYieldsEmptyList() throws Exception {
    assertEquals(list(), convert(""));
  }

  @Test
  public void lonelyDotYieldsLonelyDot() throws Exception {
    assertEquals(list(fragment(".")), convert("."));
  }

  @Test
  public void converterSkipsEmptyStrings() throws Exception {
    assertEquals(list(fragment("foo"), fragment("bar")), convert("foo::bar:"));
  }

  @Test
  public void multiplePaths() throws Exception {
    assertEquals(list(fragment("foo"), fragment("/bar/baz"), fragment("."),
                 fragment("/tmp/bang")), convert("foo:/bar/baz:.:/tmp/bang"));
  }

  @Test
  public void valueisUnmodifiable() throws Exception {
    try {
      new PathFragmentListConverter().convert("value").add(new PathFragment("other"));
      fail("could modify value");
    } catch (UnsupportedOperationException expected) {}
  }
}
