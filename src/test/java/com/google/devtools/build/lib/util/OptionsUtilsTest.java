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

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import com.google.common.collect.Lists;
import com.google.devtools.build.lib.util.OptionsUtils.PathFragmentListConverter;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionPriority;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParser.OptionUsageRestrictions;
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

    @Option(
      name = "gamma",
      optionUsageRestrictions = OptionUsageRestrictions.UNDOCUMENTED,
      defaultValue = "gamma"
    )
    public String gamma;

    @Option(
      name = "delta",
      optionUsageRestrictions = OptionUsageRestrictions.UNDOCUMENTED,
      defaultValue = "delta"
    )
    public String delta;

    @Option(
      name = "echo",
      optionUsageRestrictions = OptionUsageRestrictions.HIDDEN,
      defaultValue = "echo"
    )
    public String echo;
  }

  @Test
  public void asStringOfExplicitOptions() throws Exception {
    OptionsParser parser = OptionsParser.newOptionsParser(IntrospectionExample.class);
    parser.parse("--alpha=no", "--gamma=no", "--echo=no");
    assertThat(OptionsUtils.asShellEscapedString(parser)).isEqualTo("--alpha=no --gamma=no");
    assertThat(OptionsUtils.asArgumentList(parser))
        .containsExactly("--alpha=no", "--gamma=no")
        .inOrder();
  }

  @Test
  public void asStringOfExplicitOptionsCorrectSortingByPriority() throws Exception {
    OptionsParser parser = OptionsParser.newOptionsParser(IntrospectionExample.class);
    parser.parse(OptionPriority.COMMAND_LINE, null, Arrays.asList("--alpha=no"));
    parser.parse(OptionPriority.COMPUTED_DEFAULT, null, Arrays.asList("--beta=no"));
    assertThat(OptionsUtils.asShellEscapedString(parser)).isEqualTo("--beta=no --alpha=no");
    assertThat(OptionsUtils.asArgumentList(parser))
        .containsExactly("--beta=no", "--alpha=no")
        .inOrder();
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
    assertThat(OptionsUtils.asShellEscapedString(parser)).isEqualTo("--b_one --nob_two");
    assertThat(OptionsUtils.asArgumentList(parser))
        .containsExactly("--b_one", "--nob_two")
        .inOrder();

    parser = OptionsParser.newOptionsParser(BooleanOpts.class);
    parser.parse(OptionPriority.COMMAND_LINE, null, Arrays.asList("--b_one=true", "--b_two=0"));
    assertThat(parser.getOptions(BooleanOpts.class).bOne).isTrue();
    assertThat(parser.getOptions(BooleanOpts.class).bTwo).isFalse();
    assertThat(OptionsUtils.asShellEscapedString(parser)).isEqualTo("--b_one --nob_two");
    assertThat(OptionsUtils.asArgumentList(parser))
        .containsExactly("--b_one", "--nob_two")
        .inOrder();
  }

  @Test
  public void asStringOfExplicitOptionsMultipleOptionsAreMultipleTimes() throws Exception {
    OptionsParser parser = OptionsParser.newOptionsParser(IntrospectionExample.class);
    parser.parse(OptionPriority.COMMAND_LINE, null, Arrays.asList("--alpha=one"));
    parser.parse(OptionPriority.COMMAND_LINE, null, Arrays.asList("--alpha=two"));
    assertThat(OptionsUtils.asShellEscapedString(parser)).isEqualTo("--alpha=one --alpha=two");
    assertThat(OptionsUtils.asArgumentList(parser))
        .containsExactly("--alpha=one", "--alpha=two")
        .inOrder();
  }

  private static List<PathFragment> list(PathFragment... fragments) {
    return Lists.newArrayList(fragments);
  }

  private PathFragment fragment(String string) {
    return PathFragment.create(string);
  }

  private List<PathFragment> convert(String input) throws Exception {
    return new PathFragmentListConverter().convert(input);
  }

  @Test
  public void emptyStringYieldsEmptyList() throws Exception {
    assertThat(convert("")).isEqualTo(list());
  }

  @Test
  public void lonelyDotYieldsLonelyDot() throws Exception {
    assertThat(convert(".")).containsExactly(fragment("."));
  }

  @Test
  public void converterSkipsEmptyStrings() throws Exception {
    assertThat(convert("foo::bar:")).containsExactly(fragment("foo"), fragment("bar")).inOrder();
  }

  @Test
  public void multiplePaths() throws Exception {
    assertThat(convert("foo:/bar/baz:.:/tmp/bang"))
        .containsExactly(
            fragment("foo"), fragment("/bar/baz"), fragment("."), fragment("/tmp/bang"))
        .inOrder();
  }

  @Test
  public void valueisUnmodifiable() throws Exception {
    try {
      new PathFragmentListConverter().convert("value").add(PathFragment.create("other"));
      fail("could modify value");
    } catch (UnsupportedOperationException expected) {}
  }
}
