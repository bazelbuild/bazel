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

import com.google.devtools.build.lib.util.OptionsUtils.PathFragmentConverter;
import com.google.devtools.build.lib.util.OptionsUtils.PathFragmentListConverter;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import com.google.devtools.common.options.OptionPriority.PriorityCategory;
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
    @Option(
      name = "alpha",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "alpha"
    )
    public String alpha;

    @Option(
      name = "beta",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "beta"
    )
    public String beta;

    @Option(
      name = "gamma",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "gamma"
    )
    public String gamma;

    @Option(
      name = "delta",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "delta"
    )
    public String delta;

    @Option(
      name = "echo",
      metadataTags = {OptionMetadataTag.HIDDEN},
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.NO_OP},
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
    parser.parse(PriorityCategory.COMMAND_LINE, null, Arrays.asList("--alpha=no"));
    parser.parse(PriorityCategory.COMPUTED_DEFAULT, null, Arrays.asList("--beta=no"));
    assertThat(OptionsUtils.asShellEscapedString(parser)).isEqualTo("--beta=no --alpha=no");
    assertThat(OptionsUtils.asArgumentList(parser))
        .containsExactly("--beta=no", "--alpha=no")
        .inOrder();
  }

  public static class BooleanOpts extends OptionsBase {
    @Option(
      name = "b_one",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "true"
    )
    public boolean bOne;

    @Option(
      name = "b_two",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "false"
    )
    public boolean bTwo;
  }

  @Test
  public void asStringOfExplicitOptionsWithBooleans() throws Exception {
    OptionsParser parser = OptionsParser.newOptionsParser(BooleanOpts.class);
    parser.parse(PriorityCategory.COMMAND_LINE, null, Arrays.asList("--b_one", "--nob_two"));
    assertThat(OptionsUtils.asShellEscapedString(parser)).isEqualTo("--b_one --nob_two");
    assertThat(OptionsUtils.asArgumentList(parser))
        .containsExactly("--b_one", "--nob_two")
        .inOrder();

    parser = OptionsParser.newOptionsParser(BooleanOpts.class);
    parser.parse(PriorityCategory.COMMAND_LINE, null, Arrays.asList("--b_one=true", "--b_two=0"));
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
    parser.parse(PriorityCategory.COMMAND_LINE, null, Arrays.asList("--alpha=one"));
    parser.parse(PriorityCategory.COMMAND_LINE, null, Arrays.asList("--alpha=two"));
    assertThat(OptionsUtils.asShellEscapedString(parser)).isEqualTo("--alpha=one --alpha=two");
    assertThat(OptionsUtils.asArgumentList(parser))
        .containsExactly("--alpha=one", "--alpha=two")
        .inOrder();
  }

  private PathFragment fragment(String string) {
    return PathFragment.create(string);
  }

  private List<PathFragment> convert(String input) throws Exception {
    return new PathFragmentListConverter().convert(input);
  }

  private PathFragment convertOne(String input) throws Exception {
    return new PathFragmentConverter().convert(input);
  }

  @Test
  public void emptyStringYieldsEmptyList() throws Exception {
    assertThat(convert("")).isEmpty();
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
    assertThat(convert("~/foo:foo:/bar/baz:.:/tmp/bang"))
        .containsExactly(
            fragment(System.getProperty("user.home") + "/foo"),
            fragment("foo"),
            fragment("/bar/baz"),
            fragment("."),
            fragment("/tmp/bang"))
        .inOrder();
  }

  @Test
  public void singlePath() throws Exception {
    assertThat(convertOne("foo")).isEqualTo(fragment("foo"));
    assertThat(convertOne("foo/bar/baz")).isEqualTo(fragment("foo/bar/baz"));
    assertThat(convertOne("~/foo")).isEqualTo(fragment(System.getProperty("user.home") + "/foo"));
  }
}
