// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link PlatformMappingFunction}. */
@RunWith(JUnit4.class)
public class PlatformMappingFunctionParserTest {

  private static final Label PLATFORM1 = Label.parseAbsoluteUnchecked("//platforms:one");
  private static final Label PLATFORM2 = Label.parseAbsoluteUnchecked("//platforms:two");

  @Test
  public void testParse() throws Exception {
    PlatformMappingFunction.Mappings mappings =
        parse(
            "platforms:",
            "  //platforms:one",
            "    --cpu=one",
            "  //platforms:two",
            "    --cpu=two",
            "flags:",
            "  --cpu=one",
            "    //platforms:one",
            "  --cpu=two",
            "    //platforms:two");

    assertThat(mappings.platformsToFlags.keySet()).containsExactly(PLATFORM1, PLATFORM2);
    assertThat(mappings.platformsToFlags.get(PLATFORM1)).containsExactly("--cpu=one");
    assertThat(mappings.platformsToFlags.get(PLATFORM2)).containsExactly("--cpu=two");

    assertThat(mappings.flagsToPlatforms.keySet())
        .containsExactly(ImmutableSet.of("--cpu=one"), ImmutableSet.of("--cpu=two"));
    assertThat(mappings.flagsToPlatforms.get(ImmutableSet.of("--cpu=one"))).isEqualTo(PLATFORM1);
    assertThat(mappings.flagsToPlatforms.get(ImmutableSet.of("--cpu=two"))).isEqualTo(PLATFORM2);
  }

  @Test
  public void testParseComment() throws Exception {
    PlatformMappingFunction.Mappings mappings =
        parse(
            "# A mapping file!",
            "platforms:",
            "  # comment1",
            "  //platforms:one",
            "# comment2",
            "    --cpu=one",
            "  //platforms:two",
            "    --cpu=two",
            "flags:",
            "# another comment",
            "  --cpu=one",
            "    //platforms:one",
            "  --cpu=two",
            "    //platforms:two");

    assertThat(mappings.platformsToFlags.keySet()).containsExactly(PLATFORM1, PLATFORM2);
    assertThat(mappings.platformsToFlags.get(PLATFORM1)).containsExactly("--cpu=one");
    assertThat(mappings.platformsToFlags.get(PLATFORM2)).containsExactly("--cpu=two");

    assertThat(mappings.flagsToPlatforms.keySet())
        .containsExactly(ImmutableSet.of("--cpu=one"), ImmutableSet.of("--cpu=two"));
    assertThat(mappings.flagsToPlatforms.get(ImmutableSet.of("--cpu=one"))).isEqualTo(PLATFORM1);
    assertThat(mappings.flagsToPlatforms.get(ImmutableSet.of("--cpu=two"))).isEqualTo(PLATFORM2);
  }

  @Test
  public void testParseWhitespace() throws Exception {
    PlatformMappingFunction.Mappings mappings =
        parse(
            "",
            "platforms:",
            "  ",
            "  //platforms:one",
            "",
            "    --cpu=one",
            "    //platforms:two    ",
            "      --cpu=two ",
            "flags:",
            "           ",
            "",
            "--cpu=one",
            "  //platforms:one",
            "  --cpu=two",
            "  //platforms:two");

    assertThat(mappings.platformsToFlags.keySet()).containsExactly(PLATFORM1, PLATFORM2);
    assertThat(mappings.platformsToFlags.get(PLATFORM1)).containsExactly("--cpu=one");
    assertThat(mappings.platformsToFlags.get(PLATFORM2)).containsExactly("--cpu=two");

    assertThat(mappings.flagsToPlatforms.keySet())
        .containsExactly(ImmutableSet.of("--cpu=one"), ImmutableSet.of("--cpu=two"));
    assertThat(mappings.flagsToPlatforms.get(ImmutableSet.of("--cpu=one"))).isEqualTo(PLATFORM1);
    assertThat(mappings.flagsToPlatforms.get(ImmutableSet.of("--cpu=two"))).isEqualTo(PLATFORM2);
  }

  @Test
  public void testParseMultipleFlagsInPlatform() throws Exception {
    PlatformMappingFunction.Mappings mappings =
        parse(
            "platforms:",
            "  //platforms:one",
            "    --cpu=one",
            "    --compilation_mode=dbg",
            "  //platforms:two",
            "    --cpu=two");

    assertThat(mappings.platformsToFlags.keySet()).containsExactly(PLATFORM1, PLATFORM2);
    assertThat(mappings.platformsToFlags.get(PLATFORM1))
        .containsExactly("--cpu=one", "--compilation_mode=dbg");
  }

  @Test
  public void testParseMultipleFlagsInFlags() throws Exception {
    PlatformMappingFunction.Mappings mappings =
        parse(
            "flags:",
            "  --compilation_mode=dbg",
            "  --cpu=one",
            "    //platforms:one",
            "  --cpu=two",
            "    //platforms:two");

    assertThat(mappings.flagsToPlatforms.keySet())
        .containsExactly(
            ImmutableSet.of("--cpu=one", "--compilation_mode=dbg"), ImmutableSet.of("--cpu=two"));
    assertThat(
            mappings.flagsToPlatforms.get(ImmutableSet.of("--cpu=one", "--compilation_mode=dbg")))
        .isEqualTo(PLATFORM1);
  }

  @Test
  public void testParseOnlyPlatforms() throws Exception {
    PlatformMappingFunction.Mappings mappings =
        parse(
            "platforms:", // Force line break
            "  //platforms:one", // Force line break
            "    --cpu=one" // Force line break
            );

    assertThat(mappings.platformsToFlags.keySet()).containsExactly(PLATFORM1);
    assertThat(mappings.platformsToFlags.get(PLATFORM1)).containsExactly("--cpu=one");
    assertThat(mappings.flagsToPlatforms).isEmpty();
  }

  @Test
  public void testParseOnlyFlags() throws Exception {
    PlatformMappingFunction.Mappings mappings =
        parse(
            "flags:", // Force line break
            "  --cpu=one", // Force line break
            "    //platforms:one" // Force line break
            );

    assertThat(mappings.flagsToPlatforms.keySet()).containsExactly(ImmutableSet.of("--cpu=one"));
    assertThat(mappings.flagsToPlatforms.get(ImmutableSet.of("--cpu=one"))).isEqualTo(PLATFORM1);
    assertThat(mappings.platformsToFlags).isEmpty();
  }

  @Test
  public void testParseEmpty() throws Exception {
    PlatformMappingFunction.Mappings mappings = parse();

    assertThat(mappings.flagsToPlatforms).isEmpty();
    assertThat(mappings.platformsToFlags).isEmpty();
  }

  @Test
  public void testParseEmptySections() throws Exception {
    PlatformMappingFunction.Mappings mappings = parse("platforms:", "flags:");

    assertThat(mappings.flagsToPlatforms).isEmpty();
    assertThat(mappings.platformsToFlags).isEmpty();
  }

  @Test
  public void testParseCommentOnly() throws Exception {
    PlatformMappingFunction.Mappings mappings = parse("#No mappings");

    assertThat(mappings.flagsToPlatforms).isEmpty();
    assertThat(mappings.platformsToFlags).isEmpty();
  }

  @Test
  public void testParseExtraPlatformInFlags() throws Exception {
    PlatformMappingFunction.PlatformMappingException exception =
        assertThrows(
            PlatformMappingFunction.PlatformMappingException.class,
            () ->
                parse(
                    "flags:", // Force line break
                    "  --cpu=one", // Force line break
                    "    //platforms:one", // Force line break
                    "    //platforms:two" // Force line break
                    ));

    assertThat(exception).hasMessageThat().contains("//platforms:two");
  }

  @Test
  public void testParsePlatformWithoutFlags() throws Exception {
    PlatformMappingFunction.PlatformMappingException exception =
        assertThrows(
            PlatformMappingFunction.PlatformMappingException.class,
            () ->
                parse(
                    "platforms:", // Force line break
                    "  //platforms:one" // Force line break
                    ));

    assertThat(exception).hasMessageThat().contains("end of file");
  }

  @Test
  public void testParseFlagsWithoutPlatform() throws Exception {
    PlatformMappingFunction.PlatformMappingException exception =
        assertThrows(
            PlatformMappingFunction.PlatformMappingException.class,
            () ->
                parse(
                    "flags:", // Force line break
                    "  --cpu=one" // Force line break
                    ));

    assertThat(exception).hasMessageThat().contains("end of file");
  }

  @Test
  public void testParseCommentEndOfFile() throws Exception {
    PlatformMappingFunction.Mappings mappings =
        parse(
            "platforms:", // Force line break
            "  //platforms:one", // Force line break
            "    --cpu=one", // Force line break
            "# No more mappings" // Force line break
            );

    assertThat(mappings.platformsToFlags).isNotEmpty();
  }

  @Test
  public void testParseUnknownSection() throws Exception {
    PlatformMappingFunction.PlatformMappingException exception =
        assertThrows(
            PlatformMappingFunction.PlatformMappingException.class,
            () ->
                parse(
                    "platform:", // Force line break
                    "  //platforms:one", // Force line break
                    "    --cpu=one" // Force line break
                    ));

    assertThat(exception).hasMessageThat().contains("platform:");

    assertThrows(
        PlatformMappingFunction.PlatformMappingException.class,
        () ->
            parse(
                "platforms:",
                "  //platforms:one",
                "    --cpu=one",
                "flag:",
                "  --cpu=one",
                "    //platforms:one"));

    assertThat(exception).hasMessageThat().contains("platform");
  }

  @Test
  public void testParsePlatformsInvalidPlatformLabel() throws Exception {
    PlatformMappingFunction.PlatformMappingException exception =
        assertThrows(
            PlatformMappingFunction.PlatformMappingException.class,
            () ->
                parse(
                    "platforms:", // Force line break
                    "  @@@", // Force line break
                    "    --cpu=one"));

    assertThat(exception).hasMessageThat().contains("@@@");
    assertThat(exception).hasCauseThat().hasCauseThat().isInstanceOf(LabelSyntaxException.class);
  }

  @Test
  public void testParseFlagsInvalidPlatformLabel() throws Exception {
    PlatformMappingFunction.PlatformMappingException exception =
        assertThrows(
            PlatformMappingFunction.PlatformMappingException.class,
            () ->
                parse(
                    "flags:", // Force line break
                    "  --cpu=one", // Force line break
                    "    @@@"));

    assertThat(exception).hasMessageThat().contains("@@@");
    assertThat(exception).hasCauseThat().hasCauseThat().isInstanceOf(LabelSyntaxException.class);
  }

  @Test
  public void testParsePlatformsInvalidFlag() throws Exception {
    PlatformMappingFunction.PlatformMappingException exception =
        assertThrows(
            PlatformMappingFunction.PlatformMappingException.class,
            () ->
                parse(
                    "platforms:", // Force line break
                    "  //platforms:one", // Force line break
                    "    -cpu=one"));

    assertThat(exception).hasMessageThat().contains("-cpu");
  }

  @Test
  public void testParseFlagsInvalidFlag() throws Exception {
    PlatformMappingFunction.PlatformMappingException exception =
        assertThrows(
            PlatformMappingFunction.PlatformMappingException.class,
            () ->
                parse(
                    "flags:", // Force line break
                    "  -cpu=one", // Force line break
                    "    //platforms:one"));

    assertThat(exception).hasMessageThat().contains("-cpu");
  }

  @Test
  public void testParsePlatformsDuplicatePlatform() throws Exception {
    PlatformMappingFunction.PlatformMappingException exception =
        assertThrows(
            PlatformMappingFunction.PlatformMappingException.class,
            () ->
                parse(
                    "platforms:", // Force line break
                    "  //platforms:one", // Force line break
                    "    --cpu=one", // Force line break
                    "  //platforms:one", // Force line break
                    "    --cpu=two"));

    assertThat(exception).hasMessageThat().contains("duplicate");
    assertThat(exception)
        .hasCauseThat()
        .hasCauseThat()
        .hasMessageThat()
        .contains("//platforms:one");
  }

  @Test
  public void testParseFlagsDuplicateFlags() throws Exception {
    PlatformMappingFunction.PlatformMappingException exception =
        assertThrows(
            PlatformMappingFunction.PlatformMappingException.class,
            () ->
                parse(
                    "flags:", // Force line break
                    "  --compilation_mode=dbg", // Force line break
                    "  --cpu=one", // Force line break
                    "    //platforms:one", // Force line break
                    "  --compilation_mode=dbg", // Force line break
                    "  --cpu=one", // Force line break
                    "    //platforms:two"));

    assertThat(exception).hasMessageThat().contains("duplicate");
    assertThat(exception).hasCauseThat().hasCauseThat().hasMessageThat().contains("--cpu=one");
    assertThat(exception)
        .hasCauseThat()
        .hasCauseThat()
        .hasMessageThat()
        .contains("--compilation_mode=dbg");
  }

  private static PlatformMappingFunction.Mappings parse(String... lines)
      throws PlatformMappingFunction.PlatformMappingException {
    return new PlatformMappingFunction.Parser(ImmutableList.copyOf(lines).iterator()).parse();
  }
}
