// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.rules.ninja;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.bazel.rules.ninja.actions.NinjaPhonyTargetsUtil;
import com.google.devtools.build.lib.bazel.rules.ninja.actions.PhonyTarget;
import com.google.devtools.build.lib.bazel.rules.ninja.file.FileFragment;
import com.google.devtools.build.lib.bazel.rules.ninja.file.GenericParsingException;
import com.google.devtools.build.lib.bazel.rules.ninja.lexer.NinjaLexer;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaParserStep;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaScope;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaTarget;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link com.google.devtools.build.lib.bazel.rules.ninja.actions.NinjaPhonyTargetsUtil}.
 */
@RunWith(JUnit4.class)
public class NinjaPhonyTargetsUtilTest {
  @Test
  public void testPathsTree() throws Exception {
    ImmutableList<String> targetTexts =
        ImmutableList.of(
            "build alias9: phony alias2 alias3 direct1 direct2",
            "build alias2: phony direct3 direct4",
            "build alias3: phony alias4 direct4 direct5",
            "build alias4: phony alias2");

    ImmutableSortedMap<PathFragment, PhonyTarget> pathsMap =
        NinjaPhonyTargetsUtil.getPhonyPathsMap(buildPhonyTargets(targetTexts));

    assertThat(pathsMap).hasSize(4);
    checkMapping(pathsMap, "alias9", "direct1", "direct2", "direct3", "direct4", "direct5");
    checkMapping(pathsMap, "alias2", "direct3", "direct4");
    checkMapping(pathsMap, "alias3", "direct3", "direct4", "direct5");
    checkMapping(pathsMap, "alias4", "direct3", "direct4");
  }

  @Test
  public void testDag() throws Exception {
    ImmutableList<String> targetTexts =
        ImmutableList.of(
            "build _alias9: phony alias1 alias2",
            "build alias1: phony deep1",
            "build alias2: phony deep2",
            "build deep1: phony leaf1",
            "build deep2: phony leaf2 alias1");

    ImmutableSortedMap<PathFragment, PhonyTarget> pathsMap =
        NinjaPhonyTargetsUtil.getPhonyPathsMap(buildPhonyTargets(targetTexts));

    assertThat(pathsMap).hasSize(5);
    checkMapping(pathsMap, "_alias9", "leaf1", "leaf2");
    checkMapping(pathsMap, "alias1", "leaf1");
    checkMapping(pathsMap, "alias2", "leaf1", "leaf2");
    checkMapping(pathsMap, "deep1", "leaf1");
    checkMapping(pathsMap, "deep2", "leaf1", "leaf2");
  }

  @Test
  public void testAlwaysDirty() throws Exception {
    ImmutableList<String> targetTexts =
        ImmutableList.of(
            "build alias9: phony alias2 alias3 direct1 direct2",
            "build alias2: phony direct3 direct4",
            "build alias3: phony alias4 direct4 direct5",
            // alias4 is always-dirty
            "build alias4: phony");

    ImmutableSortedMap<PathFragment, PhonyTarget> pathsMap =
        NinjaPhonyTargetsUtil.getPhonyPathsMap(buildPhonyTargets(targetTexts));

    assertThat(pathsMap).hasSize(4);
    // alias4 and its transitive closure is always-dirty
    checkMapping(pathsMap, "alias9", true, "direct1", "direct2", "direct3", "direct4", "direct5");
    checkMapping(pathsMap, "alias2", false, "direct3", "direct4");
    checkMapping(pathsMap, "alias3", true, "direct4", "direct5");
    checkMapping(pathsMap, "alias4", true);
  }

  private static void checkMapping(
      ImmutableSortedMap<PathFragment, PhonyTarget> pathsMap, String key, String... values) {
    checkMapping(pathsMap, key, false, values);
  }

  private static void checkMapping(
      ImmutableSortedMap<PathFragment, PhonyTarget> pathsMap,
      String key,
      boolean isAlwaysDirty,
      String... values) {
    Set<PathFragment> expectedPaths =
        Arrays.stream(values).map(PathFragment::create).collect(Collectors.toSet());
    PhonyTarget phonyTarget = pathsMap.get(PathFragment.create(key));
    assertThat(phonyTarget).isNotNull();
    assertThat(phonyTarget.isAlwaysDirty()).isEqualTo(isAlwaysDirty);

    ImmutableSortedSet.Builder<PathFragment> paths = ImmutableSortedSet.naturalOrder();
    pathsMap.get(PathFragment.create(key)).visitExplicitInputs(pathsMap, paths::addAll);
    assertThat(paths.build()).containsExactlyElementsIn(expectedPaths);
  }

  private static ImmutableSortedMap<PathFragment, NinjaTarget> buildPhonyTargets(
      ImmutableList<String> targetTexts) throws Exception {
    ImmutableSortedMap.Builder<PathFragment, NinjaTarget> builder =
        ImmutableSortedMap.naturalOrder();
    for (String text : targetTexts) {
      NinjaTarget ninjaTarget = parseNinjaTarget(text);
      builder.put(Iterables.getOnlyElement(ninjaTarget.getAllOutputs()), ninjaTarget);
    }
    return builder.build();
  }

  @Test
  public void testEmptyMap() throws Exception {
    assertThat(NinjaPhonyTargetsUtil.getPhonyPathsMap(ImmutableSortedMap.of())).isEmpty();
  }

  @Test
  public void testCycle() {
    ImmutableList<String> targetTexts =
        ImmutableList.of(
            "build alias1: phony alias2 direct1", "build alias2: phony alias1 direct2");

    GenericParsingException exception =
        assertThrows(
            GenericParsingException.class,
            () -> NinjaPhonyTargetsUtil.getPhonyPathsMap(buildPhonyTargets(targetTexts)));
    assertThat(exception)
        .hasMessageThat()
        .isEqualTo("Detected a dependency cycle involving the phony target 'alias1'");
  }

  private static NinjaTarget parseNinjaTarget(String text) throws Exception {
    NinjaScope fileScope = new NinjaScope();
    return createParser(text).parseNinjaTarget(fileScope, 0);
  }

  private static NinjaParserStep createParser(String text) {
    ByteBuffer buffer = ByteBuffer.wrap(text.getBytes(StandardCharsets.ISO_8859_1));
    NinjaLexer lexer = new NinjaLexer(new FileFragment(buffer, 0, 0, buffer.limit()));
    return new NinjaParserStep(
        lexer, BlazeInterners.newWeakInterner(), BlazeInterners.newWeakInterner());
  }
}
