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
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.bazel.rules.ninja.actions.NinjaPhonyTargetsUtil;
import com.google.devtools.build.lib.bazel.rules.ninja.file.ByteBufferFragment;
import com.google.devtools.build.lib.bazel.rules.ninja.file.GenericParsingException;
import com.google.devtools.build.lib.bazel.rules.ninja.lexer.NinjaLexer;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaParserStep;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaScope;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaTarget;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link com.google.devtools.build.lib.bazel.rules.ninja.actions.NinjaPhonyTargetsUtil}. */
@RunWith(JUnit4.class)
public class NinjaPhonyTargetsUtilTest {
  @Test
  public void testPathsTree() throws Exception {
    ImmutableList<String> targetTexts = ImmutableList.of(
        "build alias9: phony alias2 alias3 direct1 direct2",
        "build alias2: phony direct3 direct4",
        "build alias3: phony alias4 direct4 direct5",
        "build alias4: phony alias2");

    ImmutableSortedMap<PathFragment, NestedSet<PathFragment>> pathsMap =
        new NinjaPhonyTargetsUtil(buildPhonyTargets(targetTexts)).getPhonyPathsMap();

    assertThat(pathsMap).hasSize(4);
    assertThat(pathsMap.get(PathFragment.create("alias9")).toSet())
        .containsExactly(PathFragment.create("direct1"), PathFragment.create("direct2"),
            PathFragment.create("direct3"), PathFragment.create("direct4"),
            PathFragment.create("direct5"));
    assertThat(pathsMap.get(PathFragment.create("alias2")).toSet())
        .containsExactly(PathFragment.create("direct3"), PathFragment.create("direct4"));
    assertThat(pathsMap.get(PathFragment.create("alias3")).toSet())
        .containsExactly(PathFragment.create("direct3"), PathFragment.create("direct4"),
            PathFragment.create("direct5"));
    assertThat(pathsMap.get(PathFragment.create("alias4")).toSet())
        .containsExactly(PathFragment.create("direct3"), PathFragment.create("direct4"));
  }

  private ImmutableSortedMap<PathFragment, NinjaTarget> buildPhonyTargets(
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
    assertThat(new NinjaPhonyTargetsUtil(ImmutableSortedMap.of()).getPhonyPathsMap()).isEmpty();
  }

  @Test
  public void testCycle() {
    ImmutableList<String> targetTexts = ImmutableList.of(
        "build alias1: phony alias2 direct1",
        "build alias2: phony alias1 direct2"
        );

    GenericParsingException exception = assertThrows(GenericParsingException.class,
        () -> new NinjaPhonyTargetsUtil(buildPhonyTargets(targetTexts)).getPhonyPathsMap());
    assertThat(exception).hasMessageThat().isEqualTo(
        "Detected a dependency cycle involving the phony target 'alias1'"
    );
  }

  private static NinjaTarget parseNinjaTarget(String text) throws Exception {
    NinjaScope fileScope = new NinjaScope();
    return createParser(text).parseNinjaTarget(fileScope, 0);
  }

  private static NinjaParserStep createParser(String text) {
    ByteBuffer buffer = ByteBuffer.wrap(text.getBytes(StandardCharsets.ISO_8859_1));
    NinjaLexer lexer = new NinjaLexer(new ByteBufferFragment(buffer, 0, buffer.limit()));
    return new NinjaParserStep(lexer);
  }
}
