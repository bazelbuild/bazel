// Copyright 2023 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.starlarkdocextract;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import java.util.Optional;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.StarlarkList;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link LabelRenderer}. */
@RunWith(JUnit4.class)
public final class LabelRendererTest {

  @Test
  public void defaultRenderer() throws Exception {
    Label mainRepoLabel = Label.parseCanonicalUnchecked("//foo:bar");
    Label depRepoLabel = Label.parseCanonicalUnchecked("@dep//foo:baz");

    assertThat(LabelRenderer.DEFAULT.render(mainRepoLabel)).isEqualTo("//foo:bar");
    assertThat(LabelRenderer.DEFAULT.reprWithoutLabelConstructor(mainRepoLabel))
        .isEqualTo("\"//foo:bar\"");
    assertThat(LabelRenderer.DEFAULT.repr(mainRepoLabel)).isEqualTo("Label(\"//foo:bar\")");

    assertThat(LabelRenderer.DEFAULT.render(depRepoLabel)).isEqualTo("@@dep//foo:baz");
    assertThat(LabelRenderer.DEFAULT.reprWithoutLabelConstructor(depRepoLabel))
        .isEqualTo("\"@@dep//foo:baz\"");
    assertThat(LabelRenderer.DEFAULT.repr(depRepoLabel)).isEqualTo("Label(\"@@dep//foo:baz\")");
  }

  @Test
  public void mainRepoLabel_withoutMainRepoName() throws Exception {
    Label label = Label.parseCanonicalUnchecked("//foo:bar");
    Label shorthandLabel = Label.parseCanonicalUnchecked("//foo");
    Object dict = Dict.immutableCopyOf(ImmutableMap.of(label, shorthandLabel));

    RepositoryMapping repositoryMapping = RepositoryMapping.EMPTY;
    LabelRenderer labelRenderer = new LabelRenderer(repositoryMapping, Optional.empty());

    assertThat(labelRenderer.render(label)).isEqualTo("//foo:bar");
    assertThat(labelRenderer.render(shorthandLabel)).isEqualTo("//foo");

    assertThat(labelRenderer.reprWithoutLabelConstructor(dict))
        .isEqualTo("{\"//foo:bar\": \"//foo\"}");
    assertThat(labelRenderer.repr(dict)).isEqualTo("{Label(\"//foo:bar\"): Label(\"//foo:foo\")}");
  }

  @Test
  public void mainRepoLabel_withMainRepoName() throws Exception {
    Label label = Label.parseCanonicalUnchecked("//foo:bar");
    Label shorthandLabel = Label.parseCanonicalUnchecked("//foo");
    Label ultraShorthandLabel = Label.parseCanonicalUnchecked("//:my_main");
    Object list =
        StarlarkList.immutableCopyOf(ImmutableList.of(label, shorthandLabel, ultraShorthandLabel));

    RepositoryMapping repositoryMapping = RepositoryMapping.EMPTY;
    LabelRenderer labelRenderer = new LabelRenderer(repositoryMapping, Optional.of("my_main"));

    assertThat(labelRenderer.render(label)).isEqualTo("@my_main//foo:bar");
    assertThat(labelRenderer.render(shorthandLabel)).isEqualTo("@my_main//foo");
    assertThat(labelRenderer.render(ultraShorthandLabel)).isEqualTo("@my_main");

    assertThat(labelRenderer.reprWithoutLabelConstructor(list))
        .isEqualTo("[\"@my_main//foo:bar\", \"@my_main//foo\", \"@my_main\"]");
    assertThat(labelRenderer.repr(list))
        .isEqualTo(
            "[Label(\"@my_main//foo:bar\"), Label(\"@my_main//foo:foo\"),"
                + " Label(\"@my_main//:my_main\")]");
  }

  @Test
  public void remappedRepoLabel() throws Exception {
    Label label = Label.parseCanonicalUnchecked("@canonical//foo:bar");
    Label shorthandLabel = Label.parseCanonicalUnchecked("@canonical//foo");
    Object list = StarlarkList.immutableCopyOf(ImmutableList.of(label, shorthandLabel));

    RepositoryMapping repositoryMapping =
        RepositoryMapping.create(
            ImmutableMap.of("local", RepositoryName.create("canonical")), RepositoryName.MAIN);
    LabelRenderer labelRenderer = new LabelRenderer(repositoryMapping, Optional.of("my_main"));

    assertThat(labelRenderer.render(label)).isEqualTo("@local//foo:bar");
    assertThat(labelRenderer.render(shorthandLabel)).isEqualTo("@local//foo");

    assertThat(labelRenderer.reprWithoutLabelConstructor(list))
        .isEqualTo("[\"@local//foo:bar\", \"@local//foo\"]");
    assertThat(labelRenderer.repr(list))
        .isEqualTo("[Label(\"@local//foo:bar\"), Label(\"@local//foo:foo\")]");
  }
}
