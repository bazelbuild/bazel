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

package com.google.devtools.build.skydoc.rendering;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import java.util.Optional;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.Starlark;
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

    assertThat(LabelRenderer.DEFAULT.render(mainRepoLabel))
        .isEqualTo(mainRepoLabel.toShorthandString());
    assertThat(LabelRenderer.DEFAULT.reprWithoutLabelConstructor(mainRepoLabel))
        .isEqualTo(Starlark.repr(mainRepoLabel.toShorthandString()));
    assertThat(LabelRenderer.DEFAULT.repr(mainRepoLabel)).isEqualTo(Starlark.repr(mainRepoLabel));

    assertThat(LabelRenderer.DEFAULT.render(mainRepoLabel))
        .isEqualTo(mainRepoLabel.toShorthandString());
    assertThat(LabelRenderer.DEFAULT.reprWithoutLabelConstructor(depRepoLabel))
        .isEqualTo(Starlark.repr(depRepoLabel.toShorthandString()));
    assertThat(LabelRenderer.DEFAULT.repr(depRepoLabel)).isEqualTo(Starlark.repr(depRepoLabel));
  }

  private static void verifyConsistency(
      LabelRenderer labelRenderer, Label label, RepositoryMapping repositoryMapping) {
    String rendering = labelRenderer.render(label);
    Label parsedRenderedLabel = Label.parseCanonicalUnchecked(rendering);

    assertThat(rendering)
        .isEqualTo(
            parsedRenderedLabel.getShorthandDisplayForm(
                // If we are prepending an explicit main repo name, it will not be in the repository
                // mapping, so we need to allow fallback when calling Label#getShorthandDisplayForm.
                RepositoryMapping.createAllowingFallback(repositoryMapping.entries())));
    assertThat(labelRenderer.reprWithoutLabelConstructor(label))
        .isEqualTo(Starlark.repr(rendering));
    assertThat(labelRenderer.repr(label)).isEqualTo(Starlark.repr(parsedRenderedLabel));
  }

  @Test
  public void mainRepoLabel_withoutMainRepoName() throws Exception {
    Label label = Label.parseCanonicalUnchecked("//foo:bar");
    Label shorthandLabel = Label.parseCanonicalUnchecked("//foo");
    Object dict = Dict.immutableCopyOf(ImmutableMap.of(label, shorthandLabel));

    RepositoryMapping repositoryMapping = RepositoryMapping.ALWAYS_FALLBACK;
    LabelRenderer labelRenderer = new LabelRenderer(repositoryMapping, Optional.empty());

    assertThat(labelRenderer.render(label)).isEqualTo("//foo:bar");
    assertThat(labelRenderer.render(shorthandLabel)).isEqualTo("//foo");

    assertThat(labelRenderer.reprWithoutLabelConstructor(dict))
        .isEqualTo("{\"//foo:bar\": \"//foo\"}");
    assertThat(labelRenderer.repr(dict)).isEqualTo("{Label(\"//foo:bar\"): Label(\"//foo:foo\")}");

    verifyConsistency(labelRenderer, label, repositoryMapping);
    verifyConsistency(labelRenderer, shorthandLabel, repositoryMapping);
  }

  @Test
  public void mainRepoLabel_withMainRepoName() throws Exception {
    Label label = Label.parseCanonicalUnchecked("//foo:bar");
    Label shorthandLabel = Label.parseCanonicalUnchecked("//foo");
    Label ultraShorthandLabel = Label.parseCanonicalUnchecked("//:my_main");
    Object list =
        StarlarkList.immutableCopyOf(ImmutableList.of(label, shorthandLabel, ultraShorthandLabel));

    RepositoryMapping repositoryMapping = RepositoryMapping.ALWAYS_FALLBACK;
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

    verifyConsistency(labelRenderer, label, repositoryMapping);
    verifyConsistency(labelRenderer, shorthandLabel, repositoryMapping);
    verifyConsistency(labelRenderer, ultraShorthandLabel, repositoryMapping);
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

    verifyConsistency(labelRenderer, label, repositoryMapping);
    verifyConsistency(labelRenderer, shorthandLabel, repositoryMapping);
  }
}
