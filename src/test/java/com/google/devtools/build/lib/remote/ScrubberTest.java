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

package com.google.devtools.build.lib.remote;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.exec.util.SpawnBuilder;
import com.google.devtools.build.lib.remote.RemoteScrubbing.Config;
import com.google.devtools.build.lib.remote.Scrubber.SpawnScrubber;
import com.google.devtools.build.lib.vfs.PathFragment;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link Scrubber}. */
@RunWith(JUnit4.class)
public class ScrubberTest {

  @Test
  public void noScrubbing() {
    var scrubber = new Scrubber(Config.getDefaultInstance());

    assertThat(scrubber.forSpawn(createSpawn("//foo:bar", "Foo"))).isNull();
  }

  @Test
  public void matchExactMnemonic() {
    var scrubber =
        new Scrubber(
            Config.newBuilder()
                .addRules(
                    Config.Rule.newBuilder()
                        .setMatcher(Config.Matcher.newBuilder().setMnemonic("Foo")))
                .build());

    assertThat(scrubber.forSpawn(createSpawn("//foo:bar", "Foo"))).isNotNull();
    assertThat(scrubber.forSpawn(createSpawn("//foo:bar", "Foobar"))).isNull();
  }

  @Test
  public void matchUnionMnemonic() {
    var scrubber =
        new Scrubber(
            Config.newBuilder()
                .addRules(
                    Config.Rule.newBuilder()
                        .setMatcher(Config.Matcher.newBuilder().setMnemonic("Foo|Bar")))
                .build());

    assertThat(scrubber.forSpawn(createSpawn("//foo:bar", "Foo"))).isNotNull();
    assertThat(scrubber.forSpawn(createSpawn("//foo:bar", "Bar"))).isNotNull();
    assertThat(scrubber.forSpawn(createSpawn("//foo:bar", "Baz"))).isNull();
  }

  @Test
  public void matchWildcardMnemonic() {
    var scrubber =
        new Scrubber(
            Config.newBuilder()
                .addRules(
                    Config.Rule.newBuilder()
                        .setMatcher(Config.Matcher.newBuilder().setMnemonic("Foo.*")))
                .build());

    assertThat(scrubber.forSpawn(createSpawn("//foo:bar", "Foo"))).isNotNull();
    assertThat(scrubber.forSpawn(createSpawn("//foo:bar", "Foobar"))).isNotNull();
    assertThat(scrubber.forSpawn(createSpawn("//foo:bar", "Bar"))).isNull();
  }

  @Test
  public void matchExactLabel() {
    var scrubber =
        new Scrubber(
            Config.newBuilder()
                .addRules(
                    Config.Rule.newBuilder()
                        .setMatcher(Config.Matcher.newBuilder().setLabel("//foo:bar")))
                .build());

    assertThat(scrubber.forSpawn(createSpawn("//foo:bar", "Foo"))).isNotNull();
    assertThat(scrubber.forSpawn(createSpawn("//foo:barbaz", "Foo"))).isNull();
  }

  @Test
  public void matchUnionLabel() {
    var scrubber =
        new Scrubber(
            Config.newBuilder()
                .addRules(
                    Config.Rule.newBuilder()
                        .setMatcher(Config.Matcher.newBuilder().setLabel("//foo:bar|//spam:eggs")))
                .build());

    assertThat(scrubber.forSpawn(createSpawn("//foo:bar", "Foo"))).isNotNull();
    assertThat(scrubber.forSpawn(createSpawn("//spam:eggs", "Foo"))).isNotNull();
    assertThat(scrubber.forSpawn(createSpawn("//quux:xyzzy", "Foo"))).isNull();
  }

  @Test
  public void matchWildcardLabel() {
    var scrubber =
        new Scrubber(
            Config.newBuilder()
                .addRules(
                    Config.Rule.newBuilder()
                        .setMatcher(Config.Matcher.newBuilder().setLabel("//foo:.*")))
                .build());

    assertThat(scrubber.forSpawn(createSpawn("//foo:bar", "Foo"))).isNotNull();
    assertThat(scrubber.forSpawn(createSpawn("//foo:baz", "Foo"))).isNotNull();
    assertThat(scrubber.forSpawn(createSpawn("//spam:eggs", "Foo"))).isNull();
  }

  @Test
  public void matchExactKind() {
    var scrubber =
        new Scrubber(
            Config.newBuilder()
                .addRules(
                    Config.Rule.newBuilder()
                        .setMatcher(Config.Matcher.newBuilder().setKind("java_library")))
                .build());

    assertThat(scrubber.forSpawn(createSpawn("//foo:bar", "Foo", "java_library", false)))
        .isNotNull();
    assertThat(scrubber.forSpawn(createSpawn("//foo:barbaz", "Foo", "java_test", false))).isNull();
  }

  @Test
  public void matchUnionKind() {
    var scrubber =
        new Scrubber(
            Config.newBuilder()
                .addRules(
                    Config.Rule.newBuilder()
                        .setMatcher(Config.Matcher.newBuilder().setKind("java_library|java_test")))
                .build());

    assertThat(scrubber.forSpawn(createSpawn("//foo:bar", "Foo", "java_library", false)))
        .isNotNull();
    assertThat(scrubber.forSpawn(createSpawn("//spam:eggs", "Foo", "java_test", false)))
        .isNotNull();
    assertThat(scrubber.forSpawn(createSpawn("//quux:xyzzy", "Foo", "go_library", false))).isNull();
  }

  @Test
  public void matchWildcardKind() {
    var scrubber =
        new Scrubber(
            Config.newBuilder()
                .addRules(
                    Config.Rule.newBuilder()
                        .setMatcher(Config.Matcher.newBuilder().setKind("java_.*")))
                .build());

    assertThat(scrubber.forSpawn(createSpawn("//foo:bar", "Foo", "java_library", false)))
        .isNotNull();
    assertThat(scrubber.forSpawn(createSpawn("//foo:baz", "Foo", "java_test", false))).isNotNull();
    assertThat(scrubber.forSpawn(createSpawn("//spam:eggs", "Foo", "go_library", false))).isNull();
  }

  @Test
  public void rejectToolAction() {
    var scrubber =
        new Scrubber(
            Config.newBuilder()
                .addRules(
                    Config.Rule.newBuilder()
                        .setMatcher(
                            Config.Matcher.newBuilder().setLabel("//foo:bar").setMnemonic("Foo")))
                .build());

    assertThat(scrubber.forSpawn(createSpawn("//foo:bar", "Foo"))).isNotNull();
    assertThat(
            scrubber.forSpawn(createSpawn("//foo:bar", "Foo", "java_library", /* forTool= */ true)))
        .isNull();
  }

  @Test
  public void acceptToolAction() {
    var scrubber =
        new Scrubber(
            Config.newBuilder()
                .addRules(
                    Config.Rule.newBuilder()
                        .setMatcher(
                            Config.Matcher.newBuilder()
                                .setLabel("//foo:bar")
                                .setMnemonic("Foo")
                                .setMatchTools(true)))
                .build());

    assertThat(scrubber.forSpawn(createSpawn("//foo:bar", "Foo"))).isNotNull();
    assertThat(
            scrubber.forSpawn(createSpawn("//foo:bar", "Foo", "java_library", /* forTool= */ true)))
        .isNotNull();
  }

  @Test
  public void noOmittedInputs() {
    var spawnScrubber =
        new Scrubber(Config.newBuilder().addRules(Config.Rule.getDefaultInstance()).build())
            .forSpawn(createSpawn());

    assertThat(spawnScrubber.shouldOmitInput(PathFragment.create("foo/bar"))).isFalse();
  }

  @Test
  public void exactOmittedInput() {
    var spawnScrubber =
        new Scrubber(
                Config.newBuilder()
                    .addRules(
                        Config.Rule.newBuilder()
                            .setTransform(
                                Config.Transform.newBuilder().addOmittedInputs("foo/bar")))
                    .build())
            .forSpawn(createSpawn());

    assertThat(spawnScrubber.shouldOmitInput(PathFragment.create("foo/bar"))).isTrue();
    assertThat(spawnScrubber.shouldOmitInput(PathFragment.create("foo/bar/baz"))).isFalse();
    assertThat(spawnScrubber.shouldOmitInput(PathFragment.create("bazel-out/foo/bar"))).isFalse();
  }

  @Test
  public void wildcardOmittedInput() {
    var spawnScrubber =
        new Scrubber(
                Config.newBuilder()
                    .addRules(
                        Config.Rule.newBuilder()
                            .setTransform(
                                Config.Transform.newBuilder().addOmittedInputs("foo/bar.*")))
                    .build())
            .forSpawn(createSpawn());

    assertThat(spawnScrubber.shouldOmitInput(PathFragment.create("foo/bar"))).isTrue();
    assertThat(spawnScrubber.shouldOmitInput(PathFragment.create("foo/bar/baz"))).isTrue();
    assertThat(spawnScrubber.shouldOmitInput(PathFragment.create("bazel-out/foo/bar"))).isFalse();
  }

  @Test
  public void multipleOmittedInputs() {
    var spawnScrubber =
        new Scrubber(
                Config.newBuilder()
                    .addRules(
                        Config.Rule.newBuilder()
                            .setTransform(
                                Config.Transform.newBuilder()
                                    .addOmittedInputs("foo/bar")
                                    .addOmittedInputs("spam/eggs")))
                    .build())
            .forSpawn(createSpawn());

    assertThat(spawnScrubber.shouldOmitInput(PathFragment.create("foo/bar"))).isTrue();
    assertThat(spawnScrubber.shouldOmitInput(PathFragment.create("spam/eggs"))).isTrue();
    assertThat(spawnScrubber.shouldOmitInput(PathFragment.create("foo/bar/baz"))).isFalse();
    assertThat(spawnScrubber.shouldOmitInput(PathFragment.create("bazel-out/foo/bar"))).isFalse();
    assertThat(spawnScrubber.shouldOmitInput(PathFragment.create("spam/eggs/bacon"))).isFalse();
    assertThat(spawnScrubber.shouldOmitInput(PathFragment.create("bazel-out/spam/eggs"))).isFalse();
  }

  @Test
  public void simpleArgReplacement() {
    var spawnScrubber =
        new Scrubber(
                Config.newBuilder()
                    .addRules(
                        Config.Rule.newBuilder()
                            .setTransform(
                                Config.Transform.newBuilder()
                                    .addArgReplacements(
                                        Config.Replacement.newBuilder()
                                            .setSource("foo")
                                            .setTarget("bar"))))
                    .build())
            .forSpawn(createSpawn());

    assertThat(spawnScrubber.transformArgument("foo")).isEqualTo("bar");
    assertThat(spawnScrubber.transformArgument("abcfooxyz")).isEqualTo("abcbarxyz");
    assertThat(spawnScrubber.transformArgument("bar")).isEqualTo("bar");
    assertThat(spawnScrubber.transformArgument("foofoo")).isEqualTo("barfoo");
  }

  @Test
  public void anchoredArgReplacement() {
    var spawnScrubber =
        new Scrubber(
                Config.newBuilder()
                    .addRules(
                        Config.Rule.newBuilder()
                            .setTransform(
                                Config.Transform.newBuilder()
                                    .addArgReplacements(
                                        Config.Replacement.newBuilder()
                                            .setSource("^foo$")
                                            .setTarget("bar"))))
                    .build())
            .forSpawn(createSpawn());

    assertThat(spawnScrubber.transformArgument("foo")).isEqualTo("bar");
    assertThat(spawnScrubber.transformArgument("abcfooxyz")).isEqualTo("abcfooxyz");
  }

  @Test
  public void wildcardArgReplacement() {
    var spawnScrubber =
        new Scrubber(
                Config.newBuilder()
                    .addRules(
                        Config.Rule.newBuilder()
                            .setTransform(
                                Config.Transform.newBuilder()
                                    .addArgReplacements(
                                        Config.Replacement.newBuilder()
                                            .setSource("foo[12]")
                                            .setTarget("bar"))))
                    .build())
            .forSpawn(createSpawn());

    assertThat(spawnScrubber.transformArgument("foo1")).isEqualTo("bar");
    assertThat(spawnScrubber.transformArgument("foo2")).isEqualTo("bar");
    assertThat(spawnScrubber.transformArgument("foo3")).isEqualTo("foo3");
  }

  @Test
  public void multipleArgReplacements() {
    var spawnScrubber =
        new Scrubber(
                Config.newBuilder()
                    .addRules(
                        Config.Rule.newBuilder()
                            .setTransform(
                                Config.Transform.newBuilder()
                                    .addArgReplacements(
                                        Config.Replacement.newBuilder()
                                            .setSource("foo")
                                            .setTarget("bar"))
                                    .addArgReplacements(
                                        Config.Replacement.newBuilder()
                                            .setSource("spam")
                                            .setTarget("eggs"))))
                    .build())
            .forSpawn(createSpawn());

    assertThat(spawnScrubber.transformArgument("abcfoo123spamxyz")).isEqualTo("abcbar123eggsxyz");
    assertThat(spawnScrubber.transformArgument("abcfoo")).isEqualTo("abcbar");
    assertThat(spawnScrubber.transformArgument("abcspam")).isEqualTo("abceggs");
    assertThat(spawnScrubber.transformArgument("bareggs")).isEqualTo("bareggs");
  }

  @Test
  public void withoutSalt() {
    var spawnScrubber =
        new Scrubber(Config.newBuilder().addRules(Config.Rule.getDefaultInstance()).build())
            .forSpawn(createSpawn());

    assertThat(spawnScrubber.getSalt()).isEmpty();
  }

  @Test
  public void withSalt() {
    var spawnScrubber =
        new Scrubber(
                Config.newBuilder()
                    .addRules(
                        Config.Rule.newBuilder()
                            .setTransform(Config.Transform.newBuilder().setSalt("NaCl")))
                    .build())
            .forSpawn(createSpawn());

    assertThat(spawnScrubber.getSalt()).isEqualTo("NaCl");
  }

  @Test
  public void orthogonalRules() {
    var scrubber =
        new Scrubber(
            Config.newBuilder()
                .addRules(
                    Config.Rule.newBuilder()
                        .setMatcher(Config.Matcher.newBuilder().setLabel("//foo:bar"))
                        .setTransform(
                            Config.Transform.newBuilder()
                                .addArgReplacements(
                                    Config.Replacement.newBuilder()
                                        .setSource("foo")
                                        .setTarget("bar"))))
                .addRules(
                    Config.Rule.newBuilder()
                        .setMatcher(Config.Matcher.newBuilder().setLabel("//spam:eggs"))
                        .setTransform(
                            Config.Transform.newBuilder()
                                .addArgReplacements(
                                    Config.Replacement.newBuilder()
                                        .setSource("spam")
                                        .setTarget("eggs"))))
                .build());

    SpawnScrubber spawnScrubberForFooBar = scrubber.forSpawn(createSpawn("//foo:bar", "Foo"));
    assertThat(spawnScrubberForFooBar).isNotNull();
    assertThat(spawnScrubberForFooBar.transformArgument("foospam")).isEqualTo("barspam");

    SpawnScrubber spawnScrubberForSpamEggs = scrubber.forSpawn(createSpawn("//spam:eggs", "Spam"));
    assertThat(spawnScrubberForSpamEggs).isNotNull();
    assertThat(spawnScrubberForSpamEggs.transformArgument("foospam")).isEqualTo("fooeggs");
  }

  @Test
  public void lastRuleWins() {
    var spawnScrubber =
        new Scrubber(
                Config.newBuilder()
                    .addRules(
                        Config.Rule.newBuilder()
                            .setTransform(
                                Config.Transform.newBuilder()
                                    .addArgReplacements(
                                        Config.Replacement.newBuilder()
                                            .setSource("foo")
                                            .setTarget("bar"))))
                    .addRules(
                        Config.Rule.newBuilder()
                            .setTransform(
                                Config.Transform.newBuilder()
                                    .addArgReplacements(
                                        Config.Replacement.newBuilder()
                                            .setSource("spam")
                                            .setTarget("eggs"))))
                    .build())
            .forSpawn(createSpawn());

    assertThat(spawnScrubber.transformArgument("foospam")).isEqualTo("fooeggs");
  }

  private static Spawn createSpawn() {
    return createSpawn("//foo:bar", "Foo");
  }

  private static Spawn createSpawn(String label, String mnemonic) {
    return createSpawn(label, mnemonic, /* ruleKind= */ "dummy-target-kind", /* forTool= */ false);
  }

  private static Spawn createSpawn(
      String label, String mnemonic, String ruleKind, boolean forTool) {
    return new SpawnBuilder("cmd")
        .withOwnerLabel(label)
        .withMnemonic(mnemonic)
        .withOwnerRuleKind(ruleKind)
        .setBuiltForToolConfiguration(forTool)
        .build();
  }
}
