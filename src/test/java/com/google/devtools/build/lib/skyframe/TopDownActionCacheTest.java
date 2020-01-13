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

import com.google.common.cache.Cache;
import com.google.common.cache.CacheBuilder;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.util.TestAction;
import com.google.devtools.build.lib.actionsketch.ActionSketch;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for top-down, transitive action caching. */
@RunWith(JUnit4.class)
public class TopDownActionCacheTest extends TimestampBuilderTestCase {

  @Override
  protected TopDownActionCache initTopDownActionCache() {
    return new InMemoryTopDownActionCache();
  }

  private void buildArtifacts(Artifact... artifacts) throws Exception {
    buildArtifacts(amnesiacBuilder(), artifacts);
  }

  private static NestedSet<Artifact> asNestedSet(Artifact... artifacts) {
    return NestedSetBuilder.create(Order.STABLE_ORDER, artifacts);
  }

  @Test
  public void testAmnesiacBuilderGetsTopDownHit() throws Exception {
    Artifact hello = createDerivedArtifact("hello");
    Button button = createActionButton(emptyNestedSet, ImmutableSet.of(hello));

    button.pressed = false;
    buildArtifacts(hello);
    assertThat(button.pressed).isTrue();

    button.pressed = false;
    buildArtifacts(hello);
    assertThat(button.pressed).isFalse();
  }

  @Test
  public void testTransitiveTopDownCache() throws Exception {
    Artifact hello = createDerivedArtifact("hello");
    Artifact hello2 = createDerivedArtifact("hello2");
    Button button = createActionButton(emptyNestedSet, ImmutableSet.of(hello));
    Button button2 = createActionButton(asNestedSet(hello), ImmutableSet.of(hello2));

    button.pressed = false;
    button2.pressed = false;
    buildArtifacts(hello2);
    assertThat(button.pressed).isTrue();
    assertThat(button2.pressed).isTrue();

    button.pressed = false;
    button2.pressed = false;
    buildArtifacts(hello2);
    assertThat(button.pressed).isFalse();
    assertThat(button2.pressed).isFalse();
  }

  @Test
  public void testActionKeyCaching() throws Exception {
    Artifact hello = createDerivedArtifact("hello");
    Artifact hello2 = createDerivedArtifact("hello2");

    ActionKeyButton button = createActionKeyButton(emptyNestedSet, ImmutableSet.of(hello), "abc");
    ActionKeyButton button2 =
        createActionKeyButton(asNestedSet(hello), ImmutableSet.of(hello2), "xyz");

    button.pressed = false;
    button2.pressed = false;
    buildArtifacts(hello2);
    assertThat(button.pressed).isTrue();
    assertThat(button2.pressed).isTrue();

    button.pressed = false;
    button2.pressed = false;
    buildArtifacts(hello2);
    assertThat(button.pressed).isFalse();
    assertThat(button2.pressed).isFalse();

    clearActions();
    hello = createDerivedArtifact("hello");
    hello2 = createDerivedArtifact("hello2");
    button = createActionKeyButton(emptyNestedSet, ImmutableSet.of(hello), "abc");
    button2 = createActionKeyButton(asNestedSet(hello), ImmutableSet.of(hello2), "123");
    button.pressed = false;
    button2.pressed = false;
    buildArtifacts(hello2);
    assertThat(button.pressed).isFalse();
    assertThat(button2.pressed).isTrue();

    button.pressed = false;
    button2.pressed = false;
    buildArtifacts(hello2);
    assertThat(button.pressed).isFalse();
    assertThat(button2.pressed).isFalse();

    clearActions();
    hello = createDerivedArtifact("hello");
    hello2 = createDerivedArtifact("hello2");
    button = createActionKeyButton(emptyNestedSet, ImmutableSet.of(hello), "456");
    button2 = createActionKeyButton(asNestedSet(hello), ImmutableSet.of(hello2), "123");
    button.pressed = false;
    button2.pressed = false;
    buildArtifacts(hello2);
    assertThat(button.pressed).isTrue();
    assertThat(button2.pressed).isTrue();

    button.pressed = false;
    button2.pressed = false;
    buildArtifacts(hello2);
    assertThat(button.pressed).isFalse();
    assertThat(button2.pressed).isFalse();
  }

  @Test
  public void testSingleSourceArtifactChanged() throws Exception {
    Artifact hello = createSourceArtifact("hello");
    hello.getPath().getParentDirectory().createDirectoryAndParents();
    FileSystemUtils.writeContentAsLatin1(hello.getPath(), "content1");

    Artifact goodbye = createDerivedArtifact("goodbye");
    Button button = createActionButton(asNestedSet(hello), ImmutableSet.of(goodbye));

    button.pressed = false;
    buildArtifacts(goodbye);
    assertThat(button.pressed).isTrue();

    button.pressed = false;
    buildArtifacts(goodbye);
    assertThat(button.pressed).isFalse(); // top-down cached

    FileSystemUtils.writeContentAsLatin1(hello.getPath(), "content1");
    button.pressed = false;
    buildArtifacts(goodbye);
    assertThat(button.pressed).isFalse(); // top-down cached

    FileSystemUtils.writeContentAsLatin1(hello.getPath(), "content2");
    button.pressed = false;
    buildArtifacts(goodbye);
    assertThat(button.pressed).isTrue(); // built

    button.pressed = false;
    buildArtifacts(goodbye);
    assertThat(button.pressed).isFalse(); // top-down cached
  }

  private static class InMemoryTopDownActionCache implements TopDownActionCache {
    private final Cache<ActionSketch, ActionExecutionValue> cache =
        CacheBuilder.newBuilder().build();

    @Nullable
    @Override
    public ActionExecutionValue get(ActionSketch sketch) {
      return cache.getIfPresent(sketch);
    }

    @Override
    public void put(ActionSketch sketch, ActionExecutionValue value) {
      cache.put(sketch, value);
    }
  }

  private static class MutableActionKeyAction extends TestAction {

    private final ActionKeyButton button;

    public MutableActionKeyAction(
        ActionKeyButton button, NestedSet<Artifact> inputs, ImmutableSet<Artifact> outputs) {
      super(button, inputs, outputs);
      this.button = button;
    }

    @Override
    protected void computeKey(ActionKeyContext actionKeyContext, Fingerprint fp) {
      super.computeKey(actionKeyContext, fp);
      fp.addString(button.key);
    }
  }

  private static class ActionKeyButton extends Button {
    private final String key;

    public ActionKeyButton(String key) {
      this.key = key;
    }
  }

  private ActionKeyButton createActionKeyButton(
      NestedSet<Artifact> inputs, ImmutableSet<Artifact> outputs, String key) {
    ActionKeyButton button = new ActionKeyButton(key);
    registerAction(new MutableActionKeyAction(button, inputs, outputs));
    return button;
  }
}
