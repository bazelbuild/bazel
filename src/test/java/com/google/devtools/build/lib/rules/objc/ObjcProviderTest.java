// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.objc;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.rules.objc.ObjcProvider.Key;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link ObjcProvider}. */
@RunWith(JUnit4.class)
public class ObjcProviderTest {

  private static ObjcProvider.StarlarkBuilder objcProviderBuilder() {
    return new ObjcProvider.StarlarkBuilder(StarlarkSemantics.DEFAULT);
  }

  private static ImmutableList<ObjcProvider.Key<?>> getAllKeys() throws Exception {
    ImmutableList.Builder<ObjcProvider.Key<?>> builder = new ImmutableList.Builder<>();
    for (Field field : ObjcProvider.class.getDeclaredFields()) {
      if (Modifier.isStatic(field.getModifiers()) && field.getType() == ObjcProvider.Key.class) {
        builder.add((ObjcProvider.Key<?>) field.get(null));
      }
    }
    return builder.build();
  }

  private static Artifact createArtifact(String path) {
    return ActionsTestUtil.createArtifact(
        ArtifactRoot.asSourceRoot(Root.absoluteRoot(new InMemoryFileSystem())), path);
  }

  @Test
  public void emptyProvider() {
    ObjcProvider empty = objcProviderBuilder().build();
    assertThat(empty.get(ObjcProvider.SDK_DYLIB).toList()).isEmpty();
  }

  @Test
  public void directFieldsDontPropagateTransitively() {
    Artifact leafArtifact = createArtifact("/main.m");
    ObjcProvider leaf = objcProviderBuilder().add(ObjcProvider.SOURCE, leafArtifact).build();

    Artifact rootArtifact = createArtifact("/root.m");
    ObjcProvider root =
        objcProviderBuilder()
            .addDirect(ObjcProvider.SOURCE, rootArtifact)
            .addTransitiveAndPropagate(leaf)
            .build();
    assertThat(root.getDirect(ObjcProvider.SOURCE)).containsExactly(rootArtifact);
  }

  @Test
  public void directFieldsSingleAdd() {
    Artifact source = createArtifact("/main.m");
    Artifact header = createArtifact("/Foo.h");
    Artifact module = createArtifact("/module.modulemap");
    ObjcProvider provider =
        objcProviderBuilder()
            .addDirect(ObjcProvider.SOURCE, source)
            .addDirect(ObjcProvider.HEADER, header)
            .addDirect(ObjcProvider.MODULE_MAP, module)
            .build();
    assertThat(provider.getDirect(ObjcProvider.SOURCE)).containsExactly(source);
    assertThat(provider.getDirect(ObjcProvider.HEADER)).containsExactly(header);
    assertThat(provider.getDirect(ObjcProvider.MODULE_MAP)).containsExactly(module);
  }

  @Test
  public void directFieldsAddAll() {
    ImmutableList<Artifact> artifacts =
        ImmutableList.of(createArtifact("/foo"), createArtifact("/bar"));
    ObjcProvider provider =
        objcProviderBuilder()
            .addAllDirect(ObjcProvider.SOURCE, artifacts)
            .addAllDirect(ObjcProvider.HEADER, artifacts)
            .addAllDirect(ObjcProvider.MODULE_MAP, artifacts)
            .build();
    assertThat(provider.getDirect(ObjcProvider.SOURCE)).containsExactlyElementsIn(artifacts);
    assertThat(provider.getDirect(ObjcProvider.HEADER)).containsExactlyElementsIn(artifacts);
    assertThat(provider.getDirect(ObjcProvider.MODULE_MAP)).containsExactlyElementsIn(artifacts);
  }

  @Test
  public void directFieldsAddFromSkylark() throws Exception {
    ImmutableList<Artifact> artifacts =
        ImmutableList.of(createArtifact("/foo"), createArtifact("/bar"));
    Depset set = Depset.of(Artifact.TYPE, NestedSetBuilder.wrap(Order.STABLE_ORDER, artifacts));
    ObjcProvider.StarlarkBuilder builder = objcProviderBuilder();
    builder.addElementsFromSkylark(ObjcProvider.SOURCE, set);
    builder.addElementsFromSkylark(ObjcProvider.HEADER, set);
    builder.addElementsFromSkylark(ObjcProvider.MODULE_MAP, set);
    ObjcProvider provider = builder.build();
    assertThat(provider.getDirect(ObjcProvider.SOURCE)).containsExactlyElementsIn(artifacts);
    assertThat(provider.getDirect(ObjcProvider.HEADER)).containsExactlyElementsIn(artifacts);
    assertThat(provider.getDirect(ObjcProvider.MODULE_MAP)).containsExactlyElementsIn(artifacts);
  }

  @Test
  public void onlyPropagatesProvider() {
    ObjcProvider onlyPropagates = objcProviderBuilder()
        .add(ObjcProvider.SDK_DYLIB, "foo")
        .build();
    assertThat(onlyPropagates.get(ObjcProvider.SDK_DYLIB).toList()).containsExactly("foo");
  }

  @Test
  public void keysExportedToSkylark() throws Exception {
    ImmutableSet<Key<?>> allRegisteredKeys = ImmutableSet.<Key<?>>builder()
        .addAll(ObjcProvider.KEYS_FOR_SKYLARK)
        .addAll(ObjcProvider.KEYS_NOT_IN_SKYLARK)
        .build();

    for (ObjcProvider.Key<?> key : getAllKeys()) {
      assertWithMessage(
              "Key %s must either be exposed to Starlark or explicitly blacklisted",
              key.getSkylarkKeyName())
          .that(allRegisteredKeys)
          .contains(key);
    }
  }
}
