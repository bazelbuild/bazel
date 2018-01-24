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
import com.google.devtools.build.lib.rules.objc.ObjcProvider.Key;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link ObjcProvider}. */
@RunWith(JUnit4.class)
public class ObjcProviderTest {

  @Test
  public void emptyProvider() {
    ObjcProvider empty = new ObjcProvider.Builder().build();
    assertThat(empty.get(ObjcProvider.SDK_DYLIB)).isEmpty();
  }

  @Test
  public void onlyPropagatesProvider() {
    ObjcProvider onlyPropagates = new ObjcProvider.Builder()
        .add(ObjcProvider.SDK_DYLIB, "foo")
        .build();
    assertThat(onlyPropagates.get(ObjcProvider.SDK_DYLIB)).containsExactly("foo");
  }

  @Test
  public void onlyNonPropagatesProvider() {
    ObjcProvider dep = new ObjcProvider.Builder()
        .add(ObjcProvider.SDK_DYLIB, "foo")
        .build();
    ObjcProvider notPropagates = new ObjcProvider.Builder()
        .addTransitiveWithoutPropagating(ImmutableList.of(dep))
        .build();
    assertThat(notPropagates.get(ObjcProvider.SDK_DYLIB)).containsExactly("foo");
  }

  @Test
  public void propagatesAndNonPropagatesProvider() {
    ObjcProvider dep = new ObjcProvider.Builder()
        .add(ObjcProvider.SDK_DYLIB, "foo")
        .build();
    ObjcProvider provider = new ObjcProvider.Builder()
        .addTransitiveWithoutPropagating(ImmutableList.of(dep))
        .add(ObjcProvider.SDK_DYLIB, "bar")
        .build();
    assertThat(provider.get(ObjcProvider.SDK_DYLIB)).containsExactly("foo", "bar").inOrder();
  }

  @Test
  public void doesNotPropagate() {
    ObjcProvider dep = new ObjcProvider.Builder()
        .add(ObjcProvider.SDK_DYLIB, "foo")
        .build();
    ObjcProvider provider = new ObjcProvider.Builder()
        .addTransitiveWithoutPropagating(ImmutableList.of(dep))
        .add(ObjcProvider.SDK_DYLIB, "bar")
        .build();
    ObjcProvider depender = new ObjcProvider.Builder()
        .addTransitiveAndPropagate(provider)
        .build();
    assertThat(depender.get(ObjcProvider.SDK_DYLIB)).containsExactly("bar");
  }

  @Test
  public void strictDependencyDoesNotPropagateMoreThanOneLevel() {
    PathFragment strictInclude = PathFragment.create("strict_path");
    PathFragment propagatedInclude = PathFragment.create("propagated_path");

    ObjcProvider strictDep =
        new ObjcProvider.Builder()
            .addForDirectDependents(ObjcProvider.INCLUDE, strictInclude)
            .build();
    ObjcProvider propagatedDep =
        new ObjcProvider.Builder().add(ObjcProvider.INCLUDE, propagatedInclude).build();

    ObjcProvider provider =
        new ObjcProvider.Builder()
            .addTransitiveAndPropagate(ImmutableList.of(strictDep, propagatedDep))
            .build();
    ObjcProvider depender = new ObjcProvider.Builder().addTransitiveAndPropagate(provider).build();

    assertThat(provider.get(ObjcProvider.INCLUDE))
        .containsExactly(strictInclude, propagatedInclude);
    assertThat(depender.get(ObjcProvider.INCLUDE)).containsExactly(propagatedInclude);
  }

  @Test
  public void strictDependencyDoesNotPropagateMoreThanOneLevelOnSkylark() {
    PathFragment strictInclude = PathFragment.create("strict_path");
    PathFragment propagatedInclude = PathFragment.create("propagated_path");

    ObjcProvider strictDep =
        new ObjcProvider.Builder()
            .addForDirectDependents(ObjcProvider.INCLUDE, strictInclude)
            .build();
    ObjcProvider propagatedDep =
        new ObjcProvider.Builder().add(ObjcProvider.INCLUDE, propagatedInclude).build();

    ObjcProvider provider =
        new ObjcProvider.Builder()
            .addTransitiveAndPropagate(ImmutableList.of(strictDep, propagatedDep))
            .build();
    ObjcProvider depender = new ObjcProvider.Builder().addTransitiveAndPropagate(provider).build();

    assertThat(provider.include().toCollection())
        .containsExactly(strictInclude.toString(), propagatedInclude.toString());
    assertThat(depender.include().toCollection())
        .containsExactly(propagatedInclude.toString());
  }

  @Test
  public void keysExportedToSkylark() throws Exception {
    List<Field> keyFields = new ArrayList<>();
    for (Field field : ObjcProvider.class.getDeclaredFields()) {
      if (Modifier.isStatic(field.getModifiers()) && field.getType() == ObjcProvider.Key.class) {
        keyFields.add(field);
      }
    }
    ImmutableSet<Key<?>> allRegisteredKeys = ImmutableSet.<Key<?>>builder()
        .addAll(ObjcProvider.KEYS_FOR_SKYLARK)
        .addAll(ObjcProvider.KEYS_NOT_IN_SKYLARK)
        .build();

    for (Field field : keyFields) {
      ObjcProvider.Key<?> key = (Key<?>) field.get(null);
      assertWithMessage("Key %s must either be exposed to skylark or explicitly blacklisted",
          key.getSkylarkKeyName()).that(allRegisteredKeys).contains(key);
    }
  }
}
