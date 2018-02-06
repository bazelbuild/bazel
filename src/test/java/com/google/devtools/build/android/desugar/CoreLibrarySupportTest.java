// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.desugar;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import java.util.Collection;
import java.util.Comparator;
import java.util.Map;
import java.util.concurrent.ConcurrentMap;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.objectweb.asm.Opcodes;

@RunWith(JUnit4.class)
public class CoreLibrarySupportTest {

  @Test
  public void testIsRenamedCoreLibrary() throws Exception {
    CoreLibrarySupport support =
        new CoreLibrarySupport(
            new CoreLibraryRewriter(""), null, ImmutableList.of("java/time/"), ImmutableList.of());
    assertThat(support.isRenamedCoreLibrary("java/time/X")).isTrue();
    assertThat(support.isRenamedCoreLibrary("java/time/y/X")).isTrue();
    assertThat(support.isRenamedCoreLibrary("java/io/X")).isFalse();
    assertThat(support.isRenamedCoreLibrary("com/google/X")).isFalse();
  }

  @Test
  public void testIsRenamedCoreLibrary_prefixedLoader() throws Exception {
    CoreLibrarySupport support =
        new CoreLibrarySupport(
            new CoreLibraryRewriter("__/"),
            null,
            ImmutableList.of("java/time/"),
            ImmutableList.of());
    assertThat(support.isRenamedCoreLibrary("__/java/time/X")).isTrue();
    assertThat(support.isRenamedCoreLibrary("__/java/time/y/X")).isTrue();
    assertThat(support.isRenamedCoreLibrary("__/java/io/X")).isFalse();
    assertThat(support.isRenamedCoreLibrary("com/google/X")).isFalse();
  }
  @Test
  public void testRenameCoreLibrary() throws Exception {
    CoreLibrarySupport support =
        new CoreLibrarySupport(
            new CoreLibraryRewriter(""), null, ImmutableList.of(), ImmutableList.of());
    assertThat(support.renameCoreLibrary("java/time/X")).isEqualTo("j$/time/X");
    assertThat(support.renameCoreLibrary("com/google/X")).isEqualTo("com/google/X");
  }

  @Test
  public void testRenameCoreLibrary_prefixedLoader() throws Exception {
    CoreLibrarySupport support =
        new CoreLibrarySupport(
            new CoreLibraryRewriter("__/"), null, ImmutableList.of(), ImmutableList.of());
    assertThat(support.renameCoreLibrary("__/java/time/X")).isEqualTo("j$/time/X");
    assertThat(support.renameCoreLibrary("com/google/X")).isEqualTo("com/google/X");
  }

  @Test
  public void testIsEmulatedCoreLibraryInvocation() throws Exception {
    CoreLibrarySupport support =
        new CoreLibrarySupport(
            new CoreLibraryRewriter(""),
            Thread.currentThread().getContextClassLoader(),
            ImmutableList.of(),
            ImmutableList.of("java/util/Collection"));
    assertThat(
            support.isEmulatedCoreLibraryInvocation(
                Opcodes.INVOKEINTERFACE,
                "java/util/Collection",
                "removeIf",
                "(Ljava/util/function/Predicate;)Z",
                true))
        .isTrue(); // true for default method
    assertThat(
            support.isEmulatedCoreLibraryInvocation(
                Opcodes.INVOKEINTERFACE, "java/util/Collection", "size", "()I", true))
        .isFalse(); // false for abstract method
  }

  @Test
  public void testGetEmulatedCoreLibraryInvocationTarget_defaultMethod() throws Exception {
    CoreLibrarySupport support =
        new CoreLibrarySupport(
            new CoreLibraryRewriter(""),
            Thread.currentThread().getContextClassLoader(),
            ImmutableList.of(),
            ImmutableList.of("java/util/Collection"));
    assertThat(
            support.getEmulatedCoreLibraryInvocationTarget(
                Opcodes.INVOKEINTERFACE,
                "java/util/Collection",
                "removeIf",
                "(Ljava/util/function/Predicate;)Z",
                true))
        .isEqualTo(Collection.class);
    assertThat(
            support.getEmulatedCoreLibraryInvocationTarget(
                Opcodes.INVOKEVIRTUAL,
                "java/util/ArrayList",
                "removeIf",
                "(Ljava/util/function/Predicate;)Z",
                false))
        .isEqualTo(Collection.class);
    assertThat(
            support.getEmulatedCoreLibraryInvocationTarget(
                Opcodes.INVOKEINTERFACE,
                "com/google/common/collect/ImmutableList",
                "removeIf",
                "(Ljava/util/function/Predicate;)Z",
                true))
        .isNull();
  }

  @Test
  public void testGetEmulatedCoreLibraryInvocationTarget_abstractMethod() throws Exception {
    CoreLibrarySupport support =
        new CoreLibrarySupport(
            new CoreLibraryRewriter(""),
            Thread.currentThread().getContextClassLoader(),
            ImmutableList.of(),
            ImmutableList.of("java/util/Collection"));
    assertThat(
            support.getEmulatedCoreLibraryInvocationTarget(
                Opcodes.INVOKEINTERFACE,
                "java/util/Collection",
                "size",
                "()I",
                true))
        .isNull();
    assertThat(
            support.getEmulatedCoreLibraryInvocationTarget(
                Opcodes.INVOKEVIRTUAL,
                "java/util/ArrayList",
                "size",
                "()I",
                false))
        .isNull();
  }

  @Test
  public void testGetEmulatedCoreLibraryInvocationTarget_defaultOverride() throws Exception {
    CoreLibrarySupport support =
        new CoreLibrarySupport(
            new CoreLibraryRewriter(""),
            Thread.currentThread().getContextClassLoader(),
            ImmutableList.of(),
            ImmutableList.of("java/util/Map"));
    assertThat(
            support.getEmulatedCoreLibraryInvocationTarget(
                Opcodes.INVOKEINTERFACE,
                "java/util/Map",
                "putIfAbsent",
                "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;",
                true))
        .isEqualTo(Map.class);
    assertThat(
            support.getEmulatedCoreLibraryInvocationTarget(
                Opcodes.INVOKEINTERFACE,
                "java/util/concurrent/ConcurrentMap",
                "putIfAbsent",
                "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;",
                true))
        .isNull(); // putIfAbsent is default in Map but abstract in ConcurrentMap
    assertThat(
            support.getEmulatedCoreLibraryInvocationTarget(
                Opcodes.INVOKEINTERFACE,
                "java/util/concurrent/ConcurrentMap",
                "getOrDefault",
                "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;",
                true))
        .isEqualTo(ConcurrentMap.class); // conversely, getOrDefault is overridden as default method
  }

  @Test
  public void testGetEmulatedCoreLibraryInvocationTarget_staticInterfaceMethod() throws Exception {
    CoreLibrarySupport support =
        new CoreLibrarySupport(
            new CoreLibraryRewriter(""),
            Thread.currentThread().getContextClassLoader(),
            ImmutableList.of(),
            ImmutableList.of("java/util/Comparator"));
    assertThat(
            support.getEmulatedCoreLibraryInvocationTarget(
                Opcodes.INVOKESTATIC,
                "java/util/Comparator",
                "reverseOrder",
                "()Ljava/util/Comparator;",
                true))
        .isEqualTo(Comparator.class);
  }

  @Test
  public void testGetEmulatedCoreLibraryInvocationTarget_ignoreRenamed() throws Exception {
    CoreLibrarySupport support =
        new CoreLibrarySupport(
            new CoreLibraryRewriter(""),
            Thread.currentThread().getContextClassLoader(),
            ImmutableList.of("java/util/concurrent/"),  // should return null for these
            ImmutableList.of("java/util/Map"));
    assertThat(
            support.getEmulatedCoreLibraryInvocationTarget(
                Opcodes.INVOKEINTERFACE,
                "java/util/Map",
                "getOrDefault",
                "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;",
                true))
        .isEqualTo(Map.class);
    assertThat(
            support.getEmulatedCoreLibraryInvocationTarget(
                Opcodes.INVOKEINTERFACE,
                "java/util/concurrent/ConcurrentMap",
                "getOrDefault",
                "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;",
                true))
        .isNull();
  }
}
