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
import com.google.devtools.build.android.desugar.io.CoreLibraryRewriter;
import java.util.Collection;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentMap;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.objectweb.asm.Opcodes;

/** Tests for {@link CoreLibrarySupport}. */
@RunWith(JUnit4.class)
public class CoreLibrarySupportTest {

  @Test
  public void testIsRenamedCoreLibrary() throws Exception {
    CoreLibrarySupport support =
        new CoreLibrarySupport(
            new CoreLibraryRewriter(""),
            null,
            ImmutableList.of("java/time/"),
            ImmutableList.of(),
            ImmutableList.of(),
            ImmutableList.of());
    assertThat(support.isRenamedCoreLibrary("java/time/X")).isTrue();
    assertThat(support.isRenamedCoreLibrary("java/time/y/X")).isTrue();
    assertThat(support.isRenamedCoreLibrary("java/io/X")).isFalse();
    assertThat(support.isRenamedCoreLibrary("java/io/X$$CC")).isTrue();
    assertThat(support.isRenamedCoreLibrary("java/io/X$$Lambda$17")).isTrue();
    assertThat(support.isRenamedCoreLibrary("com/google/X")).isFalse();
  }

  @Test
  public void testIsRenamedCoreLibrary_prefixedLoader() throws Exception {
    CoreLibrarySupport support =
        new CoreLibrarySupport(
            new CoreLibraryRewriter("__/"),
            null,
            ImmutableList.of("java/time/"),
            ImmutableList.of(),
            ImmutableList.of(),
            ImmutableList.of());
    assertThat(support.isRenamedCoreLibrary("__/java/time/X")).isTrue();
    assertThat(support.isRenamedCoreLibrary("__/java/time/y/X")).isTrue();
    assertThat(support.isRenamedCoreLibrary("__/java/io/X")).isFalse();
    assertThat(support.isRenamedCoreLibrary("__/java/io/X$$CC")).isTrue();
    assertThat(support.isRenamedCoreLibrary("__/java/io/X$$Lambda$17")).isTrue();
    assertThat(support.isRenamedCoreLibrary("com/google/X")).isFalse();
  }

  @Test
  public void testRenameCoreLibrary() throws Exception {
    CoreLibrarySupport support =
        new CoreLibrarySupport(
            new CoreLibraryRewriter(""),
            null,
            ImmutableList.of(),
            ImmutableList.of(),
            ImmutableList.of(),
            ImmutableList.of());
    assertThat(support.renameCoreLibrary("java/time/X")).isEqualTo("j$/time/X");
    assertThat(support.renameCoreLibrary("com/google/X")).isEqualTo("com/google/X");
  }

  @Test
  public void testRenameCoreLibrary_prefixedLoader() throws Exception {
    CoreLibrarySupport support =
        new CoreLibrarySupport(
            new CoreLibraryRewriter("__/"),
            null,
            ImmutableList.of(),
            ImmutableList.of(),
            ImmutableList.of(),
            ImmutableList.of());
    assertThat(support.renameCoreLibrary("__/java/time/X")).isEqualTo("j$/time/X");
    assertThat(support.renameCoreLibrary("com/google/X")).isEqualTo("com/google/X");
  }

  @Test
  public void testMoveTarget() throws Exception {
    CoreLibrarySupport support =
        new CoreLibrarySupport(
            new CoreLibraryRewriter("__/"),
            null,
            ImmutableList.of("java/util/Helper"),
            ImmutableList.of(),
            ImmutableList.of(
                "java/util/Existing#match -> java/util/Helper",
                "java/util/Existing#unused -> com/google/Unused"),
            ImmutableList.of());
    assertThat(support.getMoveTarget("__/java/util/Existing", "match")).isEqualTo("j$/util/Helper");
    assertThat(support.getMoveTarget("java/util/Existing", "match")).isEqualTo("j$/util/Helper");
    assertThat(support.getMoveTarget("__/java/util/Existing", "matchesnot")).isNull();
    assertThat(support.getMoveTarget("__/java/util/ExistingOther", "match")).isNull();
    assertThat(support.usedRuntimeHelpers()).containsExactly("j$/util/Helper");
  }

  @Test
  public void testIsEmulatedCoreClassOrInterface() throws Exception {
    CoreLibrarySupport support =
        new CoreLibrarySupport(
            new CoreLibraryRewriter(""),
            Thread.currentThread().getContextClassLoader(),
            ImmutableList.of("java/util/concurrent/"),
            ImmutableList.of("java/util/Map"),
            ImmutableList.of(),
            ImmutableList.of());
    assertThat(support.isEmulatedCoreClassOrInterface("java/util/Map")).isTrue();
    assertThat(support.isEmulatedCoreClassOrInterface("java/util/Map$$Lambda$17")).isFalse();
    assertThat(support.isEmulatedCoreClassOrInterface("java/util/Map$$CC")).isFalse();
    assertThat(support.isEmulatedCoreClassOrInterface("java/util/HashMap")).isTrue();
    assertThat(support.isEmulatedCoreClassOrInterface("java/util/concurrent/ConcurrentMap"))
        .isFalse(); // false for renamed prefixes
    assertThat(support.isEmulatedCoreClassOrInterface("com/google/Map")).isFalse();
  }

  @Test
  public void testGetCoreInterfaceRewritingTarget_emulatedDefaultMethod() throws Exception {
    CoreLibrarySupport support =
        new CoreLibrarySupport(
            new CoreLibraryRewriter(""),
            Thread.currentThread().getContextClassLoader(),
            ImmutableList.of(),
            ImmutableList.of("java/util/Collection"),
            ImmutableList.of(),
            ImmutableList.of());
    assertThat(
            support.getCoreInterfaceRewritingTarget(
                Opcodes.INVOKEINTERFACE,
                "java/util/Collection",
                "removeIf",
                "(Ljava/util/function/Predicate;)Z",
                true))
        .isEqualTo(Collection.class);
    assertThat(
            support.getCoreInterfaceRewritingTarget(
                Opcodes.INVOKEVIRTUAL,
                "java/util/ArrayList",
                "removeIf",
                "(Ljava/util/function/Predicate;)Z",
                false))
        .isEqualTo(Collection.class);
    assertThat(
            support.getCoreInterfaceRewritingTarget(
                Opcodes.INVOKEINTERFACE,
                "com/google/HypotheticalListInterface",
                "removeIf",
                "(Ljava/util/function/Predicate;)Z",
                true))
        .isNull();
  }

  @Test
  public void testGetCoreInterfaceRewritingTarget_emulatedImplementationMoved() throws Exception {
    CoreLibrarySupport support =
        new CoreLibrarySupport(
            new CoreLibraryRewriter(""),
            Thread.currentThread().getContextClassLoader(),
            ImmutableList.of("java/util/Moved"),
            ImmutableList.of("java/util/Map"),
            ImmutableList.of("java/util/LinkedHashMap#forEach->java/util/Moved"),
            ImmutableList.of());
    assertThat(
            support.getCoreInterfaceRewritingTarget(
                Opcodes.INVOKEINTERFACE,
                "java/util/Map",
                "forEach",
                "(Ljava/util/function/BiConsumer;)V",
                true))
        .isEqualTo(Map.class);
    assertThat(
            support.getCoreInterfaceRewritingTarget(
                Opcodes.INVOKESPECIAL,
                "java/util/Map",
                "forEach",
                "(Ljava/util/function/BiConsumer;)V",
                true))
        .isEqualTo(Map.class);
    assertThat(
            support.getCoreInterfaceRewritingTarget(
                Opcodes.INVOKEVIRTUAL,
                "java/util/LinkedHashMap",
                "forEach",
                "(Ljava/util/function/BiConsumer;)V",
                false))
        .isEqualTo(Map.class);
    assertThat(
            support.getCoreInterfaceRewritingTarget(
                Opcodes.INVOKESPECIAL,
                "java/util/LinkedHashMap",
                "forEach",
                "(Ljava/util/function/BiConsumer;)V",
                false))
        .isEqualTo(LinkedHashMap.class);
    assertThat(
            support.getCoreInterfaceRewritingTarget(
                Opcodes.INVOKESPECIAL,
                "java/util/HashMap",
                "forEach",
                "(Ljava/util/function/BiConsumer;)V",
                false))
        .isEqualTo(Map.class);
  }

  @Test
  public void testGetCoreInterfaceRewritingTarget_abstractMethod() throws Exception {
    CoreLibrarySupport support =
        new CoreLibrarySupport(
            new CoreLibraryRewriter(""),
            Thread.currentThread().getContextClassLoader(),
            ImmutableList.of(),
            ImmutableList.of("java/util/Collection"),
            ImmutableList.of(),
            ImmutableList.of());
    assertThat(
            support.getCoreInterfaceRewritingTarget(
                Opcodes.INVOKEINTERFACE, "java/util/Collection", "size", "()I", true))
        .isNull();
    assertThat(
            support.getCoreInterfaceRewritingTarget(
                Opcodes.INVOKEVIRTUAL, "java/util/ArrayList", "size", "()I", false))
        .isNull();
  }

  @Test
  public void testGetCoreInterfaceRewritingTarget_emulatedDefaultOverride() throws Exception {
    CoreLibrarySupport support =
        new CoreLibrarySupport(
            new CoreLibraryRewriter(""),
            Thread.currentThread().getContextClassLoader(),
            ImmutableList.of(),
            ImmutableList.of("java/util/Map"),
            ImmutableList.of(),
            ImmutableList.of());
    assertThat(
            support.getCoreInterfaceRewritingTarget(
                Opcodes.INVOKEINTERFACE,
                "java/util/Map",
                "putIfAbsent",
                "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;",
                true))
        .isEqualTo(Map.class);
    assertThat(
            support.getCoreInterfaceRewritingTarget(
                Opcodes.INVOKEINTERFACE,
                "java/util/concurrent/ConcurrentMap",
                "putIfAbsent",
                "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;",
                true))
        .isNull(); // putIfAbsent is default in Map but abstract in ConcurrentMap
    assertThat(
            support.getCoreInterfaceRewritingTarget(
                Opcodes.INVOKEINTERFACE,
                "java/util/concurrent/ConcurrentMap",
                "getOrDefault",
                "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;",
                true))
        .isEqualTo(ConcurrentMap.class); // conversely, getOrDefault is overridden as default method
  }

  @Test
  public void testGetCoreInterfaceRewritingTarget_staticInterfaceMethod() throws Exception {
    CoreLibrarySupport support =
        new CoreLibrarySupport(
            new CoreLibraryRewriter(""),
            Thread.currentThread().getContextClassLoader(),
            ImmutableList.of(),
            ImmutableList.of("java/util/Comparator"),
            ImmutableList.of(),
            ImmutableList.of());
    assertThat(
            support.getCoreInterfaceRewritingTarget(
                Opcodes.INVOKESTATIC,
                "java/util/Comparator",
                "reverseOrder",
                "()Ljava/util/Comparator;",
                true))
        .isEqualTo(Comparator.class);
  }

  /**
   * Tests that call sites of renamed core libraries are treated like call sites in regular {@link
   * InterfaceDesugaring}.
   */
  @Test
  public void testGetCoreInterfaceRewritingTarget_renamed() throws Exception {
    CoreLibrarySupport support =
        new CoreLibrarySupport(
            new CoreLibraryRewriter(""),
            Thread.currentThread().getContextClassLoader(),
            ImmutableList.of("java/util/"),
            ImmutableList.of(),
            ImmutableList.of(),
            ImmutableList.of());

    // regular invocations of default methods: ignored
    assertThat(
            support.getCoreInterfaceRewritingTarget(
                Opcodes.INVOKEINTERFACE,
                "java/util/Collection",
                "removeIf",
                "(Ljava/util/function/Predicate;)Z",
                true))
        .isNull();
    assertThat(
            support.getCoreInterfaceRewritingTarget(
                Opcodes.INVOKEVIRTUAL,
                "java/util/ArrayList",
                "removeIf",
                "(Ljava/util/function/Predicate;)Z",
                false))
        .isNull();

    // abstract methods: ignored
    assertThat(
            support.getCoreInterfaceRewritingTarget(
                Opcodes.INVOKEINTERFACE, "java/util/Collection", "size", "()I", true))
        .isNull();

    // static interface method
    assertThat(
            support.getCoreInterfaceRewritingTarget(
                Opcodes.INVOKESTATIC,
                "java/util/Comparator",
                "reverseOrder",
                "()Ljava/util/Comparator;",
                true))
        .isEqualTo(Comparator.class);

    // invokespecial for default methods: find nearest definition
    assertThat(
            support.getCoreInterfaceRewritingTarget(
                Opcodes.INVOKESPECIAL,
                "java/util/List",
                "removeIf",
                "(Ljava/util/function/Predicate;)Z",
                true))
        .isEqualTo(Collection.class);
    // invokespecial on a class: ignore (even if there's an inherited default method)
    assertThat(
            support.getCoreInterfaceRewritingTarget(
                Opcodes.INVOKESPECIAL,
                "java/util/ArrayList",
                "removeIf",
                "(Ljava/util/function/Predicate;)Z",
                false))
        .isNull();
  }

  @Test
  public void testGetCoreInterfaceRewritingTarget_ignoreRenamedInvokeInterface() throws Exception {
    CoreLibrarySupport support =
        new CoreLibrarySupport(
            new CoreLibraryRewriter(""),
            Thread.currentThread().getContextClassLoader(),
            ImmutableList.of("java/util/concurrent/"), // should return null for these
            ImmutableList.of("java/util/Map"),
            ImmutableList.of(),
            ImmutableList.of());
    assertThat(
            support.getCoreInterfaceRewritingTarget(
                Opcodes.INVOKEINTERFACE,
                "java/util/Map",
                "getOrDefault",
                "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;",
                true))
        .isEqualTo(Map.class);
    assertThat(
            support.getCoreInterfaceRewritingTarget(
                Opcodes.INVOKEINTERFACE,
                "java/util/concurrent/ConcurrentMap",
                "getOrDefault",
                "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;",
                true))
        .isNull();
  }

  @Test
  public void testGetCoreInterfaceRewritingTarget_excludedMethodIgnored() throws Exception {
    CoreLibrarySupport support =
        new CoreLibrarySupport(
            new CoreLibraryRewriter(""),
            Thread.currentThread().getContextClassLoader(),
            ImmutableList.of(),
            ImmutableList.of("java/util/Collection"),
            ImmutableList.of(),
            ImmutableList.of("java/util/Collection#removeIf"));
    assertThat(
            support.getCoreInterfaceRewritingTarget(
                Opcodes.INVOKEINTERFACE,
                "java/util/List",
                "removeIf",
                "(Ljava/util/function/Predicate;)Z",
                true))
        .isNull();
    assertThat(
            support.getCoreInterfaceRewritingTarget(
                Opcodes.INVOKEVIRTUAL,
                "java/util/ArrayList",
                "removeIf",
                "(Ljava/util/function/Predicate;)Z",
                false))
        .isNull();
  }

  @Test
  public void testEmulatedMethod_nullExceptions() throws Exception {
    CoreLibrarySupport.EmulatedMethod m =
        CoreLibrarySupport.EmulatedMethod.create(1, Number.class, "a", "()V", null);
    assertThat(m.access()).isEqualTo(1);
    assertThat(m.owner()).isEqualTo(Number.class);
    assertThat(m.name()).isEqualTo("a");
    assertThat(m.descriptor()).isEqualTo("()V");
    assertThat(m.exceptions()).isEmpty();
  }

  @Test
  public void testEmulatedMethod_givenExceptions() throws Exception {
    CoreLibrarySupport.EmulatedMethod m =
        CoreLibrarySupport.EmulatedMethod.create(
            1, Number.class, "a", "()V", new String[] {"b", "c"});
    assertThat(m.access()).isEqualTo(1);
    assertThat(m.owner()).isEqualTo(Number.class);
    assertThat(m.name()).isEqualTo("a");
    assertThat(m.descriptor()).isEqualTo("()V");
    assertThat(m.exceptions()).containsExactly("b", "c").inOrder();
  }
}
