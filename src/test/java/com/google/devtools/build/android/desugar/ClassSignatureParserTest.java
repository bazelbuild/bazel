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
package com.google.devtools.build.android.desugar;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.android.desugar.ClassSignatureParser.ClassSignature;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for {@link ClassSignatureParser} based on <a
 * href="https://docs.oracle.com/javase/specs/jvms/se12/html/jvms-4.html#jvms-ClassSignature">$4.7.9.1</a>
 * of the Java Virtual Machine Specification.
 */
@RunWith(JUnit4.class)
public class ClassSignatureParserTest {

  @Test
  public void superclass_noGenerics() {
    ClassSignature classSignature =
        ClassSignatureParser.readTypeParametersForInterfaces(
            "Foo", "Ljava/lang/Object;", "Object", new String[] {});

    assertThat(classSignature.superClassSignature().identifier()).isEqualTo("Ljava/lang/Object");
    assertThat(classSignature.superClassSignature().typeParameters()).isEmpty();
    assertThat(classSignature.typeParameters()).isEmpty();
    assertThat(classSignature.interfaceTypeParameters()).isEmpty();
  }

  @Test
  public void superClass_oneGenericParameter() {
    ClassSignature classSignature =
        ClassSignatureParser.readTypeParametersForInterfaces(
            "Foo",
            "Ljava/util/LinkedHashSet<Ljava/lang/String;>;",
            "LinkedHashSet",
            new String[] {});

    assertThat(classSignature.superClassSignature().identifier())
        .isEqualTo("Ljava/util/LinkedHashSet");
    assertThat(classSignature.superClassSignature().typeParameters())
        .isEqualTo("<Ljava/lang/String;>");
    assertThat(classSignature.typeParameters()).isEmpty();
    assertThat(classSignature.interfaceTypeParameters()).isEmpty();
  }

  @Test
  public void superClass_twoGenericParameters() {
    ClassSignature classSignature =
        ClassSignatureParser.readTypeParametersForInterfaces(
            "Foo",
            "Ljava/util/concurrent/ConcurrentSkipListMap<Ljava/lang/Integer;Ljava/lang/String;>;",
            "ConcurrentSkipListMap",
            new String[] {});

    assertThat(classSignature.superClassSignature().identifier())
        .isEqualTo("Ljava/util/concurrent/ConcurrentSkipListMap");
    assertThat(classSignature.superClassSignature().typeParameters())
        .isEqualTo("<Ljava/lang/Integer;Ljava/lang/String;>");
    assertThat(classSignature.typeParameters()).isEmpty();
    assertThat(classSignature.interfaceTypeParameters()).isEmpty();
  }

  @Test
  public void superClass_nestedClass() {
    ClassSignature classSignature =
        ClassSignatureParser.readTypeParametersForInterfaces(
            "Foo",
            "Lj$/util/stream/SpinedBuffer$OfPrimitive"
                + "<Ljava/lang/Double;[DLj$/util/function/DoubleConsumer;>"
                + ".BaseSpliterator<Lj$/util/Spliterator$OfDouble;>;"
                + "Lj$/util/Spliterator$OfDouble;",
            "BaseSpliterator",
            new String[] {"Spliterator$OfDouble"});

    assertThat(classSignature.superClassSignature().identifier())
        .isEqualTo(
            "Lj$/util/stream/SpinedBuffer$OfPrimitive"
                + "<Ljava/lang/Double;[DLj$/util/function/DoubleConsumer;>"
                + ".BaseSpliterator");
    assertThat(classSignature.superClassSignature().typeParameters())
        .isEqualTo("<Lj$/util/Spliterator$OfDouble;>");
    assertThat(classSignature.typeParameters()).isEmpty();
    assertThat(classSignature.interfaceTypeParameters()).containsExactly("");
  }

  @Test
  public void genericBounds_oneBound() {
    ClassSignature classSignature =
        ClassSignatureParser.readTypeParametersForInterfaces(
            "Foo",
            "<T:Ljava/lang/Object;>Ljava/util/AbstractList<TT;>;",
            "AbstractList",
            new String[] {});

    assertThat(classSignature.superClassSignature().identifier())
        .isEqualTo("Ljava/util/AbstractList");
    assertThat(classSignature.superClassSignature().typeParameters()).isEqualTo("<TT;>");
    assertThat(classSignature.typeParameters()).isEqualTo("<T:Ljava/lang/Object;>");
    assertThat(classSignature.interfaceTypeParameters()).isEmpty();
  }

  @Test
  public void genericBounds_twoBounds() {
    ClassSignature classSignature =
        ClassSignatureParser.readTypeParametersForInterfaces(
            "Foo",
            "<E:Ljava/lang/Number;T::Ljava/util/List<TE;>;>"
                + "Ljava/lang/Object;Ljava/util/List<TE;>;",
            "Object",
            new String[] {"List"});

    assertThat(classSignature.superClassSignature().identifier()).isEqualTo("Ljava/lang/Object");
    assertThat(classSignature.superClassSignature().typeParameters()).isEmpty();
    assertThat(classSignature.typeParameters())
        .isEqualTo("<E:Ljava/lang/Number;T::Ljava/util/List<TE;>;>");
    assertThat(classSignature.interfaceTypeParameters()).containsExactly("<TE;>");
  }

  @Test
  public void interfaces_oneImplementation() {
    ClassSignature classSignature =
        ClassSignatureParser.readTypeParametersForInterfaces(
            "Foo",
            "<T:Ljava/lang/Object;>Ljava/lang/Object;" + "Ljava/util/List<TT;>;",
            "Object",
            new String[] {"List"});

    assertThat(classSignature.superClassSignature().identifier()).isEqualTo("Ljava/lang/Object");
    assertThat(classSignature.superClassSignature().typeParameters()).isEmpty();
    assertThat(classSignature.typeParameters()).isEqualTo("<T:Ljava/lang/Object;>");
    assertThat(classSignature.interfaceTypeParameters()).containsExactly("<TT;>");
  }

  @Test
  public void interfaces_twoImplementations() {
    ClassSignature classSignature =
        ClassSignatureParser.readTypeParametersForInterfaces(
            "Foo",
            "<E:Ljava/lang/Number;T:Ljava/lang/Object;>Ljava/lang/Object;"
                + "Ljava/util/List<TT;>;Ljava/util/Set<TE;>;",
            "Object",
            new String[] {"List", "Set"});

    assertThat(classSignature.superClassSignature().identifier()).isEqualTo("Ljava/lang/Object");
    assertThat(classSignature.superClassSignature().typeParameters()).isEmpty();
    assertThat(classSignature.typeParameters())
        .isEqualTo("<E:Ljava/lang/Number;T:Ljava/lang/Object;>");
    assertThat(classSignature.interfaceTypeParameters())
        .containsExactly("<TT;>", "<TE;>")
        .inOrder();
  }

  @Test
  public void interface_nestedGenerics() {
    ClassSignature classSignature =
        ClassSignatureParser.readTypeParametersForInterfaces(
            "Foo",
            "Ljava/lang/Object;Ldagger/internal/Factory"
                + "<Lcom/google/android/libraries/storage/protostore/ProtoDataStore"
                + "<Lcom/google/android/apps/nbu/paisa/user/discover/search/searchconfigservice/SearchConfigDataStore;>;"
                + ">;",
            "Object",
            new String[] {"Factory"});

    assertThat(classSignature.superClassSignature().identifier()).isEqualTo("Ljava/lang/Object");
    assertThat(classSignature.superClassSignature().typeParameters()).isEmpty();
    assertThat(classSignature.typeParameters()).isEmpty();
    assertThat(classSignature.interfaceTypeParameters())
        .containsExactly(
            "<Lcom/google/android/libraries/storage/protostore/ProtoDataStore"
                + "<Lcom/google/android/apps/nbu/paisa/user/discover/search/searchconfigservice/SearchConfigDataStore;>;"
                + ">");
  }
}
