/*
 * Copyright 2020 The Bazel Authors. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.devtools.build.android.desugar.io;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.android.desugar.langmodel.ClassName;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for {@link FileBasedTypeReferenceClosure}.
 *
 * <p>The set of tests is based on the following type reference graph. For example A(lpha) contains
 * direct type symbol references to B(ravo) and C(harlie).
 *
 * <pre>
 *     A
 *     ↓ ↘
 *     B  C
 *      ↘ ↕
 *        D
 *        ↓
 *        E
 * </pre>
 */
@RunWith(JUnit4.class)
public class FileBasedTypeReferenceClosureTest {

  private final ClassName alpha = ClassName.create("desugar/io/testlib/Alpha");
  private final ClassName bravo = ClassName.create("desugar/io/testlib/Bravo");
  private final ClassName charlie = ClassName.create("desugar/io/testlib/Charlie");
  private final ClassName delta = ClassName.create("desugar/io/testlib/Delta");
  private final ClassName echo = ClassName.create("desugar/io/testlib/Echo");
  private final ClassName implicitTypeRef =
      ClassName.create("desugar/io/testlib/ImplicitTypeReferenceSource");

  @Test
  public void findReachableReferencedTypes_fromEmpty() {
    FileBasedTypeReferenceClosure fileBasedTypeReferenceClosure =
        new FileBasedTypeReferenceClosure(
            className -> className.hasAnyPackagePrefix("desugar/io/testlib/"),
            new ResourceBasedClassFiles());
    assertThat(fileBasedTypeReferenceClosure.findReachableReferencedTypes(ImmutableSet.of()))
        .isEmpty();
  }

  @Test
  public void findReachableReferencedTypes_fromAlpha() {
    FileBasedTypeReferenceClosure fileBasedTypeReferenceClosure =
        new FileBasedTypeReferenceClosure(
            className -> className.hasAnyPackagePrefix("desugar/io/testlib/"),
            new ResourceBasedClassFiles());
    assertThat(fileBasedTypeReferenceClosure.findReachableReferencedTypes(ImmutableSet.of(alpha)))
        .containsExactly(alpha, bravo, charlie, delta, echo);
  }

  @Test
  public void findReachableReferencedTypes_fromBravo() {
    FileBasedTypeReferenceClosure fileBasedTypeReferenceClosure =
        new FileBasedTypeReferenceClosure(
            className -> className.hasAnyPackagePrefix("desugar/io/testlib/"),
            new ResourceBasedClassFiles());
    assertThat(fileBasedTypeReferenceClosure.findReachableReferencedTypes(ImmutableSet.of(bravo)))
        .containsExactly(bravo, charlie, delta, echo);
  }

  @Test
  public void findReachableReferencedTypes_fromCharlie() {
    FileBasedTypeReferenceClosure fileBasedTypeReferenceClosure =
        new FileBasedTypeReferenceClosure(
            className -> className.hasAnyPackagePrefix("desugar/io/testlib/"),
            new ResourceBasedClassFiles());
    assertThat(fileBasedTypeReferenceClosure.findReachableReferencedTypes(ImmutableSet.of(charlie)))
        .containsExactly(charlie, delta, echo);
  }

  @Test
  public void findReachableReferencedTypes_fromDelta() {
    FileBasedTypeReferenceClosure fileBasedTypeReferenceClosure =
        new FileBasedTypeReferenceClosure(
            className -> className.hasAnyPackagePrefix("desugar/io/testlib/"),
            new ResourceBasedClassFiles());
    assertThat(fileBasedTypeReferenceClosure.findReachableReferencedTypes(ImmutableSet.of(delta)))
        .containsExactly(charlie, delta, echo);
  }

  @Test
  public void findReachableReferencedTypes_fromEcho() {
    FileBasedTypeReferenceClosure fileBasedTypeReferenceClosure =
        new FileBasedTypeReferenceClosure(
            className -> className.hasAnyPackagePrefix("desugar/io/testlib/"),
            new ResourceBasedClassFiles());
    assertThat(fileBasedTypeReferenceClosure.findReachableReferencedTypes(ImmutableSet.of(echo)))
        .containsExactly(echo);
  }

  @Test
  public void findReachableReferencedTypes_fromAll() {
    FileBasedTypeReferenceClosure fileBasedTypeReferenceClosure =
        new FileBasedTypeReferenceClosure(
            className -> className.hasAnyPackagePrefix("desugar/io/testlib/"),
            new ResourceBasedClassFiles());
    assertThat(
            fileBasedTypeReferenceClosure.findReachableReferencedTypes(
                ImmutableSet.of(alpha, bravo, charlie, delta, echo)))
        .containsExactly(alpha, bravo, charlie, delta, echo);
  }

  @Test
  public void findReachableReferencedTypes_implicitTypeReference() {
    FileBasedTypeReferenceClosure fileBasedTypeReferenceClosure =
        new FileBasedTypeReferenceClosure(
            className -> className.hasAnyPackagePrefix("desugar/io/testlib/", "java/io/"),
            new ResourceBasedClassFiles());
    ImmutableSet<ClassName> reachableReferencedTypes =
        fileBasedTypeReferenceClosure.findReachableReferencedTypes(
            ImmutableSet.of(implicitTypeRef));
    assertThat(reachableReferencedTypes).contains(ClassName.create("java/io/PrintStream"));
  }
}
