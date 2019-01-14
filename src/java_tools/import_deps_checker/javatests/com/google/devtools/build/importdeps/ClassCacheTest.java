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
package com.google.devtools.build.importdeps;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.importdeps.AbstractClassEntryState.ExistingState;
import com.google.devtools.build.importdeps.ClassCache.LazyClassEntry;
import com.google.devtools.build.importdeps.ClassCache.LazyClasspath;
import com.google.devtools.build.importdeps.ClassInfo.MemberInfo;
import java.io.IOException;
import java.util.Optional;
import java.util.stream.Collectors;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test for {@link ClassCache}. */
@RunWith(JUnit4.class)
public class ClassCacheTest extends AbstractClassCacheTest {

  @Test
  public void testLibraryJar() throws Exception {
    try (ClassCache cache =
        new ClassCache(
            ImmutableSet.of(bootclasspath),
            ImmutableSet.of(),
            ImmutableSet.of(libraryJar, libraryInterfaceJar),
            ImmutableSet.of())) {
      assertCache(
          cache,
          libraryJarPositives,
          combine(libraryAnnotationsJarPositives, libraryExceptionJarPositives));
    }
  }

  @Test
  public void testClientJarWithSuperClasses() throws IOException {
    try (ClassCache cache =
        new ClassCache(
            ImmutableSet.of(bootclasspath),
            ImmutableSet.of(libraryJar, libraryInterfaceJar),
            ImmutableSet.of(libraryJar, libraryInterfaceJar),
            ImmutableSet.of(clientJar))) {
      assertCache(
          cache,
          clientJarPositives,
          combine(libraryExceptionJarPositives, libraryAnnotationsJarPositives));
    }
  }

  @Test
  public void testClientJarWithoutSuperClasses() throws IOException {
    try (ClassCache cache =
        new ClassCache(
            ImmutableSet.of(bootclasspath),
            ImmutableSet.of(),
            ImmutableSet.of(),
            ImmutableSet.of(clientJar))) {
      // Client should be incomplete, as its parent class and interfaces are not available on the
      // classpath. The following is the resolution path.
      {
        AbstractClassEntryState state = cache.getClassState(PACKAGE_NAME + "Client");
        assertThat(state.isIncompleteState()).isTrue();

        ImmutableSet<String> successPart =
            state.asIncompleteState().resolutionFailureChain().getMissingClassesWithSubclasses()
                .values().stream()
                .distinct()
                .sorted()
                .map(ClassInfo::internalName)
                .collect(ImmutableSet.toImmutableSet());
        assertThat(successPart).containsExactly(PACKAGE_NAME + "Client").inOrder();
        assertThat(state.asIncompleteState().missingAncestors())
            .containsExactly(
                PACKAGE_NAME + "Library",
                PACKAGE_NAME + "LibraryInterface",
                PACKAGE_NAME + "LibraryInterface$One",
                PACKAGE_NAME + "LibraryInterface$Two");
      }
      assertThat(cache.getClassState(PACKAGE_NAME + "Client").isIncompleteState()).isTrue();
      assertThat(cache.getClassState(PACKAGE_NAME + "Client$NestedAnnotation"))
          .isInstanceOf(ExistingState.class);
      assertThat(cache.getClassState(PACKAGE_NAME + "Client$NestedAnnotation").isExistingState())
          .isTrue();
      assertThat(cache.getClassState("java/lang/Object").isExistingState()).isTrue();
      assertThat(cache.getClassState("java/util/List").isExistingState()).isTrue();
    }
  }

  @Test
  public void testLibraryException() throws IOException {
    try (ClassCache cache =
        new ClassCache(
            ImmutableSet.of(bootclasspath),
            ImmutableSet.of(),
            ImmutableSet.of(),
            ImmutableSet.of(libraryExceptionJar))) {
      assertCache(
          cache,
          libraryExceptionJarPositives,
          combine(libraryAnnotationsJarPositives, libraryInterfacePositives, libraryJarPositives));
    }
  }

  @Test
  public void testLibraryAnnotations() throws IOException {
    try (ClassCache cache =
        new ClassCache(
            ImmutableSet.of(bootclasspath),
            ImmutableSet.of(),
            ImmutableSet.of(),
            ImmutableSet.of(libraryAnnotationsJar))) {
      assertCache(
          cache,
          libraryAnnotationsJarPositives,
          combine(libraryExceptionJarPositives, libraryInterfacePositives, libraryJarPositives));
    }
  }

  @Test
  public void testCannotAccessClosedCache() throws IOException {
    ClassCache cache =
        new ClassCache(ImmutableSet.of(), ImmutableSet.of(), ImmutableSet.of(), ImmutableSet.of());
    cache.close();
    cache.close(); // Can close multiple times.
    assertThrows(IllegalStateException.class, () -> cache.getClassState("empty"));
  }

  /**
   * A regression test. First query the super class, which does not exist. Then query the subclass,
   * which does not exist either.
   */
  @Test
  public void testSuperNotExistThenSubclassNotExist() throws IOException {
    try (ClassCache cache =
        new ClassCache(
            ImmutableSet.of(),
            ImmutableSet.of(libraryJar, libraryJar, libraryAnnotationsJar, libraryInterfaceJar),
            ImmutableSet.of(libraryJar, libraryJar, libraryAnnotationsJar, libraryInterfaceJar),
            ImmutableSet.of(clientJar))) {
      assertThat(
              cache
                  .getClassState("com/google/devtools/build/importdeps/testdata/Library$Class9")
                  .isIncompleteState())
          .isTrue();
      assertThat(
              cache
                  .getClassState("com/google/devtools/build/importdeps/testdata/Library$Class10")
                  .isIncompleteState())
          .isTrue();
    }
  }

  @Test
  public void testJdepsOutput() throws IOException {
    try (ClassCache cache =
        new ClassCache(
            ImmutableSet.of(),
            ImmutableSet.of(libraryJar, libraryJar, libraryAnnotationsJar, libraryInterfaceJar),
            ImmutableSet.of(libraryJar, libraryJar, libraryAnnotationsJar, libraryInterfaceJar),
            ImmutableSet.of(clientJar))) {
      assertThat(
              cache
                  .getClassState("com/google/devtools/build/importdeps/testdata/Library$Class9")
                  .isIncompleteState())
          .isTrue();
      assertThat(cache.collectUsedJarsInRegularClasspath()).containsExactly(libraryJar);

      assertThat(
              cache
                  .getClassState("com/google/devtools/build/importdeps/testdata/LibraryAnnotations")
                  .isIncompleteState())
          .isTrue();
      assertThat(cache.collectUsedJarsInRegularClasspath())
          .containsExactly(libraryJar, libraryAnnotationsJar);

      assertThat(
              cache
                  .getClassState("com/google/devtools/build/importdeps/testdata/LibraryInterface")
                  .isIncompleteState())
          .isTrue();
      assertThat(cache.collectUsedJarsInRegularClasspath())
          .containsExactly(libraryJar, libraryAnnotationsJar, libraryInterfaceJar);
    }
  }

  @Test
  public void testLazyClasspathLoadsBootclasspathFirst() throws IOException {
    {
      LazyClasspath classpath =
          new LazyClasspath(
              ImmutableSet.of(libraryJar),
              ImmutableSet.of(libraryWoMembersJar),
              ImmutableSet.of(libraryWoMembersJar),
              ImmutableSet.of());
      LazyClassEntry entry =
          classpath.getLazyEntry("com/google/devtools/build/importdeps/testdata/Library$Class4");
      AbstractClassEntryState state = entry.getState(classpath);
      Optional<ClassInfo> classInfo = state.classInfo();
      assertThat(classInfo.isPresent()).isTrue();
      assertThat(classInfo.get().declaredMembers()).hasSize(2);
      assertThat(
              classInfo
                  .get()
                  .declaredMembers()
                  .stream()
                  .map(MemberInfo::memberName)
                  .collect(Collectors.toSet()))
          .containsExactly("<init>", "createClass5");
    }
    {
      LazyClasspath classpath =
          new LazyClasspath(
              ImmutableSet.of(libraryWoMembersJar),
              ImmutableSet.of(libraryJar),
              ImmutableSet.of(libraryJar),
              ImmutableSet.of());
      LazyClassEntry entry =
          classpath.getLazyEntry("com/google/devtools/build/importdeps/testdata/Library$Class4");
      AbstractClassEntryState state = entry.getState(classpath);
      Optional<ClassInfo> classInfo = state.classInfo();
      assertThat(classInfo.isPresent()).isTrue();
      // The empty class has a default constructor.
      assertThat(classInfo.get().declaredMembers()).hasSize(1);
      assertThat(classInfo.get().declaredMembers().iterator().next().memberName())
          .isEqualTo("<init>");
    }
  }

  @Test
  public void testOriginInformation() throws IOException {
    try (ClassCache cache =
        new ClassCache(
            ImmutableSet.of(bootclasspath),
            ImmutableSet.of(libraryJar),
            ImmutableSet.of(libraryJar, libraryInterfaceJar),
            ImmutableSet.of(clientJar))) {
      assertCache(
          cache,
          clientJarPositives,
          combine(libraryExceptionJarPositives, libraryAnnotationsJarPositives));

      {
        AbstractClassEntryState state = cache.getClassState(PACKAGE_NAME + "Library");
        assertThat(state.isExistingState()).isTrue();
        assertThat(state.classInfo().isPresent()).isTrue();
        assertThat(state.classInfo().get().directDep()).isTrue();
        assertThat((Object) state.classInfo().get().jarPath()).isEqualTo(libraryJar);
      }
      {
        AbstractClassEntryState state = cache.getClassState(PACKAGE_NAME + "LibraryInterface");
        assertThat(state.isExistingState()).isTrue();
        assertThat(state.classInfo().isPresent()).isTrue();
        assertThat(state.classInfo().get().directDep()).isFalse();
        assertThat((Object) state.classInfo().get().jarPath()).isEqualTo(libraryInterfaceJar);
      }
    }
  }

  private void assertCache(
      ClassCache cache, ImmutableList<String> positives, ImmutableList<String> negatives) {
    for (String positive : positives) {
      AbstractClassEntryState state = cache.getClassState(positive);
      assertWithMessage(positive).that(state.isExistingState()).isTrue();
      assertWithMessage(positive).that(state.asExistingState()).isInstanceOf(ExistingState.class);
      assertWithMessage(positive)
          .that(state.asExistingState().classInfo().get().internalName())
          .isEqualTo(positive);
    }
    for (String negative : negatives) {
      AbstractClassEntryState state = cache.getClassState(negative);
      assertWithMessage(negative).that(state.isExistingState()).isFalse();
    }
  }
}
