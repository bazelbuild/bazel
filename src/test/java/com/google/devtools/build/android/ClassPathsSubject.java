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
package com.google.devtools.build.android;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.base.Function;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.common.truth.FailureMetadata;
import com.google.common.truth.Subject;
import java.io.IOException;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import javax.annotation.Nullable;

/**
 * A testing utility that allows .java/.class related assertions against Paths.
 */
public class ClassPathsSubject extends Subject<ClassPathsSubject, Path> {

  ClassPathsSubject(FailureMetadata failureMetadata, @Nullable Path subject) {
    super(failureMetadata, subject);
  }

  void exists() {
    if (getSubject() == null) {
      fail("should not be null.");
    }
    if (!Files.exists(getSubject())) {
      fail("exists.");
    }
  }

  /**
   * Check that the contents of the java file at the current path, is equivalent to the given
   * expected contents (modulo header comments and surrounding whitespace).
   *
   * @param contents expected contents
   */
  public void javaContentsIsEqualTo(String... contents) {
    if (getSubject() == null) {
      fail("should not be null.");
    }
    exists();
    try {
      assertThat(
          trimWhitespace(
              stripJavaHeaderComments(
                  Files.readAllLines(getSubject(), StandardCharsets.UTF_8))))
          .containsExactly((Object[]) contents)
          .inOrder();
    } catch (IOException e) {
      fail(e.toString());
    }
  }

  private List<String> stripJavaHeaderComments(List<String> strings) {
    List<String> result = new ArrayList<>();
    boolean inComment = false;
    for (String string : strings) {
      if (string.trim().startsWith("/*")) {
        inComment = true;
        continue;
      }
      if (inComment) {
        if (string.trim().startsWith("*/")) {
          inComment = false;
          continue;
        }
        continue;
      }
      result.add(string);
    }
    return result;
  }

  private List<String> trimWhitespace(List<String> strings) {
    return Lists.transform(strings, new Function<String, String>() {
      @Override
      public String apply(String s) {
        return s.trim();
      }
    });
  }

  /**
   * Check the class file with the given name, assuming the current path is part of the classpath.
   *
   * @param className the fully qualified class name
   */
  public ClassNameSubject withClass(String className) {
    if (getSubject() == null) {
      fail("should not be null.");
    }
    exists();
    return check().about(ClassNameSubject.classNames(getSubject())).that(className);
  }

  static final class ClassNameSubject extends Subject<ClassNameSubject, String> {

    private final Path basePath;

    public ClassNameSubject(FailureMetadata failureMetadata, Path basePath, String subject) {
      super(failureMetadata, subject);
      this.basePath = basePath;
    }

    static Subject.Factory<ClassNameSubject, String> classNames(Path basePath) {
      return new Subject.Factory<ClassNameSubject, String>() {
        @Override
        public ClassNameSubject createSubject(FailureMetadata metadata, String actual) {
          return new ClassNameSubject(metadata, basePath, actual);
        }
      };
    }

    public void classContentsIsEqualTo(
        ImmutableMap<String, Integer> intFields,
        ImmutableMap<String, List<Integer>> intArrayFields,
        boolean areFieldsFinal) throws Exception {
      String expectedClassName = getSubject();
      URLClassLoader urlClassLoader = new URLClassLoader(new URL[]{basePath.toUri().toURL()});
      Class<?> innerClass = urlClassLoader.loadClass(expectedClassName);
      assertThat(innerClass.getSuperclass()).isEqualTo(Object.class);
      assertThat(innerClass.getEnclosingClass().toString())
          .endsWith(expectedClassName.substring(0, expectedClassName.indexOf('$')));
      ImmutableMap.Builder<String, Integer> actualIntFields = ImmutableMap.builder();
      ImmutableMap.Builder<String, List<Integer>> actualIntArrayFields = ImmutableMap.builder();
      for (Field f : innerClass.getFields()) {
        int fieldModifiers = f.getModifiers();
        assertThat(Modifier.isFinal(fieldModifiers)).isEqualTo(areFieldsFinal);
        assertThat(Modifier.isPublic(fieldModifiers)).isTrue();
        assertThat(Modifier.isStatic(fieldModifiers)).isTrue();

        Class<?> fieldType = f.getType();
        if (fieldType.isPrimitive()) {
          assertThat(fieldType).isEqualTo(Integer.TYPE);
          actualIntFields.put(f.getName(), (Integer) f.get(null));
        } else {
          assertThat(fieldType.isArray()).isTrue();
          int[] asArray = (int[]) f.get(null);
          ImmutableList.Builder<Integer> list = ImmutableList.builder();
          for (int i : asArray) {
            list.add(i);
          }
          actualIntArrayFields.put(f.getName(), list.build());
        }
      }
      assertThat(actualIntFields.build()).containsExactlyEntriesIn(intFields).inOrder();
      assertThat(actualIntArrayFields.build()).containsExactlyEntriesIn(intArrayFields).inOrder();
    }
  }
}
