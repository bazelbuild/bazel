// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.testutil;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assert_;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;

import java.lang.ref.Reference;
import java.lang.reflect.Field;
import java.util.Comparator;
import java.util.Map;
import java.util.Queue;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * A helper class for tests providing a simple interface for asserts.
 */
public class MoreAsserts {

  public static void assertContentsAnyOrderOf(Iterable<?> actual, Object... expected) {
    assertThat(ImmutableList.copyOf(actual)).has().exactlyAs(ImmutableList.copyOf(expected));
  }

  @SafeVarargs
  public static <T> void assertContentsAnyOrder(Iterable<T> actual, T... expected) {
    assertThat(ImmutableList.copyOf(actual)).has().exactlyAs(ImmutableList.copyOf(expected));
  }

  @SafeVarargs
  public static <T> void assertContentsAnyOrder(String msg, Iterable<T> actual, T... expected) {
    assertThat(ImmutableList.copyOf(actual)).labeled(msg).has()
      .exactlyAs(ImmutableList.copyOf(expected));
  }

  public static <T> void assertContentsAnyOrder(Iterable<T> expected, Iterable<T> actual) {
    assertThat(ImmutableList.copyOf(actual)).has().exactlyAs(ImmutableList.copyOf(expected));
  }

  public static <T> void assertContentsAnyOrder(
      String msg, Iterable<T> expected, Iterable<T> actual) {
    assertThat(ImmutableList.copyOf(actual)).labeled(msg).has()
      .exactlyAs(ImmutableList.copyOf(expected));
  }

  @SafeVarargs
  public static <T> void assertContentsInOrder(Iterable<T> actual, T... expected) {
    assertThat(ImmutableList.copyOf(actual)).has()
      .exactlyAs(ImmutableList.copyOf(expected)).inOrder();
  }

  @SafeVarargs
  public static <T> void assertContentsInOrder(String msg, Iterable<T> actual, T... expected) {
    assertThat(ImmutableList.copyOf(actual)).labeled(msg).has()
      .exactlyAs(ImmutableList.copyOf(expected)).inOrder();
  }

  public static <T> void assertContentsInOrder(
      String msg, Iterable<T> expected, Iterable<T> actual) {
    assertThat(ImmutableList.copyOf(actual)).labeled(msg).has()
      .exactlyAs(ImmutableList.copyOf(expected)).inOrder();
  }

  public static <T> void assertContentsInOrder(Iterable<T> expected, Iterable<T> actual) {
    assertThat(ImmutableList.copyOf(actual)).has()
    .exactlyAs(ImmutableList.copyOf(expected)).inOrder();
  }

  public static void assertEmpty(Iterable<?> items) {
    assertThat(items).isEmpty();
  }

  public static void assertEmpty(String msg, Iterable<?> items) {
    assertThat(items).labeled(msg).isEmpty();
  }

  public static void assertEmpty(Map<?, ?> map) {
    assertThat(map).isEmpty();
  }

  public static void assertNotEmpty(String msg, Iterable<?> items) {
    assertThat(items).labeled(msg).isNotEmpty();
  }

  public static void assertNotEmpty(Iterable<?> items) {
    assertThat(items).isNotEmpty();
  }

  public static void assertNotEmpty(Map<?, ?> map) {
    assertThat(map).isNotEmpty();
  }

  public static void assertNotEqual(Object expected, Object actual) {
    assertThat(actual).isNotEqualTo(expected);
  }

  public static void assertNotEqual(String msg, Object expected, Object actual) {
    assertThat(actual).labeled(msg).isNotEqualTo(expected);
  }

  public static void assertContains(String expected, String actual) {
    assertThat(actual).contains(expected);
  }

  public static void assertNotContains(String expected, String actual) {
    assertThat(actual).doesNotContain(expected);
  }

  @SafeVarargs
  public static <T> void assertContains(Iterable<T> actual, T... expected) {
    assertThat(ImmutableList.copyOf(actual)).has().allFrom(ImmutableList.copyOf(expected));
  }
  
  public static <T> void assertNotContains(Iterable<T> actual, T unexpected) {
    for (T i : actual) {
      if (i.equals(unexpected)) {
        assert_().fail();
      }
    }
  }

  
  public static void assertContains(String msg, String expected, String actual) {
    assertThat(actual).labeled(msg).contains(expected);
  }

  public static void assertNotContains(String msg, String expected, String actual) {
    assertFalse(msg, actual.contains(expected));
  }

  @SafeVarargs
  public static <T> void assertContains(String msg, Iterable<T> actual, T... expected) {
    assertThat(ImmutableList.copyOf(actual)).labeled(msg)
        .has().allFrom(ImmutableList.copyOf(expected));
  }

  public static <T> void assertNotContains(String msg, Iterable<T> actual, T unexpected) {
    assertThat(ImmutableList.copyOf(actual)).labeled(msg)
        .has().noneOf(unexpected);
  }

  private static Matcher getMatcher(String regex, String actual) {
    Pattern pattern = Pattern.compile(regex);
    return pattern.matcher(actual);
  }

  public static void assertContainsRegex(String regex, String actual) {
    assertTrue(getMatcher(regex, actual).find());
  }

  public static void assertContainsRegex(String msg, String regex, String actual) {
    assertTrue(msg, getMatcher(regex, actual).find());
  }

  public static void assertNotContainsRegex(String regex, String actual) {
    assertFalse(actual.matches(regex));
  }

  public static void assertNotContainsRegex(String msg, String regex, String actual) {
    assertFalse(msg, getMatcher(regex, actual).find());
  }

  public static void assertMatchesRegex(String regex, String actual) {
    assertTrue(actual.matches(regex));
  }

  public static void assertMatchesRegex(String msg, String regex, String actual) {
    assertTrue(msg, actual.matches(regex));
  }

  public static void assertNotMatchesRegex(String regex, String actual) {
    assertFalse(actual.matches(regex));
  }

  public static <T> void assertEquals(T expected, T actual, Comparator<T> comp) {
    assertTrue(comp.compare(expected, actual) == 0);
  }

  public static <T> void assertContentsAnyOrder(
      Iterable<? extends T> expected, Iterable<? extends T> actual,
      Comparator<? super T> comp) {
    assertTrue(Iterables.size(expected) == Iterables.size(actual));
    int i = 0;
    for (T e : expected) {
      for (T a : actual) {
        if (comp.compare(e, a) == 0) {
          i++;
        }
      }
    }
    assertTrue(i == Iterables.size(actual));
  }

  public static void assertContainsNoDuplicates(Iterable<?> iterable) {
    ImmutableList<?> list = ImmutableList.copyOf(iterable);
    for (int i = 0; i < list.size(); i++) {
      for (int j = i + 1; j < list.size(); j++) {
        assertNotEqual(list.get(i), list.get(j));
      }
    }
  }

  public static void assertGreaterThanOrEqual(long target, long actual) {
    assertTrue(target <= actual);
  }

  public static void assertGreaterThanOrEqual(String msg, long target, long actual) {
    assertTrue(msg, target <= actual);
  }

  public static void assertGreaterThan(long target, long actual) {
    assertTrue(target < actual);
  }

  public static void assertGreaterThan(String msg, long target, long actual) {
    assertTrue(msg, target < actual);
  }

  public static void assertLessThanOrEqual(long target, long actual) {
    assertTrue(target >= actual);
  }

  public static void assertLessThanOrEqual(String msg, long target, long actual) {
    assertTrue(msg, target >= actual);
  }

  public static void assertLessThan(long target, long actual) {
    assertTrue(target > actual);
  }

  public static void assertLessThan(String msg, long target, long actual) {
    assertTrue(msg, target > actual);
  }

  public static void assertEndsWith(String ending, String actual) {
    assertThat(actual).endsWith(ending);
  }

  public static void assertStartsWith(String prefix, String actual) {
    assertThat(actual).startsWith(prefix);
  }

  /**
   * Scans if an instance of given class is strongly reachable from a given
   * object.
   * <p>Runs breadth-first search in object reachability graph to check if
   * an instance of <code>clz</code> can be reached.
   * <strong>Note:</strong> This method can take a long time if analyzed
   * data structure spans across large part of heap and may need a lot of
   * memory.
   *
   * @param start object to start the search from
   * @param clazz class to look for
   */
  public static void assertInstanceOfNotReachable(
      Object start, final Class<?> clazz) {
    Predicate<Object> p = new Predicate<Object>() {
        @Override
        public boolean apply(Object obj) {
          return clazz.isAssignableFrom(obj.getClass());
        }
      };
    if (isRetained(p, start)) {
      assert_().fail("Found an instance of " + clazz.getCanonicalName() +
          " reachable from " + start.toString());
    }
  }

  private static final Field NON_STRONG_REF; 
  
  static {
    try {
      NON_STRONG_REF = Reference.class.getDeclaredField("referent");
    } catch (SecurityException e) {
      throw new RuntimeException(e);
    } catch (NoSuchFieldException e) {
      throw new RuntimeException(e);
    }
  }

  static final Predicate<Field> ALL_STRONG_REFS = new Predicate<Field>() {
    @Override
    public boolean apply(Field field) {
      return NON_STRONG_REF.equals(field);
    }
  };
  
  private static boolean isRetained(Predicate<Object> predicate, Object start) {
    Map<Object, Object> visited = Maps.newIdentityHashMap();
    visited.put(start, start);
    Queue<Object> toScan = Lists.newLinkedList();
    toScan.add(start);

    while (!toScan.isEmpty()) {
      Object current = toScan.poll();
      if (current.getClass().isArray()) {
        if (current.getClass().getComponentType().isPrimitive()) {
          continue;
        }

        for (Object ref : (Object[]) current) {
          if (ref != null) {
            if (predicate.apply(ref)) {
              return true;
            }
            if (visited.put(ref, ref) == null) {
              toScan.add(ref);
            }
          }
        }
      } else {
        // iterate *all* fields (getFields() returns only accessible ones)
        for (Class<?> clazz = current.getClass(); clazz != null;
             clazz = clazz.getSuperclass()) {
          for (Field f : clazz.getDeclaredFields()) {
            if (f.getType().isPrimitive() || ALL_STRONG_REFS.apply(f)) {
              continue;
            }

            f.setAccessible(true);
            try {
              Object ref = f.get(current);
              if (ref != null) {
                if (predicate.apply(ref)) {
                  return true;
                }
                if (visited.put(ref, ref) == null) {
                  toScan.add(ref);
                }
              }
            } catch (IllegalArgumentException e) {
              throw new IllegalStateException("Error when scanning the heap", e);
            } catch (IllegalAccessException e) {
              throw new IllegalStateException("Error when scanning the heap", e);
            }
          }
        }
      }
    }
    return false;
  }
}
