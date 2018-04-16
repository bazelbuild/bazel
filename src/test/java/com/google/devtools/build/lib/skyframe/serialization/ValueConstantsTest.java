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

package com.google.devtools.build.lib.skyframe.serialization;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.common.collect.ImmutableList;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ValueConstants}. */
@RunWith(JUnit4.class)
public class ValueConstantsTest {
  @Test
  public void smoke() {
    Object obj = new Object();
    String constString = "const";
    List<String> stringList = ImmutableList.of("element");
    ValueConstants underTest =
        new ValueConstants.Builder()
            .addSimpleConstant(obj)
            .addSimpleConstant(constString)
            .addCollectionConstant(stringList, String.class)
            .addSimpleConstant(ImmutableList.of())
            .build(7);

    assertThat(underTest.getNextTag()).isEqualTo(11);
    for (Object item : ImmutableList.of(obj, constString, ImmutableList.of(), stringList)) {
      Integer tag = underTest.maybeGetTagForConstant(item);
      assertWithMessage("%s should have had tag", item).that(tag).isNotNull();
      assertThat(underTest.maybeGetConstantByTag(tag)).isSameAs(item);
    }
    assertThat(underTest.maybeGetTagForConstant("aconst".substring(1))).isEqualTo(8);
    assertThat(underTest.maybeGetTagForConstant(ImmutableList.of("element")))
        .isEqualTo(underTest.maybeGetTagForConstant(stringList));
  }

  @Test
  public void cacheMisses() {
    ValueConstants underTest = new ValueConstants.Builder().addSimpleConstant("str").build(5);
    assertThat(underTest.maybeGetTagForConstant("newstr")).isNull();
    assertThat(underTest.maybeGetConstantByTag(4)).isNull();
    assertThat(underTest.maybeGetConstantByTag(6)).isNull();
  }

  @Test
  public void collectionWithAllNullsCantBeAdded() {
    ValueConstants.Builder underTest = new ValueConstants.Builder();
    List<Object> listWithOnlyNulls = new ArrayList<>();
    listWithOnlyNulls.add(null);
    assertThrows(
        NullPointerException.class,
        () -> underTest.addCollectionConstant(listWithOnlyNulls, Object.class));
  }

  @Test
  public void collectionWithMismatchTypeCantBeAdded() {
    ValueConstants.Builder underTest = new ValueConstants.Builder();
    List<Object> listWithOnlyNulls = new ArrayList<>();
    // Even though String is a valid element, the lookup scheme used by ValueConstants means that
    // this list would never be found during lookup, because the first non-null element is used to
    // determine the class.
    listWithOnlyNulls.add("str");
    assertThrows(
        IllegalStateException.class,
        () -> underTest.addCollectionConstant(listWithOnlyNulls, Object.class));
  }

  @Test
  public void collectionWithNullFirstOk() {
    ValueConstants.Builder builder = new ValueConstants.Builder();
    List<String> listWithANull = new ArrayList<>();
    listWithANull.add(null);
    listWithANull.add("str");
    builder.addCollectionConstant(listWithANull, String.class);
    ValueConstants constants = builder.build(0);
    List<String> newList = new ArrayList<>();
    newList.add(null);
    newList.add("str");
    assertThat(constants.maybeGetTagForConstant(newList)).isEqualTo(0);
  }

  @Test
  public void collectionWithWrongTypeDoesntTryToHit() {
    Collection<Object> badCollection = new BadCollection<>(ImmutableList.of(new Object()));
    ValueConstants underTest =
        new ValueConstants.Builder()
            .addCollectionConstant(ImmutableList.of("str"), String.class)
            .build(0);
    assertThat(underTest.maybeGetTagForConstant(badCollection)).isNull();
  }

  @Test
  public void collectionWithWrongSizeDoesntTryToHit() {
    @SuppressWarnings("unchecked")
    Collection<String> badCollection = new BadCollection<>(ImmutableList.of("str1", "str2"));
    ValueConstants underTest =
        new ValueConstants.Builder()
            .addCollectionConstant(ImmutableList.of("str"), String.class)
            .build(0);
    assertThat(underTest.maybeGetTagForConstant(badCollection)).isNull();
  }

  private static class BadCollection<T> implements Collection<T> {
    private final Collection<T> delegate;

    BadCollection(Collection<T> delegate) {
      this.delegate = delegate;
    }

    @Override
    public int size() {
      return delegate.size();
    }

    @Override
    public boolean isEmpty() {
      return delegate.isEmpty();
    }

    @Override
    public Iterator<T> iterator() {
      return delegate.iterator();
    }

    @Override
    public int hashCode() {
      throw new UnsupportedOperationException("Shouldn't try to hash");
    }

    @Override
    public boolean equals(Object obj) {
      throw new UnsupportedOperationException("Shouldn't try to compare equal");
    }

    @Override
    public boolean contains(Object o) {
      throw new UnsupportedOperationException();
    }

    @Override
    public Object[] toArray() {
      throw new UnsupportedOperationException();
    }

    @Override
    public <T1> T1[] toArray(T1[] a) {
      throw new UnsupportedOperationException();
    }

    @Override
    public boolean add(T t) {
      throw new UnsupportedOperationException();
    }

    @Override
    public boolean remove(Object o) {
      throw new UnsupportedOperationException();
    }

    @Override
    public boolean containsAll(Collection<?> c) {
      throw new UnsupportedOperationException();
    }

    @Override
    public boolean addAll(Collection<? extends T> c) {
      throw new UnsupportedOperationException();
    }

    @Override
    public boolean removeAll(Collection<?> c) {
      throw new UnsupportedOperationException();
    }

    @Override
    public boolean retainAll(Collection<?> c) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void clear() {
      throw new UnsupportedOperationException();
    }
  }
}
