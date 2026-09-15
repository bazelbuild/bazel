// Copyright 2021 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.vfs;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import java.util.Iterator;
import java.util.NoSuchElementException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link PathSegmentIterator}. */
@RunWith(JUnit4.class)
public final class PathSegmentIteratorTest {

  @Test
  public void emptyPath() {
    assertThat(segmentIterator("", /*driveStrLength=*/ 0)).isEmpty();
  }

  @Test
  public void relativePath() {
    assertThat(segmentIterator("this/is/a/relative/path", /*driveStrLength=*/ 0))
        .containsExactly("this", "is", "a", "relative", "path")
        .inOrder();
  }

  @Test
  public void root_unix() {
    assertThat(segmentIterator("/", /*driveStrLength=*/ 1)).isEmpty();
  }

  @Test
  public void root_windows() {
    assertThat(segmentIterator("C:/", /*driveStrLength=*/ 3)).isEmpty();
  }

  @Test
  public void absolutePath_unix() {
    assertThat(segmentIterator("/this/is/an/absolute/path", /*driveStrLength=*/ 1))
        .containsExactly("this", "is", "an", "absolute", "path")
        .inOrder();
  }

  @Test
  public void absolutePath_windows() {
    assertThat(segmentIterator("C:/this/is/an/absolute/path", /*driveStrLength=*/ 3))
        .containsExactly("this", "is", "an", "absolute", "path")
        .inOrder();
  }

  @Test
  public void noSuchElement() {
    Iterator<String> it = PathSegmentIterator.create("some/path", /*driveStrLength=*/ 0);
    it.next();
    it.next();
    assertThrows(NoSuchElementException.class, it::next);
  }

  private static Iterable<String> segmentIterator(String normalizedPath, int driveStrLength) {
    return () -> PathSegmentIterator.create(normalizedPath, driveStrLength);
  }
}
