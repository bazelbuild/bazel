// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import java.util.Iterator;

/**
 * Base class for tiny container types that encapsulate an iterable.
 */
abstract class IterableWrapper<E> implements Iterable<E> {
  private final Iterable<E> contents;

  IterableWrapper(Iterable<E> contents) {
    this.contents = Preconditions.checkNotNull(contents);
  }

  IterableWrapper(E... contents) {
    this.contents = ImmutableList.copyOf(contents);
  }

  @Override
  public Iterator<E> iterator() {
    return contents.iterator();
  }
}
