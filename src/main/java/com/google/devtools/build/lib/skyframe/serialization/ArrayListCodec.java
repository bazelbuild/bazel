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

import com.google.common.collect.ImmutableList;
import java.util.ArrayList;
import java.util.List;

class ArrayListCodec<T> extends NullableListCodec<T> {
  @SuppressWarnings("unchecked")
  @Override
  public Class<List<T>> getEncodedClass() {
    return (Class<List<T>>) (Class<?>) ArrayList.class;
  }

  @Override
  public List<Class<? extends List<T>>> additionalEncodedClasses() {
    return ImmutableList.of();
  }

  @Override
  protected List<T> maybeTransform(ArrayList<T> startingList) {
    return startingList;
  }
}
