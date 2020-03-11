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

package com.google.devtools.build.android.desugar.langmodel;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.collect.ImmutableMap.toImmutableMap;
import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static java.util.stream.Collectors.toCollection;

import com.google.common.collect.ConcurrentHashMultiset;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import java.util.function.Function;
import org.objectweb.asm.commons.Remapper;

/** Maps a type to another based on binary names. */
public final class TypeMapper extends Remapper {

  private final Function<ClassName, ClassName> classNameMapper;

  public TypeMapper(Function<ClassName, ClassName> classNameMapper) {
    this.classNameMapper = classNameMapper;
  }

  @Override
  public String map(String binaryName) {
    return map(ClassName.create(binaryName)).binaryName();
  }

  public ClassName map(ClassName internalName) {
    return classNameMapper.apply(internalName);
  }

  public <E extends TypeMappable<E>> ImmutableList<? extends E> map(
      ImmutableList<E> mappableTypes) {
    return mappableTypes.stream().map(e -> e.acceptTypeMapper(this)).collect(toImmutableList());
  }

  public <E extends TypeMappable<E>> ImmutableSet<? extends E> map(ImmutableSet<E> mappableTypes) {
    return mappableTypes.stream().map(e -> e.acceptTypeMapper(this)).collect(toImmutableSet());
  }

  public <E extends TypeMappable<E>> ConcurrentHashMultiset<E> map(
      ConcurrentHashMultiset<E> mappableTypes) {
    return mappableTypes.stream()
        .map(e -> e.acceptTypeMapper(this))
        .collect(toCollection(ConcurrentHashMultiset::create));
  }

  public <K extends TypeMappable<K>, V extends TypeMappable<V>> ImmutableMap<K, V> map(
      ImmutableMap<K, V> mappableTypes) {
    return mappableTypes.entrySet().stream()
        .collect(
            toImmutableMap(
                e -> e.getKey().acceptTypeMapper(this), e -> e.getValue().acceptTypeMapper(this)));
  }

  public <K extends TypeMappable<? extends K>, V> ImmutableMap<K, V> mapKey(
      ImmutableMap<K, V> mappableTypes) {
    return mappableTypes.entrySet().stream()
        .collect(toImmutableMap(e -> e.getKey().acceptTypeMapper(this), e -> e.getValue()));
  }

}
