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
}
