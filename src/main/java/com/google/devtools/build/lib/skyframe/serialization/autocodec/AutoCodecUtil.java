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

package com.google.devtools.build.lib.skyframe.serialization.autocodec;

import com.google.common.collect.ImmutableList;
import java.util.stream.Collectors;
import javax.lang.model.element.Element;
import javax.lang.model.element.TypeElement;

/** Static utilities for AutoCodec processors. */
class AutoCodecUtil {
  // Synthesized classes will have `_AutoCodec` suffix added.
  public static final String GENERATED_CLASS_NAME_SUFFIX = "AutoCodec";
  static final Class<AutoCodec> ANNOTATION = AutoCodec.class;

  /**
   * Name of the generated codec class.
   *
   * <p>For {@code Foo.Bar} this is {@code Foo_Bar_AutoCodec}.
   */
  static String getCodecName(Element element) {
    ImmutableList.Builder<String> classNamesBuilder = new ImmutableList.Builder<>();
    classNamesBuilder.add(GENERATED_CLASS_NAME_SUFFIX);
    do {
      classNamesBuilder.add(element.getSimpleName().toString());
      element = element.getEnclosingElement();
    } while (element instanceof TypeElement);
    return classNamesBuilder.build().reverse().stream().collect(Collectors.joining("_"));
  }
}
