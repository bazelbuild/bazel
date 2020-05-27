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

package com.google.devtools.build.android.desugar.corelibadapter;

import com.google.auto.value.AutoValue;
import com.google.devtools.build.android.desugar.langmodel.MethodKey;

/** Tracks the rationale that the desugar tool chooses to transform an method invocation site. */
@AutoValue
abstract class InvocationSiteTransformationReason {

  enum InvocationSiteTransformationKind {
    INLINE_PARAM_TYPE_CONVERSION,
    TYPE_ADAPTER_REPLACEMENT,
  }

  abstract InvocationSiteTransformationKind kind();

  abstract MethodKey method();

  public static InvocationSiteTransformationReason create(
      InvocationSiteTransformationKind logReason, MethodKey method) {
    return new AutoValue_InvocationSiteTransformationReason(logReason, method);
  }

  static InvocationSiteTransformationReason decode(String encodedReason) {
    int firstDelimiterPos = encodedReason.indexOf(":");
    InvocationSiteTransformationKind logReason =
        InvocationSiteTransformationKind.valueOf(encodedReason.substring(0, firstDelimiterPos));
    MethodKey methodKey = MethodKey.decode(encodedReason.substring(1 + firstDelimiterPos));
    return create(logReason, methodKey);
  }

  final String encode() {
    return String.format("%s:%s", kind(), method().encode());
  }
}
