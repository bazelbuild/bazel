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
package com.google.devtools.build.android.desugar.runtime;

import android.text.Spannable;
import android.view.textclassifier.TextLinks;
import java.util.function.Function;

/**
 * Conversion from desugared to built-in {@link Function} for calling built-in Android APIs (see
 * b/79121791).
 */
// TODO(b/74087778): Unnecessary when j$.u.f.Function becomes subtype of built-in j.u.f.Function
@SuppressWarnings("AndroidJdkLibsChecker")
public final class FunctionWrapper<T, R> implements Function<T, R> {

  private final j$.util.function.Function<T, R> wrapped;

  private FunctionWrapper(j$.util.function.Function<T, R> wrapped) {
    this.wrapped = wrapped;
  }

  @Override
  public R apply(T arg) {
    return wrapped.apply(arg);
  }

  public static int apply(
      TextLinks receiver,
      Spannable text,
      int applyStrategy,
      j$.util.function.Function<TextLinks.TextLink, TextLinks.TextLinkSpan> spanFactory) {
    return receiver.apply(
        text,
        applyStrategy,
        spanFactory != null
            ? new FunctionWrapper<TextLinks.TextLink, TextLinks.TextLinkSpan>(spanFactory)
            : null);
  }
}
