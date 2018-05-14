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
package com.google.devtools.build.lib.packages;

import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;

/**
 * A helper class for PredicateWithMessage with default predicates.
 */
public abstract class PredicatesWithMessage implements PredicateWithMessage<Object> {

  @AutoCodec @VisibleForSerialization
  static final PredicateWithMessage<?> ALWAYS_TRUE =
      new PredicateWithMessage<Object>() {
        @Override
        public boolean apply(Object input) {
          return true;
        }

        @Override
        public String getErrorReason(Object param) {
          throw new UnsupportedOperationException();
        }
      };

  @SuppressWarnings("unchecked")
  public static <T> PredicateWithMessage<T> alwaysTrue() {
    return (PredicateWithMessage<T>) ALWAYS_TRUE;
  }
}
