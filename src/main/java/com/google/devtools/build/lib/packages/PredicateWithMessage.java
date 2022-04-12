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

import com.google.common.base.Predicate;

/**
 * A predicate which supports error messages.
 * @param <T> - the predicate is applied on T objects
 */
public interface PredicateWithMessage<T> extends Predicate<T> {

  /**
   * The error message to display when predicate checks param. Only makes sense to call this method
   * if apply(param) returns false.
   */
  String getErrorReason(T param);
}
