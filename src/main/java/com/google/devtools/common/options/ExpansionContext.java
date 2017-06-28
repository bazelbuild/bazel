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

package com.google.devtools.common.options;

import java.lang.reflect.Field;
import javax.annotation.Nullable;
import javax.annotation.concurrent.ThreadSafe;

/**
 * Encapsulates the data given to {@link ExpansionFunction} objects. This lets {@link
 * ExpansionFunction} objects change how it expands flags based on the arguments given to the {@link
 * OptionsParser}.
 */
@ThreadSafe
public final class ExpansionContext {
  private final IsolatedOptionsData optionsData;
  private final Field field;
  @Nullable private final String unparsedValue;

  public ExpansionContext(
      IsolatedOptionsData optionsData, Field field, @Nullable String unparsedValue) {
    this.optionsData = optionsData;
    this.field = field;
    this.unparsedValue = unparsedValue;
  }

  /** Metadata for the option that is being expanded. */
  public IsolatedOptionsData getOptionsData() {
    return optionsData;
  }

  /** {@link Field} object for option that is being expanded. */
  public Field getField() {
    return field;
  }

  /** Argument given to this flag during options parsing. Will be null if no argument was given. */
  @Nullable
  public String getUnparsedValue() {
    return unparsedValue;
  }
}
