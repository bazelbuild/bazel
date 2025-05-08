// Copyright 2024 The Bazel Authors. All rights reserved.
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

import com.google.auto.value.AutoValue;
import javax.annotation.Nullable;

/**
 * Object representing an option that was added to a Blaze invocation by the invocation policy. The
 * value will be in its raw (unparsed) form.
 */
@AutoValue
public abstract class OptionAndRawValue {
  public static OptionAndRawValue create(String optionName, @Nullable String rawValue) {
    return new AutoValue_OptionAndRawValue(optionName, rawValue);
  }

  public abstract String getOptionName();

  /** {@code rawValue} is nullable for Void options. */
  @Nullable
  public abstract String getRawValue();
}
