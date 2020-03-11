// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.desugar.io;

import com.google.auto.value.AutoValue;

/** A value class to store the fields information. */
@AutoValue
public abstract class FieldInfo {

  public static FieldInfo create(String owner, String name, String desc) {
    return new AutoValue_FieldInfo(owner, name, desc);
  }

  public abstract String owner();

  public abstract String name();

  public abstract String desc();
}
