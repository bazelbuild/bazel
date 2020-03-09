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
package com.google.devtools.build.android.desugar.scan;

import static com.google.common.base.Preconditions.checkArgument;

import com.google.auto.value.AutoValue;
import com.google.errorprone.annotations.Immutable;

@AutoValue
@Immutable
abstract class KeepReference {
  public static KeepReference classReference(String internalName) {
    checkArgument(!internalName.isEmpty());
    return new AutoValue_KeepReference(internalName, "", "");
  }

  public static KeepReference memberReference(String internalName, String name, String desc) {
    checkArgument(!internalName.isEmpty());
    checkArgument(!name.isEmpty());
    checkArgument(!desc.isEmpty());
    return new AutoValue_KeepReference(internalName, name, desc);
  }

  public final boolean isMemberReference() {
    return !name().isEmpty();
  }

  public final boolean isMethodReference() {
    return desc().startsWith("(");
  }

  public final boolean isFieldReference() {
    return isMemberReference() && !isMethodReference();
  }

  public abstract String internalName();

  public abstract String name();

  public abstract String desc();
}
