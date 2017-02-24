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
package com.google.devtools.build.android.desugar;

import com.google.auto.value.AutoValue;
import org.objectweb.asm.Handle;

@AutoValue
abstract class LambdaInfo {
  public static LambdaInfo create(
      String desiredInternalName,
      String factoryMethodDesc,
      Handle methodReference,
      Handle bridgeMethod) {
    return new AutoValue_LambdaInfo(
        desiredInternalName, factoryMethodDesc, methodReference, bridgeMethod);
  }

  public abstract String desiredInternalName();
  public abstract String factoryMethodDesc();
  public abstract Handle methodReference();
  public abstract Handle bridgeMethod();
}
