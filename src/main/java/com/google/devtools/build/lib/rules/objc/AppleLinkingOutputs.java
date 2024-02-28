// Copyright 2021 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.objc;

import com.google.auto.value.AutoValue;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.StarlarkInfo;
import com.google.devtools.build.lib.packages.StructImpl;
import java.util.HashMap;
import net.starlark.java.syntax.Location;

/**
 * The providers and artifact outputs returned by the {@code apple_common.link_multi_arch_binary}
 * API.
 */
public class AppleLinkingOutputs {

  /** Represents an Apple target triplet (arch, platform, env) for a multi-arch target. */
  @AutoValue
  public abstract static class TargetTriplet {
    static TargetTriplet create(String architecture, String platform, String environment) {
      return new AutoValue_AppleLinkingOutputs_TargetTriplet(architecture, platform, environment);
    }

    abstract String architecture();

    abstract String platform();

    abstract String environment();

    /** Returns a Starlark Dict representation of a {@link TargetTriplet} */
    public final StructImpl toStarlarkStruct() {
      Provider constructor = new BuiltinProvider<StructImpl>("target_triplet", StructImpl.class) {};
      HashMap<String, Object> fields = new HashMap<>();
      fields.put("architecture", architecture());
      fields.put("environment", environment());
      fields.put("platform", platform());
      return StarlarkInfo.create(constructor, fields, Location.BUILTIN);
    }
  }
}
