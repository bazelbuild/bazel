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

package com.google.devtools.build.lib.rules.java;

import static java.util.Objects.requireNonNull;

import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.StarlarkInfo;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.rules.cpp.CcInfo;
import com.google.devtools.build.lib.rules.cpp.CcNativeLibraryInfo;
import com.google.devtools.build.lib.rules.java.JavaInfo.JavaInfoInternalProvider;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;

/** Provides information about C++ libraries to be linked into Java targets. */
@Immutable
@AutoCodec
public record JavaCcInfoProvider(CcInfo ccInfo) implements JavaInfoInternalProvider {
  public JavaCcInfoProvider {
    requireNonNull(ccInfo, "ccInfo");
  }

  // TODO(b/183579145): Replace CcInfo with only linking information.

  public static JavaCcInfoProvider create(CcInfo ccInfo) {
    return new JavaCcInfoProvider(
        CcInfo.builder()
            .setCcLinkingContext(ccInfo.getCcLinkingContextStruct())
            .setCcNativeLibraryInfo(ccInfo.getCcNativeLibraryInfo())
            .setCcDebugInfoContext(ccInfo.getCcDebugInfoContext())
            .build());
  }

  @Nullable
  static JavaCcInfoProvider fromStarlarkJavaInfo(StructImpl javaInfo) throws EvalException {
    CcInfo ccInfo = javaInfo.getValue("cc_link_params_info", CcInfo.class);
    if (ccInfo == null) {
      NestedSet<StarlarkInfo> transitiveCcNativeLibraries =
          Depset.cast(
              javaInfo.getValue("transitive_native_libraries"),
              StarlarkInfo.class,
              "transitive_native_libraries");
      if (transitiveCcNativeLibraries.isEmpty()) {
        return null;
      }
      ccInfo =
          CcInfo.builder()
              .setCcNativeLibraryInfo(CcNativeLibraryInfo.of(transitiveCcNativeLibraries))
              .build();
    }
    return create(ccInfo);
  }
}
