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

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.rules.cpp.CcInfo;
import com.google.devtools.build.lib.rules.cpp.CcNativeLibraryInfo;
import com.google.devtools.build.lib.rules.cpp.LibraryToLink;
import com.google.devtools.build.lib.rules.java.JavaInfo.JavaInfoInternalProvider;
import java.util.Collection;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;

/** Provides information about C++ libraries to be linked into Java targets. */
@Immutable
public final class JavaCcInfoProvider implements JavaInfoInternalProvider {

  // TODO(b/183579145): Replace CcInfo with only linking information.
  private final CcInfo ccInfo;

  public CcInfo getCcInfo() {
    return ccInfo;
  }

  public JavaCcInfoProvider(CcInfo ccInfo) {
    this.ccInfo =
        CcInfo.builder()
            .setCcLinkingContext(ccInfo.getCcLinkingContext())
            .setCcNativeLibraryInfo(ccInfo.getCcNativeLibraryInfo())
            .build();
  }

  /** Merges several JavaCcInfoProvider providers into one. */
  public static JavaCcInfoProvider merge(Collection<JavaCcInfoProvider> providers) {
    ImmutableList<CcInfo> ccInfos =
        providers.stream().map(JavaCcInfoProvider::getCcInfo).collect(toImmutableList());
    return new JavaCcInfoProvider(CcInfo.merge(ccInfos));
  }

  @Nullable
  static JavaCcInfoProvider fromStarlarkJavaInfo(StructImpl javaInfo) throws EvalException {
    CcInfo ccInfo = javaInfo.getValue("cc_link_params_info", CcInfo.class);
    if (ccInfo != null) {
      return new JavaCcInfoProvider(ccInfo);
    } else {
      NestedSet<LibraryToLink> transitiveNativeLibraries =
          Depset.cast(
              javaInfo.getValue("transitive_native_libraries"),
              LibraryToLink.class,
              "transitive_native_libraries");
      if (transitiveNativeLibraries.isEmpty()) {
        return null;
      } else {
        return new JavaCcInfoProvider(
            CcInfo.builder()
                .setCcNativeLibraryInfo(new CcNativeLibraryInfo(transitiveNativeLibraries))
                .build());
      }
    }
  }
}
