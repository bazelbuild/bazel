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
package com.google.devtools.build.lib.rules.apple;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkThread;

/** The available xcode versions computed from the {@code available_xcodes} rule. */
@Immutable
public class AvailableXcodesInfo extends NativeInfo {
  /** Starlark name for this provider. */
  public static final String STARLARK_NAME = "AvailableXcodesInfo";

  /** Provider identifier for {@link AvailableXcodesInfo}. */
  public static final BuiltinProvider<AvailableXcodesInfo> PROVIDER = new Provider();

  private final Iterable<XcodeVersionRuleData> availableXcodes;
  private final XcodeVersionRuleData defaultVersion;

  public AvailableXcodesInfo(
      Iterable<XcodeVersionRuleData> availableXcodes, XcodeVersionRuleData defaultVersion) {
    this.availableXcodes = availableXcodes;
    this.defaultVersion = defaultVersion;
  }

  @Override
  public BuiltinProvider<AvailableXcodesInfo> getProvider() {
    return PROVIDER;
  }

  /** Returns the available xcode versions from {@code available_xcodes}. */
  public Iterable<XcodeVersionRuleData> getAvailableVersions() {
    return availableXcodes;
  }

  /** Returns the default xcode version from {@code available_xcodes}. */
  public XcodeVersionRuleData getDefaultVersion() {
    return defaultVersion;
  }

  /** Provider class for {@link XcodeVersionRuleData} objects. */
  public static class Provider extends BuiltinProvider<AvailableXcodesInfo>
      implements AvailableXcodesApi<Artifact> {
    private Provider() {
      super(AvailableXcodesInfo.STARLARK_NAME, AvailableXcodesInfo.class);
    }

    @Override
    @SuppressWarnings("unchecked")
    public AvailableXcodesInfo createInfo(
        Object availableXcodes, Object defaultVersion, StarlarkThread thread) throws EvalException {
      Sequence<XcodeVersionRuleData> availableXcodesSequence =
          nullIfNone(availableXcodes, Sequence.class);
      return new AvailableXcodesInfo(
          Sequence.cast(availableXcodesSequence, XcodeVersionRuleData.class, "availableXcodes"),
          nullIfNone(defaultVersion, XcodeVersionRuleData.class));
    }

    @Nullable
    private static <T> T nullIfNone(Object object, Class<T> type) {
      return object != Starlark.NONE ? type.cast(object) : null;
    }
  }
}
