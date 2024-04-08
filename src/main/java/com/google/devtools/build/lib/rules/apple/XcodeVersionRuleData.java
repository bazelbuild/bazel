// Copyright 2015 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import java.util.List;
import java.util.Objects;
import javax.annotation.Nullable;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkThread;

/**
 * A tuple containing the information in a single target of the {@code xcode_version} rule. A single
 * target of this rule contains an official version label decided by Apple, a number of supported
 * aliases one might use to reference this version, and various properties of the Xcode version
 * (such as default SDK versions).
 *
 * <p>For example, one may want to reference official Xcode version 7.0.1 using the "7" or "7.0"
 * aliases. This official version of Xcode may have a default supported iOS SDK of 9.0.
 */
@Immutable
public class XcodeVersionRuleData extends NativeInfo {
  private static final String NAME = "XcodeVersionRuleData";

  private final Label label;
  private final XcodeVersionProperties xcodeVersionProperties;
  private final ImmutableList<String> aliases;
  public static final Provider PROVIDER = new Provider();

  XcodeVersionRuleData(
      Label label, XcodeVersionProperties xcodeVersionProperties, List<String> aliases) {
    this.label = label;
    this.xcodeVersionProperties = xcodeVersionProperties;
    this.aliases = ImmutableList.copyOf(aliases);
  }

  @Override
  public Provider getProvider() {
    return PROVIDER;
  }

  /** Returns the label of the owning target of this provider. */
  @StarlarkMethod(name = "label", structField = true, documented = false)
  public Label getLabel() {
    return label;
  }

  /** Returns the official Xcode version the owning {@code xcode_version} target is referencing. */
  @StarlarkMethod(name = "version", structField = true, documented = false)
  public DottedVersion getVersion() {
    return xcodeVersionProperties.getXcodeVersion().get();
  }

  /** Returns the properties of the {@code xcode_version} target's referenced Xcode version. */
  @StarlarkMethod(name = "xcode_version_properties", structField = true, documented = false)
  public XcodeVersionProperties getXcodeVersionProperties() {
    return xcodeVersionProperties;
  }

  /** Returns the accepted string aliases for this Xcode version. */
  @StarlarkMethod(name = "aliases", structField = true, documented = false)
  public List<String> getAliases() {
    return aliases;
  }

  @Override
  public boolean equals(Object other) {
    if (other == null) {
      return false;
    }
    if (!(other instanceof XcodeVersionRuleData)) {
      return false;
    }
    XcodeVersionRuleData otherData = (XcodeVersionRuleData) other;
    return (getVersion().equals(otherData.getVersion())
        && xcodeVersionProperties
            .getXcodeVersion()
            .equals(otherData.getXcodeVersionProperties().getXcodeVersion())
        && xcodeVersionProperties
            .getDefaultIosSdkVersion()
            .equals(otherData.getXcodeVersionProperties().getDefaultIosSdkVersion())
        && xcodeVersionProperties
            .getDefaultVisionosSdkVersion()
            .equals(otherData.getXcodeVersionProperties().getDefaultVisionosSdkVersion())
        && xcodeVersionProperties
            .getDefaultWatchosSdkVersion()
            .equals(otherData.getXcodeVersionProperties().getDefaultWatchosSdkVersion())
        && xcodeVersionProperties
            .getDefaultTvosSdkVersion()
            .equals(otherData.getXcodeVersionProperties().getDefaultTvosSdkVersion())
        && xcodeVersionProperties
            .getDefaultMacosSdkVersion()
            .equals(otherData.getXcodeVersionProperties().getDefaultMacosSdkVersion()));
  }

  @Override
  public int hashCode() {
    return Objects.hash(getVersion(), xcodeVersionProperties);
  }

  /** Provider class for {@link XcodeVersionRuleData} objects. */
  public static class Provider extends BuiltinProvider<XcodeVersionRuleData>
      implements XcodeVersionProviderApi<Artifact> {
    private Provider() {
      super(XcodeVersionRuleData.NAME, XcodeVersionRuleData.class);
    }

    @Override
    @SuppressWarnings("unchecked")
    public XcodeVersionRuleData createInfo(
        Object starlarkLabel,
        Object starlarkXcodeProperties,
        Object starlarkAliases,
        StarlarkThread thread)
        throws EvalException {
      Sequence<String> aliases = nullIfNone(starlarkAliases, Sequence.class);
      return new XcodeVersionRuleData(
          nullIfNone(starlarkLabel, Label.class),
          nullIfNone(starlarkXcodeProperties, XcodeVersionProperties.class),
          Sequence.cast(aliases, String.class, "aliases"));
    }

    @Nullable
    private static <T> T nullIfNone(Object object, Class<T> type) {
      return object != Starlark.NONE ? type.cast(object) : null;
    }
  }
}
