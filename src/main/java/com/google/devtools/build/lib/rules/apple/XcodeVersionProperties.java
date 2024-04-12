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

package com.google.devtools.build.lib.rules.apple;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Optional;
import com.google.common.base.Strings;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.starlarkbuildapi.apple.XcodePropertiesApi;
import java.util.Objects;
import javax.annotation.Nullable;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkThread;

/** A tuple containing information about a version of Xcode and its properties. */
@Immutable
public class XcodeVersionProperties extends NativeInfo implements XcodePropertiesApi {
  /** Starlark identifier for XcodeVersionProperties provider. */
  public static final Provider PROVIDER = new Provider();

  /** Starlark name for the XcodeVersionProperties provider. */
  public static final String STARLARK_NAME = "XcodeProperties";

  @VisibleForTesting public static final String DEFAULT_IOS_SDK_VERSION = "8.4";
  @VisibleForTesting public static final String DEFAULT_VISIONOS_SDK_VERSION = "1.0";
  @VisibleForTesting public static final String DEFAULT_WATCHOS_SDK_VERSION = "2.0";
  @VisibleForTesting public static final String DEFAULT_MACOS_SDK_VERSION = "10.11";
  @VisibleForTesting public static final String DEFAULT_TVOS_SDK_VERSION = "9.0";

  private final Optional<DottedVersion> xcodeVersion;
  private final DottedVersion defaultIosSdkVersion;
  private final DottedVersion defaultVisionosSdkVersion;
  private final DottedVersion defaultWatchosSdkVersion;
  private final DottedVersion defaultTvosSdkVersion;
  private final DottedVersion defaultMacosSdkVersion;

  /**
   * Creates and returns a tuple representing no known Xcode property information (defaults are used
   * where applicable).
   */
  // TODO(bazel-team): The Xcode version should be a well-defined value, either specified by the
  // user, evaluated on the local system, or set to a sensible default.
  // Unfortunately, until the local system evaluation hook is created, this constraint would break
  // some users.
  @StarlarkMethod(
      name = "unknownXcodeVersionProperties",
      documented = false,
      useStarlarkThread = true)
  public XcodeVersionProperties unknownXcodeVersionProperties(StarlarkThread thread) {
    return new XcodeVersionProperties(null);
  }

  /**
   * Constructor for when only the Xcode version is specified, but no property information is
   * specified.
   */
  public XcodeVersionProperties(Object xcodeVersion) {
    this(xcodeVersion, null, null, null, null, null);
  }

  /**
   * General constructor. Some (nullable) properties may be left unspecified. In these cases, a
   * semi-sensible default will be assigned to the property value.
   */
  XcodeVersionProperties(
      @Nullable Object xcodeVersion,
      @Nullable String defaultIosSdkVersion,
      @Nullable String defaultVisionosSdkVersion,
      @Nullable String defaultWatchosSdkVersion,
      @Nullable String defaultTvosSdkVersion,
      @Nullable String defaultMacosSdkVersion) {
    this.xcodeVersion =
        Starlark.isNullOrNone(xcodeVersion)
            ? Optional.absent()
            : Optional.of((DottedVersion) xcodeVersion);
    this.defaultIosSdkVersion =
        Strings.isNullOrEmpty(defaultIosSdkVersion)
            ? DottedVersion.fromStringUnchecked(DEFAULT_IOS_SDK_VERSION)
            : DottedVersion.fromStringUnchecked(defaultIosSdkVersion);
    this.defaultVisionosSdkVersion =
        Strings.isNullOrEmpty(defaultVisionosSdkVersion)
            ? DottedVersion.fromStringUnchecked(DEFAULT_VISIONOS_SDK_VERSION)
            : DottedVersion.fromStringUnchecked(defaultVisionosSdkVersion);
    this.defaultWatchosSdkVersion =
        Strings.isNullOrEmpty(defaultWatchosSdkVersion)
            ? DottedVersion.fromStringUnchecked(DEFAULT_WATCHOS_SDK_VERSION)
            : DottedVersion.fromStringUnchecked(defaultWatchosSdkVersion);
    this.defaultTvosSdkVersion =
        Strings.isNullOrEmpty(defaultTvosSdkVersion)
            ? DottedVersion.fromStringUnchecked(DEFAULT_TVOS_SDK_VERSION)
            : DottedVersion.fromStringUnchecked(defaultTvosSdkVersion);
    this.defaultMacosSdkVersion =
        Strings.isNullOrEmpty(defaultMacosSdkVersion)
            ? DottedVersion.fromStringUnchecked(DEFAULT_MACOS_SDK_VERSION)
            : DottedVersion.fromStringUnchecked(defaultMacosSdkVersion);
  }

  @Override
  public Provider getProvider() {
    return PROVIDER;
  }

  /** Returns the Xcode version, or null if the Xcode version is unknown. */
  @Nullable
  @Override
  public String getXcodeVersionString() {
    if (xcodeVersion.isPresent()) {
      return xcodeVersion.get().toString();
    }
    return null;
  }

  /** Returns the default iOS SDK version to use if this Xcode version is in use. */
  @Nullable
  @Override
  public String getDefaultIosSdkVersionString() {
    return defaultIosSdkVersion != null ? defaultIosSdkVersion.toString() : null;
  }

  /** Returns the default visionOS SDK version to use if this Xcode version is in use. */
  @Nullable
  @Override
  public String getDefaultVisionosSdkVersionString() {
    return defaultVisionosSdkVersion != null ? defaultVisionosSdkVersion.toString() : null;
  }

  /** Returns the default watchOS SDK version to use if this Xcode version is in use. */
  @Nullable
  @Override
  public String getDefaultWatchosSdkVersionString() {
    return defaultWatchosSdkVersion != null ? defaultWatchosSdkVersion.toString() : null;
  }

  /** Returns the default tvOS SDK version to use if this Xcode version is in use. */
  @Nullable
  @Override
  public String getDefaultTvosSdkVersionString() {
    return defaultTvosSdkVersion != null ? defaultTvosSdkVersion.toString() : null;
  }

  /** Returns the default macOS SDK version to use if this Xcode version is in use. */
  @Nullable
  @Override
  public String getDefaultMacosSdkVersionString() {
    return defaultMacosSdkVersion != null ? defaultMacosSdkVersion.toString() : null;
  }

  /** Returns the Xcode version, or {@link Optional#absent} if the Xcode version is unknown. */
  public Optional<DottedVersion> getXcodeVersion() {
    return xcodeVersion;
  }

  @Nullable
  public DottedVersion getDefaultIosSdkVersion() {
    return defaultIosSdkVersion;
  }

  @Nullable
  public DottedVersion getDefaultVisionosSdkVersion() {
    return defaultVisionosSdkVersion;
  }

  @Nullable
  public DottedVersion getDefaultWatchosSdkVersion() {
    return defaultWatchosSdkVersion;
  }

  @Nullable
  public DottedVersion getDefaultTvosSdkVersion() {
    return defaultTvosSdkVersion;
  }

  @Nullable
  public DottedVersion getDefaultMacosSdkVersion() {
    return defaultMacosSdkVersion;
  }

  @Override
  public boolean equals(Object other) {
    if (other == null) {
      return false;
    }
    if (!(other instanceof XcodeVersionProperties)) {
      return false;
    }
    XcodeVersionProperties otherData = (XcodeVersionProperties) other;
    return xcodeVersion.equals(otherData.getXcodeVersion())
        && defaultIosSdkVersion.equals(otherData.getDefaultIosSdkVersion())
        && defaultVisionosSdkVersion.equals(otherData.getDefaultVisionosSdkVersion())
        && defaultWatchosSdkVersion.equals(otherData.getDefaultWatchosSdkVersion())
        && defaultTvosSdkVersion.equals(otherData.getDefaultTvosSdkVersion())
        && defaultMacosSdkVersion.equals(otherData.getDefaultMacosSdkVersion());
  }

  @Override
  public int hashCode() {
    return Objects.hash(
        xcodeVersion,
        defaultIosSdkVersion,
        defaultVisionosSdkVersion,
        defaultWatchosSdkVersion,
        defaultTvosSdkVersion,
        defaultMacosSdkVersion);
  }

  /** Provider class for {@link XcodeVersionProperties} objects. */
  public static class Provider extends BuiltinProvider<XcodeVersionProperties>
      implements XcodePropertiesApi.Provider {
    private Provider() {
      super(XcodePropertiesApi.NAME, XcodeVersionProperties.class);
    }

    @Override
    public XcodePropertiesApi createInfo(
        Object starlarkVersion,
        Object starlarkDefaultIosSdkVersion,
        Object starlarkDefaultVisionosSdkVersion,
        Object starlarkDefaultWatchosSdkVersion,
        Object starlarkDefaultTvosSdkVersion,
        Object starlarkDefaultMacosSdkVersion,
        StarlarkThread thread)
        throws EvalException {
      return new XcodeVersionProperties(
          Starlark.isNullOrNone(starlarkVersion)
              ? null
              : DottedVersion.fromStringUnchecked((String) starlarkVersion),
          Starlark.isNullOrNone(starlarkDefaultIosSdkVersion)
              ? null
              : (String) starlarkDefaultIosSdkVersion,
          Starlark.isNullOrNone(starlarkDefaultVisionosSdkVersion)
              ? null
              : (String) starlarkDefaultVisionosSdkVersion,
          Starlark.isNullOrNone(starlarkDefaultWatchosSdkVersion)
              ? null
              : (String) starlarkDefaultWatchosSdkVersion,
          Starlark.isNullOrNone(starlarkDefaultTvosSdkVersion)
              ? null
              : (String) starlarkDefaultTvosSdkVersion,
          Starlark.isNullOrNone(starlarkDefaultMacosSdkVersion)
              ? null
              : (String) starlarkDefaultMacosSdkVersion);
    }
  }
}
