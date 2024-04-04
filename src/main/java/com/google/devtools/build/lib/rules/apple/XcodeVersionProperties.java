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
  private final DottedVersion iosSdkMinimumOs;
  private final DottedVersion defaultVisionosSdkVersion;
  private final DottedVersion visionosSdkMinimumOs;
  private final DottedVersion defaultWatchosSdkVersion;
  private final DottedVersion watchosSdkMinimumOs;
  private final DottedVersion defaultTvosSdkVersion;
  private final DottedVersion tvosSdkMinimumOs;
  private final DottedVersion defaultMacosSdkVersion;
  private final DottedVersion macosSdkMinimumOs;

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
    this(xcodeVersion, null, null, null, null, null, null, null, null, null, null);
  }

  /**
   * General constructor. Some (nullable) properties may be left unspecified. In these cases, a
   * semi-sensible default will be assigned to the property value.
   */
  XcodeVersionProperties(
      @Nullable Object xcodeVersion,
      @Nullable String defaultIosSdkVersion,
      @Nullable String iosSdkMinimumOs,
      @Nullable String defaultVisionosSdkVersion,
      @Nullable String visionosSdkMinimumOs,
      @Nullable String defaultWatchosSdkVersion,
      @Nullable String watchosSdkMinimumOs,
      @Nullable String defaultTvosSdkVersion,
      @Nullable String tvosSdkMinimumOs,
      @Nullable String defaultMacosSdkVersion,
      @Nullable String macosSdkMinimumOs) {
    this.xcodeVersion =
        Starlark.isNullOrNone(xcodeVersion)
            ? Optional.absent()
            : Optional.of((DottedVersion) xcodeVersion);
    this.defaultIosSdkVersion =
        Strings.isNullOrEmpty(defaultIosSdkVersion)
            ? DottedVersion.fromStringUnchecked(DEFAULT_IOS_SDK_VERSION)
            : DottedVersion.fromStringUnchecked(defaultIosSdkVersion);
    this.iosSdkMinimumOs =
        Strings.isNullOrEmpty(iosSdkMinimumOs)
            ? this.defaultIosSdkVersion
            : DottedVersion.fromStringUnchecked(iosSdkMinimumOs);
    this.defaultVisionosSdkVersion =
        Strings.isNullOrEmpty(defaultVisionosSdkVersion)
            ? DottedVersion.fromStringUnchecked(DEFAULT_VISIONOS_SDK_VERSION)
            : DottedVersion.fromStringUnchecked(defaultVisionosSdkVersion);
    this.visionosSdkMinimumOs =
        Strings.isNullOrEmpty(visionosSdkMinimumOs)
            ? this.defaultVisionosSdkVersion
            : DottedVersion.fromStringUnchecked(visionosSdkMinimumOs);
    this.defaultWatchosSdkVersion =
        Strings.isNullOrEmpty(defaultWatchosSdkVersion)
            ? DottedVersion.fromStringUnchecked(DEFAULT_WATCHOS_SDK_VERSION)
            : DottedVersion.fromStringUnchecked(defaultWatchosSdkVersion);
    this.watchosSdkMinimumOs =
        Strings.isNullOrEmpty(watchosSdkMinimumOs)
            ? this.defaultWatchosSdkVersion
            : DottedVersion.fromStringUnchecked(watchosSdkMinimumOs);
    this.defaultTvosSdkVersion =
        Strings.isNullOrEmpty(defaultTvosSdkVersion)
            ? DottedVersion.fromStringUnchecked(DEFAULT_TVOS_SDK_VERSION)
            : DottedVersion.fromStringUnchecked(defaultTvosSdkVersion);
    this.tvosSdkMinimumOs =
        Strings.isNullOrEmpty(tvosSdkMinimumOs)
            ? this.defaultTvosSdkVersion
            : DottedVersion.fromStringUnchecked(tvosSdkMinimumOs);
    this.defaultMacosSdkVersion =
        Strings.isNullOrEmpty(defaultMacosSdkVersion)
            ? DottedVersion.fromStringUnchecked(DEFAULT_MACOS_SDK_VERSION)
            : DottedVersion.fromStringUnchecked(defaultMacosSdkVersion);
    this.macosSdkMinimumOs =
        Strings.isNullOrEmpty(macosSdkMinimumOs)
            ? this.defaultMacosSdkVersion
            : DottedVersion.fromStringUnchecked(macosSdkMinimumOs);
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

  /** Returns the minimum OS version supported by the iOS SDK for this version of Xcode. */
  @Override
  public String getIosSdkMinimumOsString() {
    return iosSdkMinimumOs.toString();
  }

  /** Returns the default visionOS SDK version to use if this Xcode version is in use. */
  @Nullable
  @Override
  public String getDefaultVisionosSdkVersionString() {
    return defaultVisionosSdkVersion != null ? defaultVisionosSdkVersion.toString() : null;
  }

  /** Returns the minimum OS version supported by the visionOS SDK for this version of Xcode. */
  @Override
  public String getVisionosSdkMinimumOsString() {
    return visionosSdkMinimumOs.toString();
  }

  /** Returns the default watchOS SDK version to use if this Xcode version is in use. */
  @Nullable
  @Override
  public String getDefaultWatchosSdkVersionString() {
    return defaultWatchosSdkVersion != null ? defaultWatchosSdkVersion.toString() : null;
  }

  /** Returns the minimum OS version supported by the watchOS SDK for this version of Xcode. */
  @Override
  public String getWatchosSdkMinimumOsString() {
    return watchosSdkMinimumOs.toString();
  }

  /** Returns the default tvOS SDK version to use if this Xcode version is in use. */
  @Nullable
  @Override
  public String getDefaultTvosSdkVersionString() {
    return defaultTvosSdkVersion != null ? defaultTvosSdkVersion.toString() : null;
  }

  /** Returns the minimum OS version supported by the tvOS SDK for this version of Xcode. */
  @Override
  public String getTvosSdkMinimumOsString() {
    return tvosSdkMinimumOs.toString();
  }

  /** Returns the default macOS SDK version to use if this Xcode version is in use. */
  @Nullable
  @Override
  public String getDefaultMacosSdkVersionString() {
    return defaultMacosSdkVersion != null ? defaultMacosSdkVersion.toString() : null;
  }

  /** Returns the minimum OS version supported by the macOS SDK for this version of Xcode. */
  @Override
  public String getMacosSdkMinimumOsString() {
    return macosSdkMinimumOs.toString();
  }

  /** Returns the Xcode version, or {@link Optional#absent} if the Xcode version is unknown. */
  public Optional<DottedVersion> getXcodeVersion() {
    return xcodeVersion;
  }

  @Nullable
  public DottedVersion getDefaultIosSdkVersion() {
    return defaultIosSdkVersion;
  }

  public DottedVersion getIosSdkMinimumOs() {
    return iosSdkMinimumOs;
  }

  @Nullable
  public DottedVersion getDefaultVisionosSdkVersion() {
    return defaultVisionosSdkVersion;
  }

  public DottedVersion getVisionosSdkMinimumOs() {
    return visionosSdkMinimumOs;
  }

  @Nullable
  public DottedVersion getDefaultWatchosSdkVersion() {
    return defaultWatchosSdkVersion;
  }

  public DottedVersion getWatchosSdkMinimumOs() {
    return watchosSdkMinimumOs;
  }

  @Nullable
  public DottedVersion getDefaultTvosSdkVersion() {
    return defaultTvosSdkVersion;
  }

  public DottedVersion getTvosSdkMinimumOs() {
    return tvosSdkMinimumOs;
  }

  @Nullable
  public DottedVersion getDefaultMacosSdkVersion() {
    return defaultMacosSdkVersion;
  }

  public DottedVersion getMacosSdkMinimumOs() {
    return macosSdkMinimumOs;
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
        && iosSdkMinimumOs.equals(otherData.getIosSdkMinimumOs())
        && defaultVisionosSdkVersion.equals(otherData.getDefaultVisionosSdkVersion())
        && visionosSdkMinimumOs.equals(otherData.getVisionosSdkMinimumOs())
        && defaultWatchosSdkVersion.equals(otherData.getDefaultWatchosSdkVersion())
        && watchosSdkMinimumOs.equals(otherData.getWatchosSdkMinimumOs())
        && defaultTvosSdkVersion.equals(otherData.getDefaultTvosSdkVersion())
        && tvosSdkMinimumOs.equals(otherData.getTvosSdkMinimumOs())
        && defaultMacosSdkVersion.equals(otherData.getDefaultMacosSdkVersion())
        && macosSdkMinimumOs.equals(otherData.getMacosSdkMinimumOs());
  }

  @Override
  public int hashCode() {
    return Objects.hash(
        xcodeVersion,
        defaultIosSdkVersion,
        iosSdkMinimumOs,
        defaultVisionosSdkVersion,
        visionosSdkMinimumOs,
        defaultWatchosSdkVersion,
        watchosSdkMinimumOs,
        defaultTvosSdkVersion,
        tvosSdkMinimumOs,
        defaultMacosSdkVersion,
        macosSdkMinimumOs);
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
        Object starlarkIosSdkMinimumOs,
        Object starlarkDefaultVisionosSdkVersion,
        Object starlarkVisionosSdkMinimumOs,
        Object starlarkDefaultWatchosSdkVersion,
        Object starlarkWatchosSdkMinimumOs,
        Object starlarkDefaultTvosSdkVersion,
        Object starlarkTvosSdkMinimumOs,
        Object starlarkDefaultMacosSdkVersion,
        Object starlarkMacosSdkMinimumOs,
        StarlarkThread thread)
        throws EvalException {
      return new XcodeVersionProperties(
          Starlark.isNullOrNone(starlarkVersion)
              ? null
              : DottedVersion.fromStringUnchecked((String) starlarkVersion),
          Starlark.isNullOrNone(starlarkDefaultIosSdkVersion)
              ? null
              : (String) starlarkDefaultIosSdkVersion,
          Starlark.isNullOrNone(starlarkIosSdkMinimumOs)
              ? null
              : (String) starlarkIosSdkMinimumOs,
          Starlark.isNullOrNone(starlarkDefaultVisionosSdkVersion)
              ? null
              : (String) starlarkDefaultVisionosSdkVersion,
          Starlark.isNullOrNone(starlarkVisionosSdkMinimumOs)
              ? null
              : (String) starlarkVisionosSdkMinimumOs,
          Starlark.isNullOrNone(starlarkDefaultWatchosSdkVersion)
              ? null
              : (String) starlarkDefaultWatchosSdkVersion,
          Starlark.isNullOrNone(starlarkWatchosSdkMinimumOs)
              ? null
              : (String) starlarkWatchosSdkMinimumOs,
          Starlark.isNullOrNone(starlarkDefaultTvosSdkVersion)
              ? null
              : (String) starlarkDefaultTvosSdkVersion,
          Starlark.isNullOrNone(starlarkTvosSdkMinimumOs)
              ? null
              : (String) starlarkTvosSdkMinimumOs,
          Starlark.isNullOrNone(starlarkDefaultMacosSdkVersion)
              ? null
              : (String) starlarkDefaultMacosSdkVersion,
          Starlark.isNullOrNone(starlarkMacosSdkMinimumOs)
              ? null
              : (String) starlarkMacosSdkMinimumOs);
    }
  }
}
