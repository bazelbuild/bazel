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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.Constants;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.DefaultLabelConverter;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.common.options.Converters.CommaSeparatedOptionListConverter;
import com.google.devtools.common.options.EnumConverter;
import com.google.devtools.common.options.Option;

import java.util.List;

/**
 * Command-line options for building for Apple platforms.
 */
public class AppleCommandLineOptions extends FragmentOptions {

  @Option(
    name = "xcode_version",
    defaultValue = "null",
    category = "build",
    converter = DottedVersionConverter.class,
    help =
        "If specified, uses xcode of the given version for relevant build actions. "
            + "If unspecified, uses the executor default version of xcode."
  )
  // TODO(bazel-team): This should be of String type, to allow referencing an alias based
  // on an xcode_config target.
  public DottedVersion xcodeVersion;

  @Option(
    name = "ios_sdk_version",
    // TODO(bazel-team): Make this flag optional, and infer SDKROOT based on executor default.
    defaultValue = DEFAULT_IOS_SDK_VERSION,
    converter = DottedVersionConverter.class,
    category = "build",
    help = "Specifies the version of the iOS SDK to use to build iOS applications."
  )
  public DottedVersion iosSdkVersion;

  @Option(
    name = "watchos_sdk_version",
    // TODO(bazel-team): Make this flag optional, and infer SDKROOT based on executor default.
    defaultValue = DEFAULT_WATCHOS_SDK_VERSION,
    converter = DottedVersionConverter.class,
    category = "build",
    help = "Specifies the version of the WatchOS SDK to use to build WatchOS applications."
  )
  public DottedVersion watchOsSdkVersion;

  @Option(
    name = "tvos_sdk_version",
    // TODO(bazel-team): Make this flag optional, and infer SDKROOT based on executor default.
    defaultValue = DEFAULT_APPLETVOS_SDK_VERSION,
    converter = DottedVersionConverter.class,
    category = "build",
    help = "Specifies the version of the AppleTVOS SDK to use to build AppleTVOS applications."
  )
  public DottedVersion tvOsSdkVersion;

  @Option(
    name = "macosx_sdk_version",
    // TODO(bazel-team): Make this flag optional, and infer SDKROOT based on executor default.
    defaultValue = DEFAULT_MACOSX_SDK_VERSION,
    converter = DottedVersionConverter.class,
    category = "build",
    help = "Specifies the version of the Mac OS X SDK to use to build Mac OS X applications."
  )
  public DottedVersion macOsXSdkVersion;

  @VisibleForTesting public static final String DEFAULT_IOS_SDK_VERSION = "8.4";
  @VisibleForTesting public static final String DEFAULT_WATCHOS_SDK_VERSION = "2.0";
  @VisibleForTesting public static final String DEFAULT_MACOSX_SDK_VERSION = "10.10";
  @VisibleForTesting public static final String DEFAULT_APPLETVOS_SDK_VERSION = "1.0";

  @Option(name = "ios_cpu",
      defaultValue = DEFAULT_IOS_CPU,
      category = "build",
      help = "Specifies to target CPU of iOS compilation.")
  public String iosCpu;

  @Option(name = "ios_multi_cpus",
      converter = CommaSeparatedOptionListConverter.class,
      defaultValue = "",
      category = "flags",
      help = "Comma-separated list of architectures to build an ios_application with. The result "
          + "is a universal binary containing all specified architectures.")
  public List<String> iosMultiCpus;
  
  @VisibleForTesting static final String DEFAULT_IOS_CPU = "x86_64";
  
  @Option(name = "default_ios_provisioning_profile",
      defaultValue = "",
      category = "undocumented",
      converter = DefaultProvisioningProfileConverter.class)
  public Label defaultProvisioningProfile;
  
  @Option(name = "xcode_version_config",
      defaultValue = "",
      category = "undocumented",
      converter = XcodeVersionConfigConverter.class,
      help = "The label of the xcode_config rule to be used for selecting the xcode version "
          + "in the build configuration")
  public Label xcodeVersionConfig;

  /**
   * The default label of the build-wide {@code xcode_config} configuration rule. This can be
   * changed from the default using the {@code xcode_version_config} build flag.
   */
  static final String DEFAULT_XCODE_VERSION_CONFIG_LABEL =
      Constants.TOOLS_REPOSITORY + "//tools/objc:host_xcodes";

  /** Converter for --default_ios_provisioning_profile. */
  public static class DefaultProvisioningProfileConverter extends DefaultLabelConverter {
    public DefaultProvisioningProfileConverter() {
      super("//tools/objc:default_provisioning_profile");
    }
  }

  @Option(name = "apple_bitcode",
      converter = AppleBitcodeMode.Converter.class,
      // TODO(blaze-team): Default to embedded_markers when fully implemented.
      defaultValue = "none",
      category = "flags",
      help = "Specify the Apple bitcode mode for compile steps. "
             + "Values: 'none', 'embedded_markers', 'embedded'.")
  public AppleBitcodeMode appleBitcodeMode;

  /** Converter for {@code --xcode_version_config}. */
  public static class XcodeVersionConfigConverter extends DefaultLabelConverter {
    public XcodeVersionConfigConverter() {
      super(DEFAULT_XCODE_VERSION_CONFIG_LABEL);
    }
  }
  
  private Platform getPlatform() {
    for (String architecture : iosMultiCpus) {
      if (Platform.forIosArch(architecture) == Platform.IOS_DEVICE) {
        return Platform.IOS_DEVICE;
      }
    }
    return Platform.forIosArch(iosCpu);
  }
  
  @Override
  public void addAllLabels(Multimap<String, Label> labelMap) {
    if (getPlatform() == Platform.IOS_DEVICE) {
      labelMap.put("default_provisioning_profile", defaultProvisioningProfile);
    }
    labelMap.put("xcode_version_config", xcodeVersionConfig);
  }
  
  /**
   * Represents the Apple Bitcode mode for compilation steps.
   * 
   * <p>Bitcode is an intermediate representation of a compiled program. For many platforms,
   * Apple requires app submissions to contain bitcode in order to be uploaded to the app store.
   * 
   * <p>This is a build-wide value, as bitcode mode needs to be consistent among a target and
   * its compiled dependencies.
   */
  public enum AppleBitcodeMode {

    /** 
     * Do not compile bitcode.
     */
    NONE("none"),
    /**
     * Compile the minimal set of bitcode markers. This is often the best option for
     * developer/debug builds.
     */
    EMBEDDED_MARKERS("embedded_markers", "-fembed-bitcode-marker"),
    /**
     * Fully embed bitcode in compiled files. This is often the best option for release builds.
     */
    EMBEDDED("embedded", "-fembed-bitcode");

    private final String mode;
    private final ImmutableList<String> compilerFlags;

    private AppleBitcodeMode(String mode, String... compilerFlags) {
      this.mode = mode;
      this.compilerFlags = ImmutableList.copyOf(compilerFlags);
    }

    @Override
    public String toString() {
      return mode;
    }

    /**
     * Returns the flags that should be added to compile actions to use this bitcode setting.
     */
    public ImmutableList<String> getCompilerFlags() {
      return compilerFlags;
    }

    /**
     * Converts to {@link AppleBitcodeMode}.
     */
    public static class Converter extends EnumConverter<AppleBitcodeMode> {
      public Converter() {
        super(AppleBitcodeMode.class, "apple bitcode mode");
      }
    }
  }

  @Override
  public FragmentOptions getHost(boolean fallback) {
    AppleCommandLineOptions host = (AppleCommandLineOptions) super.getHost(fallback);

    // Set options needed in the host configuration.
    host.xcodeVersionConfig = xcodeVersionConfig;
    host.xcodeVersion = xcodeVersion;
    host.appleBitcodeMode = appleBitcodeMode;

    return host;
  }
}
