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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.serialization.AbstractObjectCodecTest;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for serialization of {@link AppleConfiguration}. */
@RunWith(JUnit4.class)
public class AppleConfigurationSerializationTest
    extends AbstractObjectCodecTest<AppleConfiguration> {
  public AppleConfigurationSerializationTest() {
    super(
        new ObjectCodec<AppleConfiguration>() {
          @Override
          public void serialize(AppleConfiguration obj, CodedOutputStream codedOut)
              throws SerializationException, IOException {
            obj.serialize(codedOut);
          }

          @Override
          public AppleConfiguration deserialize(CodedInputStream codedIn)
              throws SerializationException, IOException {
            return AppleConfiguration.deserialize(codedIn);
          }

          @Override
          public Class<AppleConfiguration> getEncodedClass() {
            return AppleConfiguration.class;
          }
        },
        subject());
  }

  private static AppleConfiguration subject() {
    AppleCommandLineOptions options = new AppleCommandLineOptions();
    options.mandatoryMinimumVersion = false;
    options.xcodeVersion = DottedVersion.fromString("1.0");
    options.iosSdkVersion = DottedVersion.fromString("2.0");
    options.watchOsSdkVersion = DottedVersion.fromString("3.0");
    options.tvOsSdkVersion = DottedVersion.fromString("4.0");
    options.macOsSdkVersion = DottedVersion.fromString("5.0");
    options.iosMinimumOs = DottedVersion.fromString("6.1beta3.7");
    options.watchosMinimumOs = DottedVersion.fromString("7.0");
    options.tvosMinimumOs = DottedVersion.fromString("8.0");
    options.macosMinimumOs = DottedVersion.fromString("9.0");
    options.iosCpu = "ioscpu1";
    options.appleCrosstoolTop = Label.parseAbsoluteUnchecked("//apple/crosstool:top");
    options.applePlatformType = ApplePlatform.PlatformType.WATCHOS;
    options.appleSplitCpu = "appleSplitCpu1";
    options.configurationDistinguisher =
        AppleConfiguration.ConfigurationDistinguisher.APPLEBIN_TVOS;
    options.iosMultiCpus = ImmutableList.of("iosMultiCpu1", "iosMultiCpu2");
    options.watchosCpus = ImmutableList.of("watchosCpu1", "watchosCpu2", "watchosCpu3");
    options.tvosCpus = ImmutableList.of("tvosCpu1");
    options.macosCpus = ImmutableList.of();
    options.defaultProvisioningProfile = Label.parseAbsoluteUnchecked("//default/provisioning");
    options.xcodeVersionConfig = Label.parseAbsoluteUnchecked("//xcode/version:config");
    options.xcodeToolchain = "xcodeToolchain1";
    options.appleBitcodeMode = AppleCommandLineOptions.AppleBitcodeMode.EMBEDDED_MARKERS;
    options.enableAppleCrosstoolTransition = false;
    options.targetUsesAppleCrosstool = true;
    return new AppleConfiguration(
        options,
        "iosCpuArg",
        DottedVersion.fromString("10.0"),
        DottedVersion.fromString("11.0"),
        DottedVersion.fromString("12.0"),
        DottedVersion.fromString("13.0"),
        DottedVersion.fromString("14.0"),
        DottedVersion.fromString("15.0"),
        DottedVersion.fromString("16.0"),
        DottedVersion.fromString("17.0"),
        DottedVersion.fromString("18.0"));
  }
}
