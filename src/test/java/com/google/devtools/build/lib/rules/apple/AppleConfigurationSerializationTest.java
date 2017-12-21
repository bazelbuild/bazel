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
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.skyframe.serialization.testutils.AbstractObjectCodecTest;
import com.google.devtools.common.options.OptionsParsingException;
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

  private static AppleConfiguration[] subject() {
    AppleCommandLineOptions firstOptions = new AppleCommandLineOptions();
    firstOptions.mandatoryMinimumVersion = false;
    firstOptions.iosSdkVersion = DottedVersion.fromString("2.0");
    firstOptions.watchOsSdkVersion = DottedVersion.fromString("3.0");
    firstOptions.tvOsSdkVersion = DottedVersion.fromString("4.0");
    firstOptions.macOsSdkVersion = DottedVersion.fromString("5.0");
    firstOptions.iosMinimumOs = DottedVersion.fromString("6.1beta3.7");
    firstOptions.watchosMinimumOs = DottedVersion.fromString("7.0");
    firstOptions.tvosMinimumOs = DottedVersion.fromString("8.0");
    firstOptions.macosMinimumOs = DottedVersion.fromString("9.0");
    firstOptions.iosCpu = "ioscpu1";
    firstOptions.appleCrosstoolTop = Label.parseAbsoluteUnchecked("//apple/crosstool:top");
    firstOptions.applePlatformType = ApplePlatform.PlatformType.WATCHOS;
    firstOptions.appleSplitCpu = "appleSplitCpu1";
    firstOptions.configurationDistinguisher =
        AppleConfiguration.ConfigurationDistinguisher.APPLEBIN_TVOS;
    firstOptions.iosMultiCpus = ImmutableList.of("iosMultiCpu1", "iosMultiCpu2");
    firstOptions.watchosCpus = ImmutableList.of("watchosCpu1", "watchosCpu2", "watchosCpu3");
    firstOptions.tvosCpus = ImmutableList.of("tvosCpu1");
    firstOptions.macosCpus = ImmutableList.of();
    firstOptions.defaultProvisioningProfile =
        Label.parseAbsoluteUnchecked("//default/provisioning");
    firstOptions.xcodeVersionConfig = Label.parseAbsoluteUnchecked("//xcode/version:config");
    firstOptions.appleBitcodeMode = AppleCommandLineOptions.AppleBitcodeMode.EMBEDDED_MARKERS;
    firstOptions.enableAppleCrosstoolTransition = false;
    firstOptions.targetUsesAppleCrosstool = true;
    firstOptions.xcodeVersion = "1.0";
    try {
      return new AppleConfiguration[] {
        new AppleConfiguration(
            firstOptions,
            "iosCpuArg"),
        AppleConfiguration.create(
            BuildOptions.of(ImmutableList.of(AppleCommandLineOptions.class))
                .get(AppleCommandLineOptions.class),
            "another cpu")
      };
    } catch (OptionsParsingException e) {
      throw new IllegalStateException(e);
    }
  }
}
