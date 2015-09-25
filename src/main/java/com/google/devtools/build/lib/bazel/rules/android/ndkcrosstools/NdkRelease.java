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

package com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools;

import com.google.devtools.build.lib.util.CPU;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Class for information about an NDK release.
 */
public class NdkRelease {

  public static NdkRelease create(String releaseString) {
    // NDK release should be of the format "r\d+\w?(-rc\d+)?( \(64-bit\))?", eg:
    // r8
    // r10
    // r10 (64-bit)
    // r10e
    // r10e-rc4
    // r10e-rc4 (64-bit)
    Pattern releaseRegex = Pattern.compile(
        "(?<rel>r\\d+\\w?)(-(?<rc>rc\\d+))?(?<s4> \\(64-bit\\))?");
    Matcher matcher = releaseRegex.matcher(releaseString);
    boolean isValid = matcher.matches();

    if (isValid) {
      return new NdkRelease(
          releaseString,
          isValid,
          matcher.group("rel"), /* release */
          matcher.group("rc"), /* releaseCandidate */
          matcher.group("s4") != null /* is64Bit */);
    } else {
      return new NdkRelease(releaseString, false, null, null, false);
    }
  }

  /**
   * Guesses the bit-ness of the NDK based on the current platform.
   */
  static NdkRelease guessBitness(String baseReleaseString) {
    NdkRelease baseRelease = create(baseReleaseString);
    boolean is64Bit = (CPU.getCurrent() == CPU.X86_64);
    return new NdkRelease(
        baseRelease.rawRelease + (is64Bit ? " (64-bit)" : ""),
        baseRelease.isValid,
        baseRelease.release,
        baseRelease.releaseCandidate,
        is64Bit);
  }

  public final String rawRelease;
  public final boolean isValid;

  public final String release;
  public final String releaseCandidate;
  public final boolean is64Bit;

  private NdkRelease(String rawRelease, boolean isValid, String release, String releaseCandidate,
      boolean is64Bit) {
    this.rawRelease = rawRelease;
    this.isValid = isValid;
    this.release = release;
    this.releaseCandidate = releaseCandidate;
    this.is64Bit = is64Bit;
  }

  @Override
  public String toString() {
    return rawRelease;
  }
}
