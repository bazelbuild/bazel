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

import java.io.IOException;
import java.io.StringReader;
import java.util.Properties;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Class for information about an NDK release.
 */
public class NdkRelease {

  /** Key name for the revision in the source.properties file. */
  private static final String REVISION_PROPERTY = "Pkg.Revision";

  public static NdkRelease create(String releaseString) {
    if (releaseString.contains(REVISION_PROPERTY)) {
      // For NDK r11+
      return createFromSourceProperties(releaseString);
    } else {
      // For NDK r10e
      return createFromReleaseTxt(releaseString);
    }
  }

  /**
   * Creates an NdkRelease for r11+ (uses source.properties)
   */
  private static NdkRelease createFromSourceProperties(String releaseString) {
    Properties properties = new Properties();
    try {
      properties.load(new StringReader(releaseString));
    } catch (IOException e) {
      // This shouldn't happen from a StringReader.
      throw new IllegalStateException(e);
    }
    String revision = properties.getProperty(REVISION_PROPERTY);
    String[] revisionParsed = revision.split("\\.");
    return new NdkRelease(
        revision, // raw revision
        true, // isValid
        revisionParsed[0], // major revision
        revisionParsed[1], // minor revision
        null, // release candidate
        true // is64 bit. 32-bit NDKs are provided for only windows.
    );
  }

  /**
   * Creates an NdkRelease pre-r11 (used RELEASE.TXT)
   */
  private static NdkRelease createFromReleaseTxt(String revisionString) {
    // NDK release should be of the format "r\d+\w?(-rc\d+)?( \(64-bit\))?", eg:
    // r8
    // r10
    // r10 (64-bit)
    // r10e
    // r10e-rc4
    // r10e-rc4 (64-bit)
    Pattern revisionRegex = Pattern.compile(
        "r(?<Mrev>\\d+)(?<mrev>\\w)?(-(?<rc>rc\\d+))?(?<s4> \\(64-bit\\))?");
    Matcher matcher = revisionRegex.matcher(revisionString);
    boolean isValid = matcher.matches();

    if (isValid) {
      return new NdkRelease(
          revisionString,
          isValid,
          matcher.group("Mrev"), /* major revision */
          matcher.group("mrev"), /* minor revision */
          matcher.group("rc"), /* releaseCandidate */
          matcher.group("s4") != null /* is64Bit */);
    } else {
      return new NdkRelease(revisionString, false, null, null, null, false);
    }
  }

  public final String rawRelease;
  public final boolean isValid;

  public final String majorRevision;
  public final String minorRevision;
  public final String releaseCandidate;
  public final boolean is64Bit;

  private NdkRelease(
      String rawRelease,
      boolean isValid,
      String majorRevision,
      String minorRevision,
      String releaseCandidate,
      boolean is64Bit) {
    this.rawRelease = rawRelease;
    this.isValid = isValid;
    this.majorRevision = majorRevision;
    this.minorRevision = minorRevision;
    this.releaseCandidate = releaseCandidate;
    this.is64Bit = is64Bit;
  }

  @Override
  public String toString() {
    return rawRelease;
  }
}
