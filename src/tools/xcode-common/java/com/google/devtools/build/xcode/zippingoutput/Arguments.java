// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.xcode.zippingoutput;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.xcode.util.Value;

/**
 * Arguments that have been parsed from a do-something-and-zip-output wrapper.
 */
public class Arguments extends Value<Arguments> {

  private final String outputZip;
  private final String bundleRoot;
  private final String subtoolCmd;
  private final ImmutableList<String> subtoolExtraArgs;

  Arguments(
      String outputZip,
      String bundleRoot,
      String subtoolCmd,
      ImmutableList<String> subtoolExtraArgs) {
    super(outputZip, bundleRoot, subtoolCmd, subtoolExtraArgs);
    this.outputZip = outputZip;
    this.bundleRoot = bundleRoot;
    this.subtoolCmd = subtoolCmd;
    this.subtoolExtraArgs = subtoolExtraArgs;
  }

  public String outputZip() {
    return outputZip;
  }

  public String bundleRoot() {
    return bundleRoot;
  }

  public String subtoolCmd() {
    return subtoolCmd;
  }

  public ImmutableList<String> subtoolExtraArgs() {
    return subtoolExtraArgs;
  }
}
