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

package com.google.devtools.build.xcode.ibtoolzip;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.xcode.zippingoutput.Arguments;
import com.google.devtools.build.xcode.zippingoutput.Wrapper;
import com.google.devtools.build.xcode.zippingoutput.Wrappers;

import java.io.IOException;

/**
 * A tool which wraps ibtool by running ibtool and zipping its output. See the JavaDoc for
 * {@link Wrapper} for more information.
 */
public class IbtoolZip implements Wrapper {

  @Override
  public String name() {
    return "IbtoolZip";
  }

  @Override
  public String subtoolName() {
    return "ibtool";
  }

  @Override
  public Iterable<String> subCommand(Arguments args, String outputDirectory) {
    // If we are running into problems figuring out ibtool issues, there are a couple
    // of env variables that may help. Both of the following must be set to work.
    //   IBToolDebugLogFile=<OUTPUT FILE PATH>
    //   IBToolDebugLogLevel=4
    // you may also see if
    //   IBToolNeverDeque=1
    // helps.

    // IBTool needs to have absolute paths sent to it, so we call CANONICALIZE_IF_PATH
    // on all of them.
    // Radar 21045660 ibtool has difficulty dealing with relative paths.
    return new ImmutableList.Builder<String>()
        .add(args.subtoolCmd())
        .add("--errors")
        .add("--warnings")
        .add("--notices")
        .add("--auto-activate-custom-fonts")
        .add("--output-format").add("human-readable-text")
        .add("--compile")
        .add(Wrappers.CANONICALIZE_IF_PATH.apply(outputDirectory))
        .addAll(Iterables.transform(args.subtoolExtraArgs(), Wrappers.CANONICALIZE_IF_PATH))
        .build();
  }

  public static void main(String[] args) throws IOException, InterruptedException {
    Wrappers.executePipingOutput(args, new IbtoolZip());
  }

  @Override
  public boolean outputDirectoryMustExist() {
    return true;
  }
}
