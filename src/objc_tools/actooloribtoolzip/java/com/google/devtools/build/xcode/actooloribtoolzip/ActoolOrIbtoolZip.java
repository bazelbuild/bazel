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

package com.google.devtools.build.xcode.actooloribtoolzip;

import com.google.common.base.Function;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.xcode.zippingoutput.Arguments;
import com.google.devtools.build.xcode.zippingoutput.Wrapper;
import com.google.devtools.build.xcode.zippingoutput.Wrappers;

import java.io.File;
import java.io.IOException;

/**
 * A tool which wraps actool or ibtool, by running actool/ibtool and zipping its output. See the
 * JavaDoc for {@link Wrapper} for more information.
 */
public class ActoolOrIbtoolZip implements Wrapper {

  private static final Function<String, String> CANONICAL_PATH =
      new Function<String, String>() {
    @Override
    public String apply(String path) {
      File file = new File(path);
      if (file.exists()) {
        try {
          return file.getCanonicalPath();
        } catch (IOException e) {
          // Pass through to return raw path
        }
      }
      return path;
    }
  };

  @Override
  public String name() {
    return "ActoolOrIbtoolZip";
  }

  @Override
  public String subtoolName() {
    return "actool/ibtool";
  }

  @Override
  public Iterable<String> subCommand(Arguments args, String outputDirectory) {
    return new ImmutableList.Builder<String>()
        .add(args.subtoolCmd())
        .add("--compile")
        .add(outputDirectory)
        // actool munges paths in some way which doesn't work if one of the directories in the path
        // is a symlink.
        .addAll(Iterables.transform(args.subtoolExtraArgs(), CANONICAL_PATH))
        .build();
  }

  public static void main(String[] args) throws IOException, InterruptedException {
    Wrappers.execute(args, new ActoolOrIbtoolZip());
  }

  @Override
  public boolean outputDirectoryMustExist() {
    return true;
  }
}
