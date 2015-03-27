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

package com.google.devtools.build.xcode.actoolzip;

import com.google.common.base.Function;
import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.xcode.zippingoutput.Arguments;
import com.google.devtools.build.xcode.zippingoutput.Wrapper;
import com.google.devtools.build.xcode.zippingoutput.Wrappers;
import com.google.devtools.build.xcode.zippingoutput.Wrappers.CommandFailedException;
import com.google.devtools.build.xcode.zippingoutput.Wrappers.OutErr;

import java.io.File;
import java.io.IOException;
import java.util.Set;

/**
 * A tool which wraps actool by running actool and zipping its output. See the JavaDoc for
 * {@link Wrapper} for more information.
 */
public class ActoolZip implements Wrapper {

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
    return "ActoolZip";
  }

  @Override
  public String subtoolName() {
    return "actool";
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
    Optional<File> infoPlistPath = replaceInfoPlistPath(args);
    try {
      OutErr outErr = Wrappers.executeCapturingOutput(args, new ActoolZip());
      if (infoPlistPath.isPresent() && !infoPlistPath.get().exists()) {
        outErr.print();
        System.exit(1);
      }
    } catch (CommandFailedException e) {
      Wrappers.handleException(e);
    }
  }

  /**
   * Absolute-ify output partial info plist's path.
   *
   * <p>actool occasionally writes the partial info plist file to the wrong directory if a
   * non-absolute path is passed as --output-partial-info-plist, so we optimistically try to
   * absolute-ify its path. This isn't caught by the "CANONICAL_PATH" transform above, because the
   * file doesn't exist at the time of flag parsing.
   *
   * <p>Modifies args in-place.
   *
   * @return new value of the output-partial-info-plist flag.
   */
  private static Optional<File> replaceInfoPlistPath(String[] args) {
    String flag = "output-partial-info-plist";
    Set<String> flagOptions = ImmutableSet.of(
        "-" + flag,
        "--" + flag);
    Optional<File> newPath = Optional.absent();
    for (int i = 0; i < args.length; ++i) {
      for (String flagOption : flagOptions) {
        String arg = args[i];
        String flagEquals = flagOption + "=";
        if (arg.startsWith(flagEquals)) {
          newPath = Optional.of(new File(arg.substring(flagEquals.length())));
          args[i] = flagEquals + newPath.get().getAbsolutePath();
        }
        if (arg.equals(flagOption) && i + 1 < args.length) {
          newPath = Optional.of(new File(args[i + 1]));
          args[i + 1] = newPath.get().getAbsolutePath();
        }
      }
    }
    return newPath;
  }

  @Override
  public boolean outputDirectoryMustExist() {
    return true;
  }
}
