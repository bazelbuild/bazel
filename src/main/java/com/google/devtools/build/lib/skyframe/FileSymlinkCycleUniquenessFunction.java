// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyValue;

/** A {@link SkyFunction} that has the side effect of reporting a file symlink cycle. */
public class FileSymlinkCycleUniquenessFunction
    extends AbstractChainUniquenessFunction<RootedPath> {
  @Override
  protected SkyValue getDummyValue() {
    return FileSymlinkCycleUniquenessValue.INSTANCE;
  }

  @Override
  protected String elementToString(RootedPath elt) {
    return elt.asPath().toString();
  }

  @Override
  protected String getConciseDescription() {
    return "circular symlinks";
  }

  @Override
  protected String getHeaderMessage() {
    return "[start of symlink cycle]";
  }

  @Override
  protected String getFooterMessage() {
    return "[end of symlink cycle]";
  }
}
