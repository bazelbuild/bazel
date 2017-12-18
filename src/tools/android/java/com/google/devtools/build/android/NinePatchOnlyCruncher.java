// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android;

import com.android.SdkConstants;
import com.android.ide.common.internal.AaptCruncher;
import com.android.ide.common.internal.PngException;
import com.android.ide.common.process.ProcessExecutor;
import com.android.ide.common.process.ProcessOutputHandler;
import com.google.common.io.Files;
import java.io.File;
import java.io.IOException;

/** A wrapper around a PNG cruncher that only processes nine-patch PNGs. */
public class NinePatchOnlyCruncher extends AaptCruncher {

  public NinePatchOnlyCruncher(
      String aaptLocation,
      ProcessExecutor processExecutor,
      ProcessOutputHandler processOutputHandler) {
    super(aaptLocation, processExecutor, processOutputHandler);
  }

  /**
   * Runs the cruncher on a single file (or copies the file if no crunching is needed).
   *
   * @param from the file to process
   * @param to the output file
   */
  @Override
  public void crunchPng(int key, File from, File to) throws PngException {
    if (from.getPath().endsWith(SdkConstants.DOT_9PNG)) {
      super.crunchPng(0, from, to);
    } else {
      try {
        Files.copy(from, to);
      } catch (IOException e) {
        throw new PngException(e);
      }
    }
  }
}
