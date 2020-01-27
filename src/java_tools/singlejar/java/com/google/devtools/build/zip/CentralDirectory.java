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

package com.google.devtools.build.zip;

import java.io.IOException;
import java.io.OutputStream;

class CentralDirectory {
  /**
   * Writes the central directory to an output stream for the specified {@link ZipFileData}.
   */
  static void write(ZipFileData file, boolean allowZip64, OutputStream stream)
      throws IOException {
    long directorySize = 0;
    byte[] buf = new byte[CentralDirectoryFileHeader.FIXED_DATA_SIZE];
    for (ZipFileEntry entry : file.getEntries()) {
      directorySize += CentralDirectoryFileHeader.write(entry, file, allowZip64, buf, stream);
    }
    file.setCentralDirectorySize(directorySize);
    if (file.isZip64() && allowZip64) {
      file.setZip64EndOfCentralDirectoryOffset(
          file.getCentralDirectoryOffset() + file.getCentralDirectorySize());
      stream.write(Zip64EndOfCentralDirectory.create(file));
      stream.write(Zip64EndOfCentralDirectoryLocator.create(file));
    }
    stream.write(EndOfCentralDirectoryRecord.create(file, allowZip64));
  }
}
