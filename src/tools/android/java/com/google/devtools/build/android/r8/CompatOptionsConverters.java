// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.r8;

import com.beust.jcommander.IStringConverter;
import com.beust.jcommander.ParameterException;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;

/** JCommander-compatible options converters compatible with R8 */
final class CompatOptionsConverters {
  /**
   * Validating converter for Paths. A Path is considered valid if it resolves to a file. Compatible
   * with R8.
   */
  public static class CompatPathConverter implements IStringConverter<Path> {

    private final boolean mustExist;

    public CompatPathConverter() {
      this.mustExist = false;
    }

    protected CompatPathConverter(boolean mustExist) {
      this.mustExist = mustExist;
    }

    @Override
    public Path convert(String input) throws ParameterException {
      try {
        Path path = FileSystems.getDefault().getPath(input);
        if (mustExist && !Files.exists(path)) {
          throw new ParameterException(
              String.format("%s is not a valid path: it does not exist.", input));
        }
        return path;
      } catch (InvalidPathException e) {
        throw new ParameterException(
            String.format("%s is not a valid path: %s.", input, e.getMessage()), e);
      }
    }
  }

  /**
   * Validating converter for Paths. A Path is considered valid if it resolves to a file and exists.
   * Compatible with R8.
   */
  public static class CompatExistingPathConverter extends CompatPathConverter {
    public CompatExistingPathConverter() {
      super(true);
    }
  }

  private CompatOptionsConverters() {}
}
