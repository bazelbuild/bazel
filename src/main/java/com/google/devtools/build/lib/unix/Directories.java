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

package com.google.devtools.build.lib.unix;

import com.google.devtools.build.lib.shell.AbnormalTerminationException;
import com.google.devtools.build.lib.shell.Command;
import com.google.devtools.build.lib.shell.CommandException;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.charset.StandardCharsets;

/**
 * Provides utility methods for working with directories in a Unix environment.
 */
public final class Directories {

  /**
   * Deletes a file or directory and all contents recursively, like {@code rm -rf file}.
   *
   * <p>If the file argument is a symbolic link, the link will be deleted but not the target of the
   * link. If the argument is a directory, symbolic links within the directory will not be followed.
   * If the argument does not exist, throws a FileNotFoundException.
   *
   * @param file the file or directory to delete
   * @throws FileNotFoundException if the file or directory does not exist
   * @throws IOException if an I/O error occurs
   */
  public static void deleteRecursively(File file) throws IOException {
    deleteRecursivelyImpl(file, true);
  }

  /**
   * Deletes a file or directory and all contents recursively, like {@code rm -rf file}, if it
   * exists.
   *
   * <p>If the file argument is a symbolic link, the link will be deleted but not the target of the
   * link. If the argument is a directory, symbolic links within the directory will not be followed.
   *
   * @param file the file or directory to delete
   * @return {@code true} if the file or directory was deleted by this method; {@code false} if the
   * file or directory could not be deleted because it did not exist
   * @throws IOException if an I/O error occurs
   */
  public static boolean deleteRecursivelyIfExists(File file) throws IOException {
    return deleteRecursivelyImpl(file, false);
  }

  private static boolean deleteRecursivelyImpl(File file, boolean failIfFileDoesNotExist)
      throws IOException {
    if (!file.exists()) {
      if (failIfFileDoesNotExist) {
        throw new FileNotFoundException(file.getPath());
      } else {
        return false;
      }
    }
    String filePath = file.getPath();
    if (!filePath.isEmpty() && filePath.charAt(0) == '-') {
      filePath = "./" + filePath;
    }
    try {
      new Command(new String[] {"/bin/rm", "-rf", filePath}).execute();
    } catch (AbnormalTerminationException e) {
      String message =
          e.getResult().getTerminationStatus() + ": " + new String(
              e.getResult().getStderr(), StandardCharsets.UTF_8);
      throw new IOException(message, e);
    } catch (CommandException e) {
      throw new IOException(e);
    }
    return true;
  }

  private Directories() {}
}
