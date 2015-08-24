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
package com.google.devtools.build.lib.util;

import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import javax.annotation.concurrent.Immutable;

/**
 * A set of FileTypes for grouped matching.
 */
@Immutable
public class FileTypeSet implements Predicate<String> {
  private final ImmutableSet<FileType> types;

  /** A set that matches all files. */
  public static final FileTypeSet ANY_FILE =
      new FileTypeSet() {
        @Override
        public String toString() {
          return "any files";
        }
        @Override
        public boolean matches(String filename) {
          return true;
        }
        @Override
        public List<String> getExtensions() {
          return ImmutableList.<String>of();
        }
      };

  /** A predicate that matches no files. */
  public static final FileTypeSet NO_FILE =
      new FileTypeSet(ImmutableList.<FileType>of()) {
        @Override
        public String toString() {
          return "no files";
        }
        @Override
        public boolean matches(String filename) {
          return false;
        }
      };

  private FileTypeSet() {
    this.types = null;
  }

  private FileTypeSet(FileType... fileTypes) {
    this.types = ImmutableSet.copyOf(fileTypes);
  }

  private FileTypeSet(Iterable<FileType> fileTypes) {
    this.types = ImmutableSet.copyOf(fileTypes);
  }

  /**
   * Returns a set that matches only the provided {@code fileTypes}.
   *
   * <p>If {@code fileTypes} is empty, the returned predicate will match no files.
   */
  public static FileTypeSet of(FileType... fileTypes) {
    if (fileTypes.length == 0) {
      return FileTypeSet.NO_FILE;
    } else {
      return new FileTypeSet(fileTypes);
    }
  }

  /**
   * Returns a set that matches only the provided {@code fileTypes}.
   *
   * <p>If {@code fileTypes} is empty, the returned predicate will match no files.
   */
  public static FileTypeSet of(Iterable<FileType> fileTypes) {
    if (Iterables.isEmpty(fileTypes)) {
      return FileTypeSet.NO_FILE;
    } else {
      return new FileTypeSet(fileTypes);
    }
  }

  /** Returns a copy of this {@link FileTypeSet} including the specified `fileTypes`. */
  public FileTypeSet including(FileType... fileTypes) {
    return new FileTypeSet(Iterables.concat(this.types, Arrays.asList(fileTypes)));
  }

  /** Returns true if the filename can be matched by any FileType in this set. */
  public boolean matches(String filename) {
    int slashIndex = filename.lastIndexOf('/');
    if (slashIndex != -1) {
      filename = filename.substring(slashIndex + 1);
    }
    for (FileType type : types) {
      if (type.apply(filename)) {
        return true;
      }
    }
    return false;
  }

  /** Returns true if this predicate matches nothing. */
  public boolean isNone() {
    return this == FileTypeSet.NO_FILE;
  }

  @Override
  public boolean apply(String filename) {
    return matches(filename);
  }

  /** Returns the list of possible file extensions for this file type. Can be empty. */
  public List<String> getExtensions() {
    List<String> extensions = new ArrayList<>();
    for (FileType type : types) {
      extensions.addAll(type.getExtensions());
    }
    return extensions;
  }

  @Override
  public String toString() {
    return StringUtil.joinEnglishList(getExtensions());
  }
}
