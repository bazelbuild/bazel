// Copyright 2018 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.Interner;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.AbstractSkyKey;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

/**
 * Encapsulates the errors, if any, encountered when loading a specific package.
 *
 * <p>This is a change-pruning-friendly convenience {@link SkyValue} for use cases where {@link
 * Result} is sufficient.
 */
public abstract class PackageErrorMessageValue implements SkyValue {
  /** Tri-state result of loading the package. */
  public enum Result {
    /**
     * There was no error loading the package and {@link
     * com.google.devtools.build.lib.packages.Package#containsErrors} returned {@code false}.
     */
    NO_ERROR,

    /**
     * There was no error loading the package and {@link
     * com.google.devtools.build.lib.packages.Package#containsErrors} returned {@code true}.
     */
    ERROR,

    /**
     * There was a {@link com.google.devtools.build.lib.packages.NoSuchPackageException} loading the
     * package.
     */
    NO_SUCH_PACKAGE_EXCEPTION,
  }

  /** Returns the {@link Result} from loading the package. */
  public abstract Result getResult();

  /**
   * If {@code getResult().equals(NO_SUCH_PACKAGE_EXCEPTION)}, returns the error message from the
   * {@link com.google.devtools.build.lib.packages.NoSuchPackageException} encountered.
   */
  abstract String getNoSuchPackageExceptionMessage();

  static PackageErrorMessageValue ofPackageWithNoErrors() {
    return NO_ERROR_VALUE;
  }

  static PackageErrorMessageValue ofPackageWithErrors() {
    return ERROR_VALUE;
  }

  static PackageErrorMessageValue ofNoSuchPackageException(String errorMessage) {
    return new NoSuchPackageExceptionValue(errorMessage);
  }

  public static SkyKey key(PackageIdentifier pkgId) {
    return Key.create(pkgId);
  }

  @AutoCodec.VisibleForSerialization
  @AutoCodec
  static class Key extends AbstractSkyKey<PackageIdentifier> {
    private static final Interner<Key> interner = BlazeInterners.newWeakInterner();

    private Key(PackageIdentifier arg) {
      super(arg);
    }

    @AutoCodec.VisibleForSerialization
    @AutoCodec.Instantiator
    static Key create(PackageIdentifier arg) {
      return interner.intern(new Key(arg));
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.PACKAGE_ERROR_MESSAGE;
    }
  }

  @AutoCodec
  static final PackageErrorMessageValue NO_ERROR_VALUE =
      new PackageErrorMessageValue() {
        @Override
        public Result getResult() {
          return Result.NO_ERROR;
        }

        @Override
        String getNoSuchPackageExceptionMessage() {
          throw new IllegalStateException();
        }
      };

  @AutoCodec
  static final PackageErrorMessageValue ERROR_VALUE =
      new PackageErrorMessageValue() {
        @Override
        public Result getResult() {
          return Result.ERROR;
        }

        @Override
        String getNoSuchPackageExceptionMessage() {
          throw new IllegalStateException();
        }
      };

  @AutoCodec
  static class NoSuchPackageExceptionValue extends PackageErrorMessageValue {
    private final String errorMessage;

    public NoSuchPackageExceptionValue(String errorMessage) {
      this.errorMessage = errorMessage;
    }

    @Override
    public Result getResult() {
      return Result.NO_SUCH_PACKAGE_EXCEPTION;
    }

    @Override
    String getNoSuchPackageExceptionMessage() {
      return errorMessage;
    }

    @Override
    public boolean equals(Object obj) {
      if (!(obj instanceof NoSuchPackageExceptionValue)) {
        return false;
      }
      NoSuchPackageExceptionValue other = (NoSuchPackageExceptionValue) obj;
      return errorMessage.equals(other.errorMessage);
    }

    @Override
    public int hashCode() {
      return errorMessage.hashCode();
    }
  }
}
