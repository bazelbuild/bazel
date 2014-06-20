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
package com.google.devtools.build.lib.skyframe;

import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.Node;
import com.google.devtools.build.skyframe.NodeKey;

/**
 * A node that represents a package lookup result.
 *
 * <p>Package lookups will always produce a node. On success, the {@code #getRoot} returns the
 * package path root under which the package resides and the package's BUILD file is guaranteed to
 * exist; on failure, {@code #getErrorReason} and {@code #getErrorMsg} describe why the package
 * doesn't exist.
 *
 * <p>Implementation detail: we use inheritance here to optimize for memory usage.
 */
abstract class PackageLookupNode implements Node {

  enum ErrorReason {
    // There is no BUILD file.
    NO_BUILD_FILE,

    // The package name is invalid.
    INVALID_PACKAGE_NAME,

    // The package is considered deleted because of --deleted_packages.
    DELETED_PACKAGE
  }

  protected PackageLookupNode() {
  }

  public static PackageLookupNode success(Path root) {
    return new SuccessfulPackageLookupNode(root);
  }

  public static PackageLookupNode noBuildFile() {
    return NoBuildFilePackageLookupNode.INSTANCE;
  }

  public static PackageLookupNode invalidPackageName(String errorMsg) {
    return new InvalidNamePackageLookupNode(errorMsg);
  }

  public static PackageLookupNode deletedPackage() {
    return DeletedPackageLookupNode.INSTANCE;
  }

  /**
   * For a successful package lookup, returns the root (package path entry) that the package
   * resides in.
   */
  public abstract Path getRoot();

  /**
   * Returns whether the package lookup was successful.
   */
  public abstract boolean packageExists();

  /**
   * For an unsuccessful package lookup, gets the reason why {@link #packageExists} returns
   * {@code false}.
   */
  abstract ErrorReason getErrorReason();

  /**
   * For an unsuccessful package lookup, gets a detailed error message for {@link #getErrorReason}
   * that is suitable for reporting to a user.
   */
  abstract String getErrorMsg();

  static NodeKey key(PathFragment directory) {
    return new NodeKey(NodeTypes.PACKAGE_LOOKUP, directory);
  }

  private static class SuccessfulPackageLookupNode extends PackageLookupNode {

    private final Path root;

    private SuccessfulPackageLookupNode(Path root) {
      this.root = root;
    }

    @Override
    public boolean packageExists() {
      return true;
    }

    @Override
    public Path getRoot() {
      return root;
    }

    @Override
    ErrorReason getErrorReason() {
      throw new IllegalStateException();
    }

    @Override
    String getErrorMsg() {
      throw new IllegalStateException();
    }
  }

  private abstract static class UnsuccessfulPackageLookupNode extends PackageLookupNode {

    @Override
    public boolean packageExists() {
      return false;
    }

    @Override
    public Path getRoot() {
      throw new IllegalStateException();
    }
  }

  private static class NoBuildFilePackageLookupNode extends UnsuccessfulPackageLookupNode {

    public static final NoBuildFilePackageLookupNode INSTANCE = new NoBuildFilePackageLookupNode();

    private NoBuildFilePackageLookupNode() {
    }

    @Override
    ErrorReason getErrorReason() {
      return ErrorReason.NO_BUILD_FILE;
    }

    @Override
    String getErrorMsg() {
      return "BUILD file not found on package path";
    }
  }

  private static class InvalidNamePackageLookupNode extends UnsuccessfulPackageLookupNode {

    private final String errorMsg;

    private InvalidNamePackageLookupNode(String errorMsg) {
      this.errorMsg = errorMsg;
    }

    @Override
    ErrorReason getErrorReason() {
      return ErrorReason.INVALID_PACKAGE_NAME;
    }

    @Override
    String getErrorMsg() {
      return errorMsg;
    }

    @Override
    public boolean equals(Object obj) {
      if (!(obj instanceof InvalidNamePackageLookupNode)) {
        return false;
      }
      InvalidNamePackageLookupNode other = (InvalidNamePackageLookupNode) obj;
      return errorMsg.equals(other.errorMsg);
    }

    @Override
    public int hashCode() {
      return errorMsg.hashCode();
    }
  }

  private static class DeletedPackageLookupNode extends UnsuccessfulPackageLookupNode {

    public static final DeletedPackageLookupNode INSTANCE = new DeletedPackageLookupNode();

    private DeletedPackageLookupNode() {
    }

    @Override
    ErrorReason getErrorReason() {
      return ErrorReason.DELETED_PACKAGE;
    }

    @Override
    String getErrorMsg() {
      return "Package is considered deleted due to --deleted_packages";
    }
  }
}
