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

package com.google.devtools.build.lib.packages;

import com.google.devtools.build.lib.cmdline.PackageIdentifier;

import java.io.IOException;

/**
 *  Exception indicating an attempt to access a package which is not found or
 *  does not exist.
 */
public class BuildFileNotFoundException extends NoSuchPackageException {

  public BuildFileNotFoundException(PackageIdentifier packageIdentifier, String message) {
    super(packageIdentifier, message);
  }

  public BuildFileNotFoundException(PackageIdentifier packageIdentifier, String message,
      IOException cause) {
    super(packageIdentifier, message, cause);
  }
}
