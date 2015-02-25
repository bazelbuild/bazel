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
package com.google.devtools.build.lib.unix;

import java.io.File;
import java.net.SocketAddress;

/**
 *  An implementation of SocketAddress for naming local sockets, i.e. files in
 *  the UNIX file system.
 */
public class LocalSocketAddress extends SocketAddress {

  private final File name;

  /**
   *  Constructs a SocketAddress for the specified file.
   */
  public LocalSocketAddress(File name) {
    this.name = name;
  }

  /**
   *  Returns the filename of this local socket address.
   */
  public File getName() {
    return name;
  }

  @Override
  public String toString() {
    return name.toString();
  }

  @Override
  public boolean equals(Object other) {
    return other instanceof LocalSocketAddress &&
      ((LocalSocketAddress) other).name.equals(this.name);
  }

  @Override
  public int hashCode() {
    return name.hashCode();
  }
}
