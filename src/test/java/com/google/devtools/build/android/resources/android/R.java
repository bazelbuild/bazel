// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.resources.android;

/** Example framework SDK with resources that have not had their resource ids finalized yet. */
public final class R {
  /** Attribute resources. */
  public static final class Attr {
    public static int staged = 0x0101ff00;
    public static int stagedOther = 0x0101ff01;

    private Attr() {}
  }

  private R() {}
}
