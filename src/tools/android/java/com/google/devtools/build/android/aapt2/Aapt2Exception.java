// Copyright 2017 The Bazel Authors. All rights reserved.
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
// Copyright 2017 The Bazel Authors. All rights reserved.

package com.google.devtools.build.android.aapt2;

/** Represents an error found during an aapt2 execution pass. */
public abstract class Aapt2Exception extends RuntimeException {
  protected Aapt2Exception(Throwable e) {
    super(e);
  }

  public Aapt2Exception() {
    super();
  }
}
