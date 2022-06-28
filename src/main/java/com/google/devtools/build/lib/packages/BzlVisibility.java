// Copyright 2022 The Bazel Authors. All rights reserved.
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

/** A visibility level governing the loading of a .bzl module. */
// TODO(brandjon): Replace with a proper allowlist having the same granularity as target
// visibility (i.e. package path specifications).
public enum BzlVisibility {
  /** Loadable by everyone (default). */
  PUBLIC,

  /** Loadable by BUILD and .bzl files within the same package (not subpackages). */
  PRIVATE
}
