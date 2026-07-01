// Copyright 2026 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.sandbox;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.sandbox.proto.SandboxProto.Confinement;
import com.google.devtools.build.lib.util.OS;

/**
 * The confinement mechanisms Bazel can enforce for a sandbox backend on this host. Single source of
 * truth for both the list Bazel advertises to a backend (in {@code Negotiate}) and the set it
 * enforces a backend's choice against (a {@code Created.confinement} outside it fails closed).
 */
final class SandboxBackendConfinement {
  private SandboxBackendConfinement() {}

  /** What Bazel can enforce on this OS. Always includes {@code NONE}. */
  static ImmutableList<Confinement> supportedOnThisPlatform() {
    return switch (OS.getCurrent()) {
      case DARWIN ->
          ImmutableList.of(Confinement.CONFINEMENT_NONE, Confinement.CONFINEMENT_SEATBELT);
      case LINUX ->
          ImmutableList.of(
              Confinement.CONFINEMENT_NONE, Confinement.CONFINEMENT_LINUX_NAMESPACES);
      default -> ImmutableList.of(Confinement.CONFINEMENT_NONE);
    };
  }

  /**
   * The confinement to apply when the backend leaves it unspecified: the strongest mechanism
   * available on this host (no OS confinement where none exists).
   */
  static Confinement platformDefault() {
    return switch (OS.getCurrent()) {
      case DARWIN -> Confinement.CONFINEMENT_SEATBELT;
      case LINUX -> Confinement.CONFINEMENT_LINUX_NAMESPACES;
      default -> Confinement.CONFINEMENT_NONE;
    };
  }
}
