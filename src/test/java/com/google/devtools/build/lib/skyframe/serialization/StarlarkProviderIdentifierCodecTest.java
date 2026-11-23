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

package com.google.devtools.build.lib.skyframe.serialization;

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.StarlarkProvider;
import com.google.devtools.build.lib.packages.StarlarkProviderIdentifier;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import net.starlark.java.eval.SymbolGenerator;
import net.starlark.java.syntax.Location;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Basic tests for {@link AdvertisedProviderSetCodec}. */
@RunWith(JUnit4.class)
public class StarlarkProviderIdentifierCodecTest {

  // TODO(bazel-team): Instead of constructing actual providers and grabbing their keys to obtain
  // subjects, maybe just make the constructor for NativeKey public, so we can just generate keys
  // independent of providers. This may also help us eliminate the helper methods
  // BuiltinProvider.getSerializedRepresentationForNativeKey() and its complement.

  private static final Provider NATIVE = new BuiltinProvider<StructImpl>("np", StructImpl.class) {};

  private static final StarlarkProvider STARLARK =
      StarlarkProvider.builder(Location.BUILTIN)
          .buildWithIdentityToken(SymbolGenerator.create("test").generate());

  static {
    try {
      STARLARK.export(
          ev -> {},
          Label.create("foo/bar", "x.bzl"),
          "barfoo",
          Location.fromFile("/workspace/foo/bar/x.bzl"));
    } catch (LabelSyntaxException e) {
      throw new AssertionError(e);
    }
  }

  @Test
  public void objectCodecTests() throws Exception {
    new SerializationTester(
            StarlarkProviderIdentifier.forKey(NATIVE.getKey()),
            StarlarkProviderIdentifier.forKey(STARLARK.getKey()))
        .runTests();
  }
}
