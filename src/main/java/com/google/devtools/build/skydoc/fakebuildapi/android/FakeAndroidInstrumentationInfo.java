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

package com.google.devtools.build.skydoc.fakebuildapi.android;

import com.google.devtools.build.lib.skylarkbuildapi.android.AndroidInstrumentationInfoApi;
import com.google.devtools.build.lib.skylarkbuildapi.android.ApkInfoApi;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Printer;

/** Fake implementation of {@link AndroidInstrumentationInfoApi}. */
public class FakeAndroidInstrumentationInfo
    implements AndroidInstrumentationInfoApi<ApkInfoApi<?>> {

  @Override
  public ApkInfoApi<?> getTarget() {
    return null;
  }

  @Override
  public String toProto() throws EvalException {
    return "";
  }

  @Override
  public String toJson() throws EvalException {
    return "";
  }

  @Override
  public void repr(Printer printer) {}

  /** Fake implementation of {@link AndroidInstrumentationInfoApiProvider}. */
  public static class FakeAndroidInstrumentationInfoProvider
      implements AndroidInstrumentationInfoApiProvider<ApkInfoApi<?>> {

    @Override
    public AndroidInstrumentationInfoApi<ApkInfoApi<?>> createInfo(ApkInfoApi<?> target)
        throws EvalException {
      return new FakeAndroidInstrumentationInfo();
    }

    @Override
    public void repr(Printer printer) {}
  }
}
