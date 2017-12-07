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
package android.os;

/** This class is a standin for android.os.Build for tests running in a JVM */
public final class Build {

  public static final String SYSTEM_PROPERTY_NAME = "fortest.simulated.android.sdk_int";

  /** A simple mock for the real android.os.Build.VERSION */
  public static final class VERSION {

    public static final int SDK_INT;

    static {
      String sdkInt = System.getProperty(SYSTEM_PROPERTY_NAME, "0");
      SDK_INT = Integer.parseInt(sdkInt);
    }
  }
}
