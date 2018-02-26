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

package com.google.devtools.build.lib.skyframe.serialization;

/** Constants shared between {@link CodecScanner} and {@code AutoCodecProcessor}. */
public class CodecScanningConstants {
  /**
   * Name of static field in RegisteredSingleton classes. Any class whose name ends in {@link
   * #REGISTERED_SINGLETON_SUFFIX} and that has a field with this name will have this field
   * registered as a constant by {@link CodecScanner}. Generated using uuidgen and simple
   * translation of numbers to letters.
   */
  public static final String REGISTERED_SINGLETON_INSTANCE_VAR_NAME =
      "REGISTERED_SINGLETON_INSTANCE_VAR_NAME_GLFKMEBDQFHOJQKEHHQPGMNQBOBFEJADCMDP";
  /** Suffix for RegisteredSingleton classes. */
  public static final String REGISTERED_SINGLETON_SUFFIX = "RegisteredSingleton";
}
