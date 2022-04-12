// Copyright 2021 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.profiler;

/**
 * Representation of mnemonic data used by the {@link Profiler}.
 *
 * <p>This class provides some abstraction for the representation of mnemonic data in the profiler.
 * The data in itself is used to separate profiled actions into the types of actions they correspond
 * to more clearly.
 */
public class MnemonicData {
  private static MnemonicData emptyMnemonic = null;
  private final String mnemonic;

  private MnemonicData() {
    this.mnemonic = null;
  }

  MnemonicData(String mnemonic) {
    this.mnemonic = mnemonic;
  }

  /**
   * Get a mnemonic datum without any set mnemonic.
   *
   * <p>To keep the internal representation encapsulated, {@link MnemonicData#hasBeenSet} can be
   * used to check whether or not a mnemonic has been set for the datum.
   *
   * @return a singleton of an empty mnemonic datum
   */
  public static MnemonicData getEmptyMnemonic() {
    if (emptyMnemonic == null) {
      emptyMnemonic = new MnemonicData();
    }
    return emptyMnemonic;
  }

  /**
   * Check if the datum has been assigned a value.
   *
   * @return if the mnemonic value has been set
   */
  public boolean hasBeenSet() {
    return this.mnemonic != null;
  }

  /**
   * Get the mnemonic part of a category string formatted for JSON output.
   *
   * @return a string representation of the mnemonic if it has been set, otherwise null
   */
  public String getValueForJson() {
    return mnemonic;
  }
}
