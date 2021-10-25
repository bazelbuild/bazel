package com.google.devtools.build.lib.profiler;

/** Representation of mnemonic data used by the {@link Profiler}.
 *
 * <p>This class provides some abstraction for the representation of mnemonic
 * data in the profiler. The data in itself is used to separate profiled
 * actions into the types of actions they correspond to more clearly.
 */
public class MnemonicData {
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
   * To keep the internal representation encapsulated, {@link MnemonicData#hasBeenSet}
   * can be used to check whether or not a mnemonic has been set for the datum.
   *
   * @return an empty mnemonic datum
   */
  public static MnemonicData getEmptyMnemonic() {
    return new MnemonicData();
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
  public String getJsonCategory() {
    return mnemonic;
  }
}
