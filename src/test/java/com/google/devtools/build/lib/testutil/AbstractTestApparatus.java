// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.testutil;

import com.google.common.annotations.VisibleForTesting;

/**
 * An abstract super class for test apparatus implementations. This class
 * has methods for transitioning between the initialized and the
 * uninitialized state, and provides some utility methods that allows
 * methods in subclasses to assert that they are in a particular state.
 *
 * An apparatus starts uninitialized; it gets initialized by {@link #init()}.
 * Prior to initialization, the client code may change its settings by
 * calling the setter methods (implemented in subclasses of this class),
 * which determine how the apparatus creates its state during
 * {@link #initializationHook()}. For convenience, the settings are
 * preinitialized with sensible defaults.
 *
 * Once initialized, the state maintained by the apparatus is considered
 * immutable, and subsequent calls to the setters result in
 * {@link IllegalStateException} (usually delivered via
 * {@link #ensureIsUninitialized()}).
 *
 * Accessor methods return the state maintained by the apparatus, and trigger
 * initialization if it hasn't happened yet (by calling {@link #init()}).
 *
 * Other utility methods may assume that the apparatus has been initialized
 * by calling {@link #ensureIsInitialized()}. Usually these methods will
 * be methods that examine the state maintained by the apparatus; e.g.
 * assertions in a test case. Requiring initialization prior to calling such
 * methods is not strictly necessary - it's a way to detect programming errors:
 * Usually, the programmer should better call some accessor to create some
 * state, rather than make assertions on an empty apparatus.
 */
public abstract class AbstractTestApparatus {

  private boolean initialized;

  /**
   * Initializes the apparatus if it hasn't been initialized yet. This method
   * should be called from any apparatus method that wants to do lazy
   * initialization.
   */
  @VisibleForTesting
  public final void init() {
    if (initialized) {
      return;
    }
    initializationHook();
    initialized = true;
  }

  /**
   * Subclasses must implement this method to initialize the state maintained
   * by this apparatus. Throughout the lifetime of an apparatus instance,
   * the method will only be called once.
   */
  protected abstract void initializationHook();

  /**
   * An assertion to be used by methods which require the apparatus to be
   * initialized, i.e. that its state is ready for use.
   */
  @VisibleForTesting
  public final void ensureIsInitialized() {
    if (!initialized) {
      String msg = "Precondition violated: Apparatus must be initialized";
      throw new IllegalStateException(msg);
    }
  }

  /**
   * An assertion to be used by methods which require the apparatus to be
   * uninitialized, i.e. that its state is not yet ready for use.
   */
  @VisibleForTesting
  public final void ensureIsUninitialized() {
    if (initialized) {
      String msg = "Precondition violated: Apparatus must be uninitialized";
      throw new IllegalStateException(msg);
    }
  }

}
