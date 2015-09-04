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

package com.google.devtools.build.lib.shell;

/**
 * <p>A simple implementation of {@link KillableObserver} which can be told
 * explicitly to kill its {@link Killable} by calling {@link #kill()}. This
 * is the sort of functionality that callers might expect to find available
 * on the {@link Command} class.</p>
 *
 * <p>Note that this class can only observe one {@link Killable} at a time;
 * multiple instances should be used for concurrent calls to
 * {@link Command#execute(byte[], KillableObserver, boolean)}.</p>
 */
public final class SimpleKillableObserver implements KillableObserver {

  private Killable killable;

  /**
   * Does nothing except store a reference to the given {@link Killable}.
   *
   * @param killable {@link Killable} to kill
   */
  @Override
  public synchronized void startObserving(final Killable killable) {
    this.killable = killable;
  }

  /**
   * Forgets reference to {@link Killable} provided to
   * {@link #startObserving(Killable)}
   */
  @Override
  public synchronized void stopObserving(final Killable killable) {
    if (!this.killable.equals(killable)) {
      throw new IllegalStateException("start/stopObservering called with " +
                                      "different Killables");
    }
    this.killable = null;
  }

  /**
   * Calls {@link Killable#kill()} on the saved {@link Killable}.
   */
  public synchronized void kill() {
    if (killable != null) {
      killable.kill();
    }
  }
}
