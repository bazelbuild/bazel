package com.google.devtools.build.lib.remote;

import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import java.util.concurrent.atomic.AtomicInteger;

/** Collecting always-on stats on a remote run */
@ThreadSafe
public class RemoteStats {
  AtomicInteger total_spawns;
  AtomicInteger remote_cache_hit;
  AtomicInteger remote_exec;
  AtomicInteger local;

  public RemoteStats() {
    total_spawns = new AtomicInteger();
    remote_cache_hit = new AtomicInteger();
    remote_exec = new AtomicInteger();
    local = new AtomicInteger();
  }

  public void addSpawn() {
    total_spawns.incrementAndGet();
  }

  public void addRemoteCacheHit() { remote_cache_hit.incrementAndGet(); };

  // Spawns remotely executed
  public void addRemoteExec() { remote_exec.incrementAndGet(); }

  // Spawns that are intentionally executed locally (as opposed to fall back mechanism)
  public void addLocal() { local.incrementAndGet(); }

  public String Summary() {
    // We are assuming this is called after all the increments. Calling it at concurrently
    // might produce inconsistent stats.
    int remote = remote_exec.get() + remote_cache_hit.get();

    // Actions that were attempted, but were neither successful remote nor local had to be
    // fallbacks to local from remote.
    int fallback = total_spawns.get() - remote - local.get();

    String result = String.format("Remote stats: %d/%d remote actions cached, %d local",
        remote_cache_hit.get(), remote, local.get());
    if(fallback > 0) {
      result += String.format(" (%d fallback)", fallback);
    }
    return result;
  }
}
