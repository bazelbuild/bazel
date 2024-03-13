package com.google.devtools.build.lib.sandbox;

public interface Cgroup {
    int getMemoryUsageInKb();
    boolean exists();
}
