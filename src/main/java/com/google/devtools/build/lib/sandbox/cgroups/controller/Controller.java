package com.google.devtools.build.lib.sandbox.cgroups.controller;

import com.google.common.base.Defaults;
import com.google.common.base.Suppliers;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;

import java.io.IOException;
import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.nio.file.Path;
import java.util.function.Supplier;

public interface Controller {
    default boolean isLegacy() throws IOException {
        return !getPath().resolve("cgroup.controllers").toFile().exists();
    }

    default boolean exists() {
        return getPath().toFile().isDirectory();
    }

    Path getPath();

    interface Memory extends Controller {
        Memory child(String name) throws IOException;
        void setMaxBytes(long bytes) throws IOException;
        long getMaxBytes() throws IOException;
        long getUsageInBytes() throws IOException;
    }
    interface Cpu extends Controller {
        Cpu child(String name) throws IOException;
        void setCpus(double cpus) throws IOException;
        long getCpus() throws IOException;
    }
}