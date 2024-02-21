package com.google.devtools.build.lib.sandbox.cgroups.controller;

import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.server.FailureDetails;

import java.io.IOException;
import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.nio.file.Path;

public interface Controller {
    default boolean isLegacy() throws IOException {
        return !getPath().resolve("cgroup.controllers").toFile().exists();
    }

    static <T extends Controller> T getDefault(Class<T> clazz) {
        InvocationHandler handler = new InvocationHandler() {
            @Override
            public Object invoke(Object proxy, Method method, Object[] args) throws ExecException {
                throw new ExecException("Cgroup requested by cgroups are not available!") {
                    protected FailureDetails.FailureDetail getFailureDetail(String message) {
                        return FailureDetails.FailureDetail.newBuilder().setMessage(message).build();
                    }
                };

            }
        };

        return (T) Proxy.newProxyInstance(clazz.getClassLoader(), new Class<?>[]{clazz}, handler);
    }

    default boolean exists() {
        return getPath().toFile().isDirectory();
    }

    Path getPath();

    interface Memory extends Controller {
        void setMaxBytes(long bytes) throws IOException;
        long getMaxBytes() throws IOException;
    }
    interface Cpu extends Controller {
        void setCpus(double cpus) throws IOException;
        long getCpus() throws IOException;
    }
}