/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.tonicsystems.jarjar.transform;

import java.io.File;
import java.lang.reflect.Method;
import java.net.URL;
import java.net.URLClassLoader;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;
import javax.annotation.Nonnull;
import static org.junit.Assert.*;

/**
 *
 * @author shevek
 */
public class AbstractJarTransformerTest {

    @Nonnull
    protected static File newJar(String propertyName) {
        return new File(System.getProperty(propertyName));
    }
    protected final File jar = newJar("jar");
    protected final File[] jars = new File[]{
        newJar("jar0"),
        newJar("jar1"),
        newJar("jar2"),
        newJar("jar3")
    };

    @Nonnull
    protected Method getMethod(@Nonnull File file, @Nonnull String className, @Nonnull String methodName, @Nonnull Class<?>... parameterTypes) throws Exception {
        URLClassLoader loader = new URLClassLoader(new URL[]{file.toURI().toURL()}, getClass().getClassLoader());
        Class<?> c = loader.loadClass(className);
        return c.getMethod("main", parameterTypes);
    }

    protected static void assertContains(@Nonnull JarFile jarFile, @Nonnull String resourceName) {
        JarEntry jarEntry = jarFile.getJarEntry(resourceName);
        assertNotNull("JarFile " + jarFile + " does not contain entry " + resourceName, jarEntry);
    }

    protected static void assertNotContains(@Nonnull JarFile jarFile, @Nonnull String resourceName) {
        JarEntry jarEntry = jarFile.getJarEntry(resourceName);
        assertNull("JarFile " + jarFile + " does contains unexpected entry " + resourceName, jarEntry);
    }

}
