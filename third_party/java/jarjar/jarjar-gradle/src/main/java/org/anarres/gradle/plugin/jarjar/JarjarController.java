/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.anarres.gradle.plugin.jarjar;

import groovy.lang.Closure;
import groovy.lang.GroovyObjectSupport;
import java.util.concurrent.atomic.AtomicInteger;
import javax.annotation.Nonnull;
import org.gradle.api.Project;
import org.gradle.api.artifacts.Dependency;
import org.gradle.api.file.FileCollection;
import org.gradle.api.internal.ClosureBackedAction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This object appears as 'jarjar' in the project extensions.
 *
 * @author shevek
 */
public class JarjarController extends GroovyObjectSupport {

    private static final Logger LOG = LoggerFactory.getLogger(JarjarController.class);
    private static final AtomicInteger SEQ = new AtomicInteger(0);
    private final Project project;

    public JarjarController(@Nonnull Project project) {
        this.project = project;
    }

    @Nonnull
    public Dependency dependency(Object dependencyNotation, Closure configurationClosure) {
        Dependency d = project.getDependencies().create(dependencyNotation, configurationClosure);
        LOG.info("sub is " + d);
        return new JarjarDependency(d);
    }

    @Nonnull
    public Dependency dependency(Object dependencyNotation) {
        return dependency(dependencyNotation, Closure.IDENTITY);
    }

    @Nonnull
    public FileCollection repackage(@Nonnull String name, @Nonnull Closure<?> c) {
        JarjarTask jarjar = project.getTasks().create(
                name,
                JarjarTask.class,
                new ClosureBackedAction<JarjarTask>(c));
        return jarjar.getOutputs().getFiles();
    }

    @Nonnull
    public FileCollection repackage(@Nonnull Closure<?> c) {
        return repackage("jarjar-" + SEQ.getAndIncrement(), c);
    }

}
