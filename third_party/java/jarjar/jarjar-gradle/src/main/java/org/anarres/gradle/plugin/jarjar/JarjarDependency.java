/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.anarres.gradle.plugin.jarjar;

import java.util.Collections;
import java.util.Set;
import javax.annotation.Nonnull;
import org.gradle.api.Buildable;
import org.gradle.api.Task;
import org.gradle.api.artifacts.Dependency;
import org.gradle.api.internal.artifacts.ResolvableDependency;
import org.gradle.api.tasks.TaskDependency;

/**
 *
 * @author shevek
 */
public class JarjarDependency implements /* ResolvableDependency, */ Dependency, Buildable {

    private final Dependency delegate;

    public JarjarDependency(@Nonnull Dependency delegate) {
        this.delegate = delegate;
    }

    @Override
    public String getGroup() {
        return delegate.getGroup();
    }

    @Override
    public String getName() {
        return delegate.getName();
    }

    @Override
    public String getVersion() {
        return delegate.getVersion();
    }

    @Override
    public boolean contentEquals(Dependency d) {
        while (d instanceof JarjarDependency)
            d = ((JarjarDependency) d).delegate;
        return delegate.contentEquals(d);
    }

    @Override
    public Dependency copy() {
        return new JarjarDependency(delegate);
    }

    @Override
    public TaskDependency getBuildDependencies() {
        if (delegate instanceof Buildable)
            return ((Buildable) delegate).getBuildDependencies();
        return new TaskDependency() {
            @Override
            public Set<? extends Task> getDependencies(Task task) {
                return Collections.emptySet();
            }
        };
    }
}
