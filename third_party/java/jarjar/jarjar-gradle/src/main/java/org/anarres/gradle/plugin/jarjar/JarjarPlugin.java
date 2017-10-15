/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.anarres.gradle.plugin.jarjar;

import groovy.lang.Closure;
import org.gradle.api.Plugin;
import org.gradle.api.Project;
import org.gradle.api.file.FileCollection;
import org.gradle.api.internal.ClosureBackedAction;
import static org.bouncycastle.asn1.x500.style.RFC4519Style.c;
import static org.gradle.internal.Transformers.name;

/**
 *
 * @author shevek
 */
public class JarjarPlugin implements Plugin<Project> {

    @Override
    public void apply(final Project project) {
        project.getLogger().info("Applying " + this);
        // project.getExtensions().getExtraProperties().set("Jarjar", JarjarTask.class);
        /*
         project.getExtensions().getExtraProperties().set("jarjarDependency", new Closure<FileCollection>(JarjarPlugin.this) {
         @Override
         public FileCollection call(Object... args) {
         JarjarTask jarjar = project.getTasks().create(
         name,
         JarjarTask.class,
         new ClosureBackedAction<JarjarTask>(c));
         return jarjar.getOutputs().getFiles();
         }
         });
         */
        project.getExtensions().create("jarjar", JarjarController.class, project);
    }

}
