/*
 * ProGuard -- shrinking, optimization, obfuscation, and preverification
 *             of Java bytecode.
 *
 * Copyright (c) 2002-2019 Guardsquare NV
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */
package proguard.configuration;


import proguard.*;
import proguard.classfile.*;
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.visitor.*;
import proguard.classfile.editor.*;
import proguard.classfile.instruction.Instruction;
import proguard.classfile.instruction.visitor.InstructionVisitor;
import proguard.classfile.util.*;
import proguard.classfile.visitor.*;
import proguard.io.*;
import proguard.optimize.peephole.*;
import proguard.util.MultiValueMap;

import java.io.IOException;

import static proguard.classfile.util.ClassUtil.internalClassName;

/**
 * This class can add configuration debug logging code to all code that
 * relies on reflection. The added code prints suggestions on which keep
 * rules to add to ensure the reflection code will continue working after
 * obfuscation and shrinking.
 *
 * @author Johan Leys
 */
public class ConfigurationLoggingAdder
extends      SimplifiedVisitor
implements   // Implementation interfaces.
             InstructionVisitor
{
    private final Configuration configuration;

    // Field acting as parameter for the visitor methods.
    private  MultiValueMap<String, String> injectedClassMap;


    /**
     * Creates a new ConfigurationLoggingAdder.
     */
    public ConfigurationLoggingAdder(Configuration configuration)
    {
        this.configuration = configuration;
    }


    /**
     * Instruments the given program class pool.
     */
    public void execute(ClassPool                     programClassPool,
                        ClassPool                     libraryClassPool,
                        MultiValueMap<String, String> injectedClassMap )
    {
        // Load the logging utility classes in the program class pool.
        // TODO: The initialization could be incomplete if the loaded classes depend on one another.
        ClassReader classReader =
            new ClassReader(false, false, false, null,
            new MultiClassVisitor(
                new ClassPoolFiller(programClassPool),
                new ClassReferenceInitializer(programClassPool, libraryClassPool),
                new ClassSubHierarchyInitializer()
            ));

        try
        {
            classReader.read(new ClassPathDataEntry(ConfigurationLogger.MethodSignature.class));
            classReader.read(new ClassPathDataEntry(ConfigurationLogger.class));
        }
        catch (IOException e)
        {
            throw new RuntimeException(e);
        }

        // Set up the instruction sequences and their replacements.
        ConfigurationLoggingInstructionSequenceConstants constants =
             new ConfigurationLoggingInstructionSequenceConstants(programClassPool,
                                                                  libraryClassPool);

        BranchTargetFinder  branchTargetFinder  = new BranchTargetFinder();
        CodeAttributeEditor codeAttributeEditor = new CodeAttributeEditor();

        // Set the injected class map for the extra visitor.
        this.injectedClassMap = injectedClassMap;

        // Replace the instruction sequences in all non-ProGuard classes.
        programClassPool.classesAccept(
            new ClassNameFilter("!proguard/**",
            new AllMethodVisitor(
            new AllAttributeVisitor(
            new PeepholeOptimizer(branchTargetFinder, codeAttributeEditor,
            new ConfigurationLoggingInstructionSequencesReplacer(constants.CONSTANTS,
                                                                 constants.RESOURCE,
                                                                 branchTargetFinder,
                                                                 codeAttributeEditor,
                                                                 this))))));
    }


    // Implementations for InstructionVisitor.

    public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction)
    {
        // Add a dependency from the modified class on the logging class.
        injectedClassMap.put(clazz.getName(), internalClassName(ConfigurationLogger.class.getName()));
        injectedClassMap.put(clazz.getName(), internalClassName(ConfigurationLogger.MethodSignature.class.getName()));
    }
}
