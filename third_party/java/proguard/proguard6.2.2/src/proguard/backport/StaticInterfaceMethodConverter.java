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
package proguard.backport;

import proguard.classfile.*;
import proguard.classfile.attribute.visitor.*;
import proguard.classfile.constant.*;
import proguard.classfile.constant.visitor.*;
import proguard.classfile.editor.*;
import proguard.classfile.instruction.Instruction;
import proguard.classfile.instruction.visitor.*;
import proguard.classfile.util.*;
import proguard.classfile.visitor.*;
import proguard.optimize.peephole.*;
import proguard.util.MultiValueMap;

import java.util.*;

/**
 * This ClassVisitor moves all static interface methods in the visited
 * interfaces to a separate util class and updates all invocations in
 * the program class pool.
 *
 * @author Thomas Neidhart
 */
public class StaticInterfaceMethodConverter
extends      SimplifiedVisitor
implements   ClassVisitor
{
    private final ClassPool                     programClassPool;
    private final ClassPool                     libraryClassPool;
    private final MultiValueMap<String, String> injectedClassNameMap;
    private final ClassVisitor                  modifiedClassVisitor;
    private final MemberVisitor                 extraMemberVisitor;


    public StaticInterfaceMethodConverter(ClassPool                     programClassPool,
                                          ClassPool                     libraryClassPool,
                                          MultiValueMap<String, String> injectedClassNameMap,
                                          ClassVisitor                  modifiedClassVisitor,
                                          MemberVisitor                 extraMemberVisitor)
    {
        this.programClassPool     = programClassPool;
        this.libraryClassPool     = libraryClassPool;
        this.injectedClassNameMap = injectedClassNameMap;
        this.modifiedClassVisitor = modifiedClassVisitor;
        this.extraMemberVisitor   = extraMemberVisitor;
    }


    // Implementations for ClassVisitor.

    @Override
    public void visitLibraryClass(LibraryClass libraryClass) {}


    @Override
    public void visitProgramClass(ProgramClass programClass)
    {
        // Collect all static methods of the interface class.
        Set<String> staticMethods = new HashSet<String>();
        programClass.accept(
            new AllMethodVisitor(
            new MemberAccessFilter(ClassConstants.ACC_STATIC, 0,
            new InitializerMethodFilter(null,
            new MemberCollector(false, true, true, staticMethods)))));

        if (!staticMethods.isEmpty())
        {
            ProgramClass utilityClass = createUtilityClass(programClass);

            // Copy all static interface methods to the utility class.
            MemberVisitor memberAdder = new MemberAdder(utilityClass);
            if (extraMemberVisitor != null)
            {
                memberAdder =
                    new MultiMemberVisitor(
                        memberAdder,
                        extraMemberVisitor
                    );
            }

            MemberRemover memberRemover = new MemberRemover();

            programClass.accept(
                new AllMethodVisitor(
                new MemberAccessFilter(ClassConstants.ACC_STATIC, 0,
                new InitializerMethodFilter(null,
                new MultiMemberVisitor(
                    // Add the method to the utility class.
                    memberAdder,

                    // Mark the method for removal from the
                    // interface class.
                    memberRemover
                )
                ))));

            // Add the utility class to the program class pool
            // and the injected class name map.
            programClassPool.addClass(utilityClass);
            injectedClassNameMap.put(programClass.getName(), utilityClass.getName());

            // Change all invokestatic invocations of the static interface
            // methods to use the utility class instead.
            replaceInstructions(programClass, utilityClass, staticMethods);

            // Initialize the hierarchy and references of the utility class.
            utilityClass.accept(
                new MultiClassVisitor(
                    new ClassSuperHierarchyInitializer(programClassPool, libraryClassPool),
                    new ClassReferenceInitializer(programClassPool, libraryClassPool)
                ));

            // Remove the static methods from the interface class and
            // shrink the constant pool of unused constants.
            programClass.accept(
                new MultiClassVisitor(
                    memberRemover,

                    new ConstantPoolShrinker()
                ));
        }
    }


    // Small utility methods.

    private ProgramClass createUtilityClass(ProgramClass interfaceClazz)
    {
        ProgramClass utilityClass =
            new ProgramClass(ClassConstants.CLASS_VERSION_1_2,
                             1,
                             new Constant[10],
                             ClassConstants.ACC_PUBLIC | ClassConstants.ACC_SYNTHETIC,
                             0,
                             0);

        String utilityClassName = interfaceClazz.getName() + "$$Util";
        ConstantPoolEditor constantPoolEditor =
            new ConstantPoolEditor(utilityClass, programClassPool, libraryClassPool);

        utilityClass.u2thisClass =
            constantPoolEditor.addClassConstant(utilityClassName,
                                                utilityClass);
        utilityClass.u2superClass =
            constantPoolEditor.addClassConstant(ClassConstants.NAME_JAVA_LANG_OBJECT,
                                                null);

        SimplifiedClassEditor classEditor =
            new SimplifiedClassEditor(utilityClass);

        // Add a private constructor.
        classEditor.addMethod(ClassConstants.ACC_PRIVATE,
                              ClassConstants.METHOD_NAME_INIT,
                              ClassConstants.METHOD_TYPE_INIT,
                              10)
            .aload_0()
            .invokespecial(ClassConstants.NAME_JAVA_LANG_OBJECT,
                           ClassConstants.METHOD_NAME_INIT,
                           ClassConstants.METHOD_TYPE_INIT)
            .return_();

        classEditor.finishEditing();
        return utilityClass;
    }


    /**
     * Replaces all static invocations of the given methods in the given
     * interface class by invocations of copies of these methods in the
     * given utility class.
     */
    private void replaceInstructions(ProgramClass interfaceClass,
                                     ProgramClass utilityClass,
                                     Set<String>  staticMethods)
    {
        InstructionSequenceBuilder ____ =
            new InstructionSequenceBuilder(programClassPool,
                                           libraryClassPool);

        Instruction[][][] instructions =
            new Instruction[staticMethods.size()][][];

        int index = 0;
        for (String staticMethod : staticMethods)
        {
            String[] splitArray = staticMethod.split("\\.");
            String methodName = splitArray[0];
            String methodDesc = splitArray[1];

            Instruction[][] replacement = new Instruction[][]
            {
                ____.invokestatic_interface(interfaceClass.getName(),
                                            methodName,
                                            methodDesc).__(),

                ____.invokestatic(utilityClass.getName(),
                                  methodName,
                                  methodDesc).__(),
            };

            instructions[index++] = replacement;
        }

        CodeAttributeEditor codeAttributeEditor = new CodeAttributeEditor();

        InstructionVisitor updatedClassVisitor =
            new InstructionToAttributeVisitor(
            new AttributeToClassVisitor(
            modifiedClassVisitor));

        programClassPool.classesAccept(
            new MyReferencedClassFilter(interfaceClass,
            new AllMethodVisitor(
            new AllAttributeVisitor(
            new PeepholeOptimizer(codeAttributeEditor,
            new InstructionSequencesReplacer(____.constants(),
                                             instructions,
                                             null,
                                             codeAttributeEditor,
                                             updatedClassVisitor))))));
    }


    /**
     * This ClassVisitor delegates its visits to classes that
     * reference a given class via any RefConstant.
     */
    private static class MyReferencedClassFilter
    extends    SimplifiedVisitor
    implements ClassVisitor,
               ConstantVisitor
    {
        private final Clazz        referencedClass;
        private final ClassVisitor classVisitor;

        private boolean referenceClassFound;

        public MyReferencedClassFilter(Clazz        referencedClass,
                                       ClassVisitor classVisitor)
        {
            this.referencedClass = referencedClass;
            this.classVisitor    = classVisitor;
        }


        // Implementations for ClassVisitor.

        public void visitProgramClass(ProgramClass programClass)
        {
            referenceClassFound = false;
            programClass.constantPoolEntriesAccept(this);

            if (referenceClassFound)
            {
                programClass.accept(classVisitor);
            }
        }


        // Implementations for ConstantVisitor.

        public void visitAnyConstant(Clazz clazz, Constant constant) {}


        public void visitAnyRefConstant(Clazz clazz, RefConstant refConstant)
        {
            if (refConstant.referencedClass == referencedClass)
            {
                referenceClassFound = true;
            }
        }
    }
}
