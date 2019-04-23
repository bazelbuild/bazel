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
package proguard.classfile.editor;

import proguard.classfile.*;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.classfile.visitor.*;

import java.util.*;


/**
 * This visitor removes all members it visits in a ProgramClass.
 *
 * It should be used in two steps:
 * - in the first step, the collection step, all program fields to be removed
 *   should be visited.
 * - in the second step, the removal step, the program class containing the
 *   program fields should be visited. This will actually delete all
 *   collected fields.
 *
 * For example, to remove all fields in a program class:
 *
 *   MemberRemover remover = new MemberRemover();
 *   programClass.fieldsAccept(remover);
 *   programClass.accept(remover);
 *
 * @author Johan Leys
 */
public class MemberRemover
extends      SimplifiedVisitor
implements   ClassVisitor,
             MemberVisitor
{
    private Set<Method> methodsToRemove = new HashSet<Method>();
    private Set<Field>  fieldsToRemove  = new HashSet<Field>();


    // Implementations for ClassVisitor.

    public void visitAnyClass(Clazz clazz) {}


    public void visitProgramClass(ProgramClass programClass)
    {
        ClassEditor classEditor = new ClassEditor(programClass);

        // Remove all collected methods.
        for (Method method : methodsToRemove) {
            classEditor.removeMethod(method);
        }
        methodsToRemove.clear();

        // Remove all collected fields.
        for (Field field : fieldsToRemove) {
            classEditor.removeField(field);
        }
        fieldsToRemove.clear();
    }


    // Implementations for MemberVisitor.

    public void visitAnyMember(Clazz clazz, Member member) {}


    public void visitProgramField(ProgramClass programClass, ProgramField programField)
    {
        fieldsToRemove.add(programField);
    }


    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        methodsToRemove.add(programMethod);
    }
}
