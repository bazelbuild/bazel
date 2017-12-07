/*
 * ProGuard -- shrinking, optimization, obfuscation, and preverification
 *             of Java bytecode.
 *
 * Copyright (c) 2002-2017 Eric Lafortune @ GuardSquare
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
import proguard.classfile.attribute.Attribute;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.classfile.visitor.MemberVisitor;

/**
 * This MemberVisitor copies all class members that it visits to the given
 * target class. Their visitor info is set to the class members from which they
 * were copied.
 *
 * @author Eric Lafortune
 */
public class MemberAdder
extends      SimplifiedVisitor
implements   MemberVisitor
{
    //*
    private static final boolean DEBUG = false;
    /*/
    private static       boolean DEBUG = true;
    //*/


    private static final Attribute[] EMPTY_ATTRIBUTES = new Attribute[0];


    private final ProgramClass  targetClass;
//  private final boolean       addFields;
    private final MemberVisitor extraMemberVisitor;

    private final ConstantAdder      constantAdder;
    private final ClassEditor        classEditor;
    private final ConstantPoolEditor constantPoolEditor;


    /**
     * Creates a new MemberAdder that will copy methods into the given target
     * class.
     * @param targetClass the class to which all visited class members will be
     *                    added.
     */
    public MemberAdder(ProgramClass targetClass)
    {
        this(targetClass, null);
    }


    /**
     * Creates a new MemberAdder that will copy methods into the given target
     * class.
     * @param targetClass        the class to which all visited class members
     *                           will be added.
     * @param extraMemberVisitor an optional member visitor that visits each
     *                           new member right after it has been added. This
     *                           allows changing the visitor info, for instance.
     */
//     * @param addFields   specifies whether fields should be added, or fused
//     *                    with the present fields.
    public MemberAdder(ProgramClass  targetClass,
//                     boolean       addFields,
                       MemberVisitor extraMemberVisitor)
    {
        this.targetClass        = targetClass;
//      this.addFields          = addFields;
        this.extraMemberVisitor = extraMemberVisitor;

        constantAdder      = new ConstantAdder(targetClass);
        classEditor        = new ClassEditor(targetClass);
        constantPoolEditor = new ConstantPoolEditor(targetClass);
    }


    // Implementations for MemberVisitor.

    public void visitProgramField(ProgramClass programClass, ProgramField programField)
    {
        //String name        = programField.getName(programClass);
        //String descriptor  = programField.getDescriptor(programClass);
        int    accessFlags = programField.getAccessFlags();

        // TODO: Handle field with the same name and descriptor in the target class.
        // We currently avoid this case, since renaming the identical field
        // still causes confused field references.
        //// Does the target class already have such a field?
        //ProgramField targetField = (ProgramField)targetClass.findField(name, descriptor);
        //if (targetField != null)
        //{
        //    // Is the field private or static?
        //    int targetAccessFlags = targetField.getAccessFlags();
        //    if ((targetAccessFlags &
        //         (ClassConstants.ACC_PRIVATE |
        //          ClassConstants.ACC_STATIC)) != 0)
        //    {
        //        if (DEBUG)
        //        {
        //            System.out.println("MemberAdder: renaming field ["+targetClass+"."+targetField.getName(targetClass)+" "+targetField.getDescriptor(targetClass)+"]");
        //        }
        //
        //        // Rename the private or static field.
        //        targetField.u2nameIndex =
        //            constantPoolEditor.addUtf8Constant(newUniqueMemberName(name, targetClass.getName()));
        //    }
        //    else
        //    {
        //        // Keep the non-private and non-static field, but update its
        //        // contents, in order to keep any references to it valid.
        //        if (DEBUG)
        //        {
        //            System.out.println("MemberAdder: updating field ["+programClass+"."+programField.getName(programClass)+" "+programField.getDescriptor(programClass)+"] into ["+targetClass.getName()+"]");
        //        }
        //
        //        // Combine the access flags.
        //        targetField.u2accessFlags = accessFlags | targetAccessFlags;
        //
        //        // Add and replace any attributes.
        //        programField.attributesAccept(programClass,
        //                                      new AttributeAdder(targetClass,
        //                                                         targetField,
        //                                                         true));
        //
        //        // Don't add a new field.
        //        return;
        //    }
        //}

        if (DEBUG)
        {
            System.out.println("MemberAdder: copying field ["+programClass+"."+programField.getName(programClass)+" "+programField.getDescriptor(programClass)+"] into ["+targetClass.getName()+"]");
        }

        // Create a copy of the field.
        ProgramField newProgramField =
            new ProgramField(accessFlags,
                             constantAdder.addConstant(programClass, programField.u2nameIndex),
                             constantAdder.addConstant(programClass, programField.u2descriptorIndex),
                             0,
                             programField.u2attributesCount > 0 ?
                                 new Attribute[programField.u2attributesCount] :
                                 EMPTY_ATTRIBUTES,
                             programField.referencedClass);

        // Link to its visitor info.
        newProgramField.setVisitorInfo(programField);

        // Copy its attributes.
        programField.attributesAccept(programClass,
                                      new AttributeAdder(targetClass,
                                                         newProgramField,
                                                         false));

        // Add the completed field.
        classEditor.addField(newProgramField);

        // Visit the newly added field, if necessary.
        if (extraMemberVisitor != null)
        {
            extraMemberVisitor.visitProgramField(targetClass, newProgramField);
        }
    }


    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        String name        = programMethod.getName(programClass);
        String descriptor  = programMethod.getDescriptor(programClass);
        int    accessFlags = programMethod.getAccessFlags();

        // Does the target class already have such a method?
        ProgramMethod targetMethod = (ProgramMethod)targetClass.findMethod(name, descriptor);
        if (targetMethod != null)
        {
            // is this source method abstract?
            if ((accessFlags & ClassConstants.ACC_ABSTRACT) != 0)
            {
                // Keep the target method.
                if (DEBUG)
                {
                    System.out.println("MemberAdder: skipping abstract method ["+programClass.getName()+"."+programMethod.getName(programClass)+programMethod.getDescriptor(programClass)+"] into ["+targetClass.getName()+"]");
                }

                // Don't add a new method.
                return;
            }

            // Is the target method abstract?
            int targetAccessFlags = targetMethod.getAccessFlags();
            if ((targetAccessFlags & ClassConstants.ACC_ABSTRACT) != 0)
            {
                // Keep the abstract method, but update its contents, in order
                // to keep any references to it valid.
                if (DEBUG)
                {
                    System.out.println("MemberAdder: updating method ["+programClass.getName()+"."+programMethod.getName(programClass)+programMethod.getDescriptor(programClass)+"] into ["+targetClass.getName()+"]");
                }

                // Replace the access flags.
                targetMethod.u2accessFlags =
                    accessFlags & ~ClassConstants.ACC_FINAL;

                // Add and replace the attributes.
                programMethod.attributesAccept(programClass,
                                               new AttributeAdder(targetClass,
                                                                  targetMethod,
                                                                  true));

                // Don't add a new method.
                return;
            }

            if (DEBUG)
            {
                System.out.println("MemberAdder: renaming method ["+targetClass.getName()+"."+targetMethod.getName(targetClass)+targetMethod.getDescriptor(targetClass)+"]");
            }

            // TODO: Handle non-abstract method with the same name and descriptor in the target class.
            // We currently avoid this case, since renaming the identical method
            // still causes confused method references.
            //// Rename the private (non-abstract) or static method.
            //targetMethod.u2nameIndex =
            //    constantPoolEditor.addUtf8Constant(newUniqueMemberName(name, descriptor));
        }

        if (DEBUG)
        {
            System.out.println("MemberAdder: copying method ["+programClass.getName()+"."+programMethod.getName(programClass)+programMethod.getDescriptor(programClass)+"] into ["+targetClass.getName()+"]");
        }

        // Create a copy of the method.
        ProgramMethod newProgramMethod =
            new ProgramMethod(accessFlags & ~ClassConstants.ACC_FINAL,
                              constantAdder.addConstant(programClass, programMethod.u2nameIndex),
                              constantAdder.addConstant(programClass, programMethod.u2descriptorIndex),
                              0,
                              programMethod.u2attributesCount > 0 ?
                                  new Attribute[programMethod.u2attributesCount] :
                                  EMPTY_ATTRIBUTES,
                              programMethod.referencedClasses != null ?
                                  (Clazz[])programMethod.referencedClasses.clone() :
                                  null);

        // Link to its visitor info.
        newProgramMethod.setVisitorInfo(programMethod);

        // Copy its attributes.
        programMethod.attributesAccept(programClass,
                                       new AttributeAdder(targetClass,
                                                          newProgramMethod,
                                                          false));

        // Add the completed method.
        classEditor.addMethod(newProgramMethod);

        // Visit the newly added method, if necessary.
        if (extraMemberVisitor != null)
        {
            extraMemberVisitor.visitProgramMethod(targetClass, newProgramMethod);
        }
    }


    // Small utility methods.

    /**
     * Returns a unique class member name, based on the given name and descriptor.
     */
    private String newUniqueMemberName(String name, String descriptor)
    {
        return name.equals(ClassConstants.METHOD_NAME_INIT) ?
            ClassConstants.METHOD_NAME_INIT :
            name + ClassConstants.SPECIAL_MEMBER_SEPARATOR + Long.toHexString(Math.abs((descriptor).hashCode()));
    }
}
