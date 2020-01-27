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
package proguard.classfile.util;

import proguard.classfile.*;
import proguard.classfile.visitor.*;

/**
 * This class provides methods to find class members in a given class or in its
 * hierarchy.
 *
 * @author Eric Lafortune
 */
public class MemberFinder
extends      SimplifiedVisitor
implements   MemberVisitor
{
    private static class MemberFoundException extends RuntimeException {}
    private static final MemberFoundException MEMBER_FOUND = new MemberFoundException();

    private Clazz  clazz;
    private Member member;


    /**
     * Finds the field with the given name and descriptor in the given
     * class or its hierarchy.
     */
    public Field findField(Clazz  referencingClass,
                           Clazz  clazz,
                           String name,
                           String descriptor)
    {
        return (Field)findMember(referencingClass, clazz, name, descriptor, true);
    }


    /**
     * Finds the method with the given name and descriptor in the given
     * class or its hierarchy.
     */
    public Method findMethod(Clazz  referencingClass,
                             Clazz  clazz,
                             String name,
                             String descriptor)
    {
        return (Method)findMember(referencingClass, clazz, name, descriptor, false);
    }


    /**
     * Finds the class member with the given name and descriptor in the given
     * class or its hierarchy.
     */
    public Member findMember(Clazz   referencingClass,
                             Clazz   clazz,
                             String  name,
                             String  descriptor,
                             boolean isField)
    {
        // Organize a search in the hierarchy of superclasses and interfaces.
        // The class member may be in a different class, if the code was
        // compiled with "-target 1.2" or higher (the default in JDK 1.4).
        try
        {
            this.clazz  = null;
            this.member = null;
            clazz.hierarchyAccept(true, true, true, false, isField ?
                (ClassVisitor)new NamedFieldVisitor(name, descriptor,
                              new MemberClassAccessFilter(referencingClass, this)) :
                (ClassVisitor)new NamedMethodVisitor(name, descriptor,
                              new MemberClassAccessFilter(referencingClass, this)));
        }
        catch (MemberFoundException ex)
        {
            // We've found the member we were looking for.
        }

        return member;
    }


    /**
     * Returns the corresponding class of the most recently found class
     * member.
     */
    public Clazz correspondingClass()
    {
        return clazz;
    }


    /**
     * Returns whether the given method is overridden anywhere down the class
     * hierarchy.
     */
    public boolean isOverriden(Clazz  clazz,
                               Method method)
    {
        String name       = method.getName(clazz);
        String descriptor = method.getDescriptor(clazz);

        // Go looking for the method down the class hierarchy.
        try
        {
            this.clazz  = null;
            this.member = null;

            clazz.hierarchyAccept(false, false, false, true,
                new NamedMethodVisitor(name, descriptor,
                new MemberAccessFilter(0, ClassConstants.ACC_PRIVATE, this)));
        }
        catch (MemberFoundException ex)
        {
            // We've found an overriding method.
            return true;
        }

        return false;
    }


    /**
     * Returns whether the given field is shadowed anywhere down the class
     * hierarchy.
     */
    public boolean isShadowed(Clazz clazz,
                              Field field)
    {
        String name       = field.getName(clazz);
        String descriptor = field.getDescriptor(clazz);

        // Go looking for the field down the class hierarchy.
        try
        {
            this.clazz  = null;
            this.member = null;
            clazz.hierarchyAccept(false, false, false, true,
                new NamedFieldVisitor(name, descriptor,
                new MemberAccessFilter(0, ClassConstants.ACC_PRIVATE, this)));
        }
        catch (MemberFoundException ex)
        {
            // We've found a shadowing field.
            return true;
        }

        return false;
    }


//    // Implementations for ClassVisitor.
//
//    private void visitAnyClass(Clazz clazz)
//    {
//        if (member == null)
//        {
//            member = isField ?
//                (Member)clazz.findField(name, descriptor) :
//                (Member)clazz.findMethod(name, descriptor);
//
//            if (member != null)
//            {
//                this.clazz = clazz;
//            }
//        }
//    }


    // Implementations for MemberVisitor.

    public void visitAnyMember(Clazz clazz, Member member)
    {
        this.clazz  = clazz;
        this.member = member;

        throw MEMBER_FOUND;
    }
}
