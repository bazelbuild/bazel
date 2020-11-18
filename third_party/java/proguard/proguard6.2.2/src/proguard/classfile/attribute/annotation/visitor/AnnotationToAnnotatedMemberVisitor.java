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
package proguard.classfile.attribute.annotation.visitor;

import proguard.classfile.*;
import proguard.classfile.attribute.annotation.Annotation;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.classfile.visitor.MemberVisitor;


/**
 * This AnnotationVisitor delegates all visits to a given MemberVisitor.
 * The latter visits the class member of each visited class member annotation
 * or method parameter annotation, although never twice in a row.
 *
 * @author Eric Lafortune
 */
public class AnnotationToAnnotatedMemberVisitor
extends      SimplifiedVisitor
implements   AnnotationVisitor
{
    private final MemberVisitor memberVisitor;

    private Member lastVisitedMember;


    public AnnotationToAnnotatedMemberVisitor(MemberVisitor memberVisitor)
    {
        this.memberVisitor = memberVisitor;
    }


    // Implementations for AnnotationVisitor.

    public void visitAnnotation(Clazz clazz, Member member, Annotation annotation)
    {
        if (!member.equals(lastVisitedMember))
        {
            member.accept(clazz, memberVisitor);

            lastVisitedMember = member;
        }
    }
}
